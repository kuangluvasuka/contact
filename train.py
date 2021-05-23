import os
import time
from typing import Union, List, Dict, Tuple

import numpy as np
import tensorflow as tf
from utils import time_string
from feeder import Feeder
from evaluate_metric import convert_contact_map, partition_contacts, collect_metrics


def initialize_checkpoint(model: tf.Tensor,
                          optimizer: tf.keras.optimizers.Optimizer,
                          checkpoint_dir: str) -> tf.train.CheckpointManager:

  checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1),
                                   step=tf.Variable(1),
                                   model=model,
                                   optimizer=optimizer)
  manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
  return manager


def configure_summary(summary_dir: Union[str, None]) -> Dict:
  if summary_dir is None:
    return {'train': tf.summary.create_noop_writer(), 'valid': tf.summary.create_noop_writer()}
  logdir = os.path.join(summary_dir, time_string())
  train_log = os.path.join(logdir, 'train')
  valid_log = os.path.join(logdir, 'valid')
  os.makedirs(train_log, exist_ok=True)
  os.makedirs(valid_log, exist_ok=True)
  return {'train': tf.summary.create_file_writer(train_log), 'valid': tf.summary.create_file_writer(valid_log)}


def masked_weighted_cross_entropy(range_weighted_mat):
  """Compute cross entropy with sequence mask and range weight."""

  def func(y_trues: tf.Tensor, logits: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
    loss = tf.nn.weighted_cross_entropy_with_logits(y_trues, logits, pos_weight=1)
    masked_loss = tf.multiply(loss, masks)
    length = tf.shape(loss)[1]
    weighted_loss = tf.multiply(masked_loss, range_weighted_mat[: length, : length])        # [B, L, L]
    return tf.nn.compute_average_loss(weighted_loss)

  return func


def get_range_weighted_matrix(range_wt: List, max_len: int) -> tf.Tensor:
  """
    Create a weighted symmetric matrix for close-short-medium-long range of contacts.
    The range is defined as: 0~5, 6~11, 12~23, 24~max
    Note: max_len > 24
  """
  mat = np.zeros([max_len, max_len])
  row = np.array([range_wt[0]] * 6 + [range_wt[1]] * 6 + [range_wt[2]] * 12 + [range_wt[3]] * (max_len - 24))
  for i in range(max_len):
    mat[i] = row
    row = np.roll(row, shift=1)
    row[0] = 0
  res = mat + mat.T
  np.fill_diagonal(res, range_wt[0])
  return tf.constant(res, dtype=tf.float32)


def evaluate(model: tf.keras.Model, test_loader: tf.data.Dataset):

  #TODO: check if dataset is test set

  contact_preds = []
  contact_trues = []
  for (i, data_dict) in enumerate(test_loader):
    logits = model(data_dict['primary'])
    preds = [convert_contact_map(x) for x in logits.numpy()]
    trues = [x for x in data_dict['contact_map'].numpy()]    # convert from array(B, N, N) to list of arr(N, N)
    masks = data_dict['mask_2d'].numpy()
    masked_preds = [np.multiply(x, y) for x, y in zip(masks, preds)]

    #TODO: delete
    #masked_trues = [np.multiply(x, y) for x, y in zip(masks, trues)]
    masked_trues = trues

    contact_preds.extend(masked_preds)
    contact_trues.extend(masked_trues)

  return evaluation_metrics(contact_preds, contact_trues), contact_preds, contact_trues


def evaluation_metrics(preds, trues):
  partitioned_preds = [partition_contacts(x) for x in preds]
  partitioned_trues = [partition_contacts(x) for x in trues]

  short_preds = list(zip(*partitioned_preds))[0]
  short_trues = list(zip(*partitioned_trues))[0]

  #TODO: add medium and long evaluations

  precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5 = collect_metrics(short_trues, short_preds)

  return list(map(np.mean, [precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5]))


def train(model: tf.keras.Model,
          feeder: Feeder,
          optimizer: tf.optimizers.Optimizer,
          strategy: tf.distribute.Strategy,
          hparams: Dict) -> None:

  hp = hparams
  summary_writer = configure_summary(hp['summary_dir'])
  range_weighted_mat = get_range_weighted_matrix(hp['range_weights'], hp['max_protein_length'])

  with strategy.scope():
    loss_fn = masked_weighted_cross_entropy(range_weighted_mat)

    checkpoint_manager = initialize_checkpoint(model, optimizer, hp['checkpoint_dir'])

    if hp['resume_training']:
      checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).assert_existing_objects_matched()
      print("Checkpoint restored from {}.".format(checkpoint_manager.latest_checkpoint))

    @tf.function(experimental_relax_shapes=True)
    def train_step(inputs: Dict) -> tf.Tensor:
      """Perform a distributed training step."""

      def _train_step_fn(x, y_true, mask):
        """Replicated training step."""

        with tf.GradientTape() as tape:
          logits = model(x)
          loss = loss_fn(y_true, logits, mask)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

      per_replica_loss = strategy.run(_train_step_fn, args=(inputs['primary'],
                                                            inputs['contact_map'],
                                                            inputs['mask_2d']))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    @tf.function(experimental_relax_shapes=True)
    def test_step(inputs: Dict) -> tf.Tensor:
      """Perform a distributed testing step."""

      def _test_step_fn(x, y_true, mask):
        """Replicated testing step."""


        #TODO: evaluate & test_step inconsistancy

        logits = model(x)
        masked_logits = tf.multiply(logits, mask)
        loss = loss_fn(y_true, logits, mask)
        return loss, masked_logits, y_true

      per_replica_loss, pr_logit, pr_y_true = strategy.run(_test_step_fn, args=(inputs['primary'],
                                                                                inputs['contact_map'],
                                                                                inputs['mask_2d']))
      #TODO: check here

      return [strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None),
              strategy.gather(pr_logit, axis=0),
              strategy.gather(pr_y_true, axis=0)]


    ##
    current_epoch = checkpoint_manager.checkpoint.epoch.numpy()
    print("Training started from Epoch: {}".format(current_epoch))
    for epoch in range(current_epoch, hp['epochs'] + 1):
      checkpoint_manager.checkpoint.epoch.assign_add(1)
      losses = []
      start = time.time()
      tf.summary.experimental.set_step(epoch)
      with summary_writer['train'].as_default():
        for (i, data_dict) in enumerate(feeder.valid):
          checkpoint_manager.checkpoint.step.assign_add(1)
          # TODO: add summary_step?
          with tf.summary.record_if(i == 0):
            loss = train_step(data_dict)
          losses.append(loss.numpy())
        train_average_loss = np.mean(losses)
        tf.summary.scalar('loss', train_average_loss)

      losses = []
      preds = []
      trues = []
      with summary_writer['valid'].as_default():
        for (i, data_dict) in enumerate(feeder.valid):
          with tf.summary.record_if(i == 0):
            loss, logit, y_true = test_step(data_dict)                    # logit [B, L, L]
          losses.append(loss.numpy())
          preds.extend([convert_contact_map(x) for x in logit.numpy()])
          trues.extend([x for x in y_true.numpy()])                       # list of array(N, N)

        valid_average_loss = np.mean(losses)
        tf.summary.scalar('loss', valid_average_loss)

        precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5 = evaluation_metrics(preds, trues)
        tf.summary.scalar('precision', precision)
        tf.summary.scalar('recall', recall)
        tf.summary.scalar('f1', f1)
        tf.summary.scalar('aupr', aupr)
        tf.summary.scalar('precision_L', precision_L)
        tf.summary.scalar('precision_L_2', precision_L_2)
        tf.summary.scalar('precision_L_5', precision_L_5)

      print("Epoch: {} | train average loss: {:.3f} | time: {:.2f}s | valid average loss: {:.3f})".format(
          epoch, train_average_loss, time.time() - start, valid_average_loss))

      #TODO: delete
      print(precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5)

      if epoch % hp['checkpoint_inteval'] == 0:
        save_path = checkpoint_manager.save()
        print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))



