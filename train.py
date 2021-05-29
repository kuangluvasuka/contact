import time
from typing import List, Dict

import numpy as np
import tensorflow as tf
from feeder import Feeder
from utils import configure_summary, initialize_checkpoint, log_results, time_string
from evaluate_metric import convert_contact_map, evaluation_metrics


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
    logits = model(data_dict)
    preds = [np.multiply(convert_contact_map(x), y) for x, y in zip(logits.numpy(), data_dict['mask_2d'].numpy())]
    trues = [x for x in data_dict['contact_map'].numpy()]    # convert from array(B, N, N) to list of arr(N, N)

    contact_preds.extend(preds)
    contact_trues.extend(trues)

  return evaluation_metrics(contact_preds, contact_trues), contact_preds, contact_trues


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

      def _train_step_fn(inp):
        """Replicated training step."""

        with tf.GradientTape() as tape:
          logits = model(inp)
          loss = loss_fn(inp['contact_map'], logits, inp['mask_2d'])
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

      per_replica_loss = strategy.run(_train_step_fn, args=(inputs,))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    @tf.function(experimental_relax_shapes=True)
    def test_step(inputs: Dict) -> tf.Tensor:
      """Perform a distributed testing step."""

      def _test_step_fn(inp):
        """Replicated testing step."""

        logits = model(inp)
        loss = loss_fn(inp['contact_map'], logits, inp['mask_2d'])
        return loss, logits

      per_replica_loss, pr_logit = strategy.run(_test_step_fn, args=(inputs,))
      return [strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None),
              strategy.gather(pr_logit, axis=0)]


    # train and valid
    current_epoch = checkpoint_manager.checkpoint.epoch.numpy()
    print("Training started from Epoch: {}".format(current_epoch))
    for epoch in range(current_epoch, hp['epochs'] + 1):
      checkpoint_manager.checkpoint.epoch.assign_add(1)
      losses = []
      start = time.time()
      tf.summary.experimental.set_step(epoch)
      with summary_writer['train'].as_default():
        for (i, data_dict) in enumerate(feeder.train):
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
            loss, logit = test_step(data_dict)                    # logit [B, L, L]
          losses.append(loss.numpy())

          # gather necessary items from distributed replica
          y_true = strategy.gather(data_dict['contact_map'], axis=0)
          mask = strategy.gather(data_dict['mask_2d'], axis=0)
          preds.extend([np.multiply(convert_contact_map(x), y) for x, y in zip(logit.numpy(), mask.numpy())])
          trues.extend([x for x in y_true.numpy()])                       # list of array(N, N)

        valid_average_loss = np.mean(losses)
        tf.summary.scalar('loss', valid_average_loss)

        results = evaluation_metrics(preds, trues)
        log_results(results)

      print("Epoch: {} | train average loss: {:.3f} | time: {:.2f}s | valid average loss: {:.3f})".format(
          epoch, train_average_loss, time.time() - start, valid_average_loss))

      if epoch % hp['checkpoint_inteval'] == 0:
        save_path = checkpoint_manager.save()
        print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))
