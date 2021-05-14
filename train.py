import os
import time
from typing import Union, List, Dict, Callable, Tuple

import numpy as np
import tensorflow as tf
from utils import time_string
from feeder import Feeder
from evaluate_metric import convert_contact_map, partition_contacts, collect_metrics


def configure_checkpoint(model: tf.Tensor,
                         optimizer: tf.keras.optimizers.Optimizer,
                         checkpoint_dir: str,
                         resume_training: bool = False) -> Tuple[tf.train.Checkpoint, tf.train.CheckpointManager]:

  checkpoint = tf.train.Checkpoint(
      epoch=tf.Variable(1),
      step=tf.Variable(1),
      model=model,
      optimizer=optimizer)
  manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
  if resume_training:
    checkpoint.restore(manager.latest_checkpoint).assert_consumed()
    print("Checkpoint restored from {}.".format(manager.latest_checkpoint))
  else:
    print("Initializing checkpoint object at {}.".format(checkpoint_dir))
  return checkpoint, manager


def configure_summary(summary_dir: Union[str, None]) -> Dict:
  if summary_dir is None:
    return {'train': tf.summary.create_noop_writer(), 'valid': tf.summary.create_noop_writer()}
  logdir = os.path.join(summary_dir, time_string())
  train_log = os.path.join(logdir, 'train')
  valid_log = os.path.join(logdir, 'valid')
  os.makedirs(train_log, exist_ok=True)
  os.makedirs(valid_log, exist_ok=True)
  return {'train': tf.summary.create_file_writer(train_log), 'valid': tf.summary.create_file_writer(valid_log)}


#@tf.function
def _train_on_step(model: tf.keras.Model,
                   inputs: Dict,
                   loss_fn: Callable,
                   optimizer: tf.keras.optimizers.Optimizer,
                   range_wt_mat: tf.Tensor) -> tf.Tensor:

  with tf.GradientTape() as tape:
    logits = model(inputs['primary'])
    loss = loss_fn(inputs['contact_map'], logits, inputs['mask_2D'], range_wt_mat)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss


#@tf.function
def _train_off_step(model: tf.keras.Model,
                    inputs: Dict,
                    loss_fn: Callable,
                    range_wt_mat: tf.Tensor) -> tf.Tensor:

  logits = model(inputs['primary'])
  loss = loss_fn(inputs['contact_map'], logits, inputs['mask_2D'], range_wt_mat)
  return loss


def masked_weighted_cross_entropy(y_trues: tf.Tensor,
                                  logits: tf.Tensor,
                                  masks: tf.Tensor,
                                  weight) -> tf.Tensor:

  loss = tf.nn.weighted_cross_entropy_with_logits(y_trues, logits, pos_weight=1)
  masked_loss = tf.multiply(loss, masks)
  length = loss.shape[1]
  weighted_loss = tf.multiply(masked_loss, weight[: length, : length])

  return tf.reduce_sum(tf.reduce_mean(weighted_loss, axis=0))


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


def evaluate(model: tf.keras.Model,
             feeder: Feeder,
             split_str: str = 'test'):

  loader = feeder.test
  if split_str == 'valid':
    loader = feeder.valid

  contact_preds = []
  contact_trues = []
  for (i, data_dict) in enumerate(loader):
    logits = model(data_dict['primary'])
    preds = [convert_contact_map(x) for x in logits.numpy()]
    trues = [x for x in data_dict['contact_map'].numpy()]    # convert from array(B, N, N) to list of arr(N, N)
    masks = data_dict['mask_2D'].numpy()
    masked_preds = [np.multiply(x, y) for x, y in zip(masks, preds)]
    masked_trues = [np.multiply(x, y) for x, y in zip(masks, trues)]

    contact_preds.extend(masked_preds)
    contact_trues.extend(masked_trues)

  partitioned_preds = [partition_contacts(x) for x in contact_preds]
  partitioned_trues = [partition_contacts(x) for x in contact_trues]

  short_preds = list(zip(*partitioned_preds))[0]
  short_trues = list(zip(*partitioned_trues))[0]

  precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5 = collect_metrics(short_trues, short_preds)

  return list(map(np.mean, [precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5]))



def train(model: tf.keras.Model,
          feeder: Feeder,
          hparams: Dict) -> None:

  hp = hparams
  range_weighted_mat = get_range_weighted_matrix(hp['range_weights'], hp['max_protein_length'])
  loss_fn = masked_weighted_cross_entropy
  optimizer = tf.optimizers.Adam(learning_rate=hp['learning_rate'])
  ckpt_obj, ckpt_mgr = configure_checkpoint(model, optimizer, hp['checkpoint_dir'], hp['resume_training'])
  summary_writer = configure_summary(hp['summary_dir'])

  for epoch in range(int(ckpt_obj.epoch), hp['epochs'] + 1):
    ckpt_obj.epoch.assign_add(1)
    losses = []
    start = time.time()
    tf.summary.experimental.set_step(epoch)
    with summary_writer['train'].as_default():
      for (i, data_dict) in enumerate(feeder.train):
        ckpt_obj.step.assign_add(1)
        # TODO: add summary_step?
        with tf.summary.record_if(i == 0):
          loss = _train_on_step(model, data_dict, loss_fn, optimizer, range_weighted_mat)
        losses.append(loss.numpy())
      train_loss_average = np.mean(losses) / hp['batch_size']
      tf.summary.scalar('loss', train_loss_average)

    losses = []
    with summary_writer['valid'].as_default():
      for (i, data_dict) in enumerate(feeder.valid):
        with tf.summary.record_if(i == 0):
          loss = _train_off_step(model, data_dict, loss_fn, range_weighted_mat)
        losses.append(loss.numpy())
      valid_loss_average = np.mean(losses) / hp['test_batch_size']
      tf.summary.scalar('loss', valid_loss_average)

      precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5 = evaluate(model, feeder, split_str='valid')
      tf.summary.scalar('precision', precision)
      tf.summary.scalar('recall', recall)
      tf.summary.scalar('f1', f1)
      tf.summary.scalar('aupr', aupr)
      tf.summary.scalar('precision_L', precision_L)
      tf.summary.scalar('precision_L_2', precision_L_2)
      tf.summary.scalar('precision_L_5', precision_L_5)

    print("Epoch: {} | train loss: {:.3f} | time: {:.2f}s | valid loss: {:.3f})".format(
        epoch, train_loss_average, time.time() - start, valid_loss_average))


    if epoch % hp['checkpoint_inteval'] == 0:
      ckpt_mgr.save()
    #  #save_path = ckpt_mgr.save()
    #  #print("Saved checkpoint for epoch {} at {}".format(epoch, save_path))



