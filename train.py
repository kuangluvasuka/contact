import os
from typing import List, Dict, Any, Callable
import numpy as np
import tensorflow as tf

from utils import time_string
from evaluate_metric import convert_contact_map, evaluation_metrics


# TODO: refacor this function by subclassing tf.keras.losses.Loss!!!!!!!
def masked_weighted_cross_entropy(range_weighted_mat: tf.Tensor) -> Callable:
  """Compute cross entropy with sequence mask and range weight."""

  def func(y_trues: tf.Tensor, logits: tf.Tensor, masks: tf.Tensor, pos_weight: int = 1) -> tf.Tensor:
    loss = tf.nn.weighted_cross_entropy_with_logits(y_trues, logits, pos_weight=pos_weight)
    masked_loss = tf.multiply(loss, masks)
    length = tf.shape(loss)[1]
    weighted_loss = tf.multiply(masked_loss, range_weighted_mat[: length, : length])        # [B, L, L]
    return tf.reduce_sum(weighted_loss, axis=[1, 2])

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
  lengths = []
  for (i, data_dict) in enumerate(test_loader):
    logits = model(data_dict)
    preds = [np.multiply(convert_contact_map(x), y) for x, y in zip(logits.numpy(), data_dict['mask_2d'].numpy())]
    trues = [x for x in data_dict['contact_map'].numpy()]    # convert from array(B, N, N) to list of arr(N, N)
    length = [x for x in data_dict['protein_length'].numpy()]

    contact_preds.extend(preds)
    contact_trues.extend(trues)
    lengths.extend(length)

  return evaluation_metrics(contact_preds, contact_trues, lengths), contact_preds, contact_trues


class Train():
  def __init__(self,
               model: tf.keras.Model,
               optimizer: tf.optimizers.Optimizer,
               strategy: tf.distribute.Strategy,
               hp: Dict[str, Any],
               tf_summary_dir: str,
               checkpoint_dir: str):

    self.model = model
    self.optimizer = optimizer
    self.loss_fn = masked_weighted_cross_entropy(get_range_weighted_matrix(hp['range_weights'], hp['max_protein_length']))
    self.strategy = strategy

    self._train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    self._valid_loss_metric = tf.keras.metrics.Mean(name='test_loss')

    self._pos_weight = hp['pos_weight']
    self._tf_summary_dir = tf_summary_dir
    self._checkpoint_dir = checkpoint_dir

  def build(self):
    """Setup summary writer and checkpoint."""

    time_str = time_string()

    train_summary_dir = os.path.join(os.path.join(self._tf_summary_dir, time_str), 'train')
    valid_summary_dir = os.path.join(os.path.join(self._tf_summary_dir, time_str), 'valid')
    os.makedirs(train_summary_dir, exist_ok=True)
    os.makedirs(valid_summary_dir, exist_ok=True)

    self._train_summary_writer = tf.summary.create_file_writer(train_summary_dir)
    self._valid_summary_writer = tf.summary.create_file_writer(valid_summary_dir)

    ckpt_dir = os.path.join(self._checkpoint_dir, time_str)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1),
                                     step=tf.Variable(1),
                                     model=self.model,
                                     optimizer=self.optimizer)
    self._checkpoint_manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=5)

  @tf.function(experimental_relax_shapes=True)
  def _train_step(self, inputs: Dict[str, tf.Tensor]) -> None:
    """Perform a distributed training step.

       Note: This function works normally within the scope of a Default tf.distrute.strategy,
             in other words, it will create single replica for non-distributed training.
    """

    def fn(inp):
      """Replicated training step."""

      with tf.GradientTape() as tape:
        logits = self.model(inp, training=True)
        per_example_loss = self.loss_fn(inp['contact_map'], logits, inp['mask_2d'], self._pos_weight)
        loss = tf.nn.compute_average_loss(per_example_loss)
      grads = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
      self._train_loss_metric.update_state(per_example_loss)

    self.strategy.run(fn, args=(inputs,))

  @tf.function(experimental_relax_shapes=True)
  def _test_step(self, inputs: Dict[str, tf.Tensor]) -> List[tf.Tensor]:
    """Perform a distributed testing step."""

    def fn(inp):
      """Replicated testing step."""

      logits = self.model(inp)
      per_example_loss = self.loss_fn(inp['contact_map'], logits, inp['mask_2d'], self._pos_weight)
      self._valid_loss_metric.update_state(per_example_loss)
      pred = tf.multiply(tf.sigmoid(logits), inp['mask_2d'])
      return pred

    pred = self.strategy.run(fn, args=(inputs,))
    #TODO: add metrics (f1, recall ...)

    return [self.strategy.gather(pred, axis=0),
            self.strategy.gather(inputs['contact_map'], axis=0),
            self.strategy.gather(inputs['protein_length'], axis=0)]

  def run_train_epoch(self, dataset) -> tf.Tensor:
    self._checkpoint_manager.checkpoint.epoch.assign_add(1)
    for (i, batch) in enumerate(dataset):
      self._train_step(batch)

    return self._train_loss_metric.result()

  def run_test_epoch(self, dataset) -> List[tf.Tensor]:
    preds = []
    trues = []
    lengths = []
    for (i, batch) in enumerate(dataset):
      pred, true, length = self._test_step(batch)

      preds.extend([x for x in pred.numpy()])
      trues.extend([x for x in true.numpy()])                       # list of array(N, N)
      lengths.extend([x for x in length.numpy()])

    return self._valid_loss_metric.result(), preds, trues, lengths

  def summarize_metrics(self, results: Dict, epoch: int) -> None:
    """
      NOTE: 1. Call this function within the scope of a tf.summary.SummaryWriter
      Args:
        results: a python dict with three keys: ['short', 'medium', 'long'], and each key
        has a list of float values representing: [precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5]
    """
    tf.summary.experimental.set_step(epoch)
    with self._train_summary_writer.as_default():
      tf.summary.scalar('loss', self._train_loss_metric.result())

    with self._valid_summary_writer.as_default():
      tf.summary.scalar('loss', self._valid_loss_metric.result())

      for k, v in results.items():
        tf.summary.scalar('precision/' + k, v[0])
        tf.summary.scalar('recall/' + k, v[1])
        tf.summary.scalar('f1/' + k, v[2])
        tf.summary.scalar('aupr/' + k, v[3])
        tf.summary.scalar('precision_L/' + k, v[4])
        tf.summary.scalar('precision_L_2/' + k, v[5])
        tf.summary.scalar('precision_L_5/' + k, v[6])

    self.reset_metrics()

  def reset_metrics(self) -> None:
    self._train_loss_metric.reset_state()
    self._valid_loss_metric.reset_state()

  def restore_checkpoint(self, ckpt_path=None) -> int:
    if ckpt_path is None:
      ckpt_path = self.i_checkpoint_manager.last_checkpoint
    self._checkpoint_manager.checkpoint.restore(ckpt_path).assert_existing_objects_matched()
    print("Checkpoint restored from {}.".format(ckpt_path))
    return int(self._checkpoint_manager.checkpoint.epoch.numpy())

  def save_checkpoint(self, epoch: int) -> None:
    saved_path = self._checkpoint_manager.save()
    print("Saved checkpoint for epoch {}: {}".format(epoch, saved_path))
