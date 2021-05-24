import os
from typing import Union, Dict
from datetime import datetime
import tensorflow as tf


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


def log_results(results: Dict):
  """
    NOTE: 1. Call this function within the scope of a tf.summary.SummaryWriter
    Args:
      results: a python dict with three keys: ['short', 'medium', 'long'], and each key
      has a list of float values representing: [precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5]
  """

  for k, v in results.items():
    tf.summary.scalar('precision/' + k, v[0])
    tf.summary.scalar('recall/' + k, v[1])
    tf.summary.scalar('f1/' + k, v[2])
    tf.summary.scalar('aupr/' + k, v[3])
    tf.summary.scalar('precision_L/' + k, v[4])
    tf.summary.scalar('precision_L_2/' + k, v[5])
    tf.summary.scalar('precision_L_5/' + k, v[6])



def time_string():
  return datetime.now().strftime("%b.%d_%H.%M")
