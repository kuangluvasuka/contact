import os
from glob import glob
from typing import List, Dict, Callable, Optional
import numpy as np
import tensorflow as tf


class Feeder():
  """Loads minibatch of data from preprocessed TFRecords."""

  def __init__(self,
               data_folder: str,
               batch_size: int,
               test_batch_size: int,
               bucket_boundaries: List[int],
               tf_parse_func: Callable[[bytes], Dict[str, tf.Tensor]],
               distribute_strategy: tf.distribute.Strategy,
               pretrained_embedding: Optional[Dict[str, np.ndarray]] = None,
               shuffle: bool = False,
               max_protein_length: int = 1000):

    self._tf_parse_func = tf_parse_func
    self.batch_size = batch_size
    self.test_batch_size = test_batch_size

    self._pretrained_embedding = pretrained_embedding

    self.bucket_boundaries = bucket_boundaries
    self.shuffle = shuffle
    self.max_protein_length = max_protein_length

    self._build_data_loader(data_folder, distribute_strategy)

  def _build_data_loader(self, data_folder: str,
                         strategy: tf.distribute.Strategy) -> None:
    train_files = self._get_data_files(data_folder, 'train')
    valid_files = self._get_data_files(data_folder, 'valid')
    test_files = self._get_data_files(data_folder, 'test')

    # check whether strategy is a Default Strategy
    with strategy.scope():
      has_strat = tf.distribute.has_strategy()

    train_loader = self._build(train_files, self.batch_size, drop_remainder=has_strat,
                               shuffle=self.shuffle, bucket_batch=True)
    valid_loader = self._build(valid_files, self.test_batch_size, drop_remainder=has_strat)
    self.train = strategy.experimental_distribute_dataset(train_loader)
    self.valid = strategy.experimental_distribute_dataset(valid_loader)
    self.test = self._build(test_files, self.test_batch_size, drop_remainder=False)

  def _get_data_files(self,
                      data_folder: str,
                      data_split: str = 'train') -> List[str]:

    file_name_pattern = os.path.join(data_folder, 'contact_map_' + data_split + '*.tfrecord')
    data_files = glob(file_name_pattern)

    if len(data_files) == 0:
      raise FileNotFoundError(data_files)

    return data_files

  def _build(self,
             data_files: List[str],
             batch_size: int,
             drop_remainder: bool,
             shuffle: bool = False,
             bucket_batch: bool = False) -> tf.data.Dataset:

    dataset = tf.data.TFRecordDataset(data_files)
    if isinstance(self._pretrained_embedding, dict):
      embed_dim = next(iter(self._pretrained_embedding.values())).shape[1]
      dataset = dataset.map(lambda s: self._tf_parse_func(s, self._pretrained_embedding, embed_dim),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
      dataset = dataset.map(self._tf_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(lambda d: d['protein_length'] <= self.max_protein_length)
    dataset = dataset.shuffle(1024) if shuffle else dataset.prefetch(1024)

    #pad_shapes = {'id': [],
    #              'primary': [self.max_protein_length],
    #              'contact_map': [self.max_protein_length, self.max_protein_length],
    #              'evolutionary': [self.max_protein_length, 21],    # 21 is the number of Evolutionary Entries
    #              'protein_length': [],
    #              'mask_2D': [self.max_protein_length, self.max_protein_length]}

    if bucket_batch:
      batch_size = [batch_size] * (len(self.bucket_boundaries) + 1)
      batch_func = tf.data.experimental.bucket_by_sequence_length(
          lambda d: d['protein_length'],
          self.bucket_boundaries,
          batch_size,
          #padded_shapes=pad_shapes,
          drop_remainder=drop_remainder)
      dataset = dataset.apply(batch_func)
    else:
      dataset = dataset.padded_batch(batch_size, drop_remainder=drop_remainder) #, padded_shapes=pad_shapes)

    return dataset
