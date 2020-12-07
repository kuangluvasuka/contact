import os
from glob import glob
from typing import Tuple, List, Dict, Callable
import tensorflow as tf


class Feeder():
  """Loads minibatch of data from preprocessed TFRecords."""

  def __init__(self,
               data_folder: str,
               batch_size: int,
               test_batch_size: int,
               bucket_boundaries: List[int],
               tf_parse_func: Callable[[bytes], Dict[str, tf.Tensor]],
               shuffle: bool = False,
               max_protein_length: int = 1000):

    self._tf_parse_func = tf_parse_func
    self.batch_size = batch_size
    self.test_batch_size = test_batch_size
    self.bucket_boundaries = bucket_boundaries
    self.shuffle = shuffle
    self.max_protein_length = max_protein_length

    self.train, self.valid = self._get_data_loader(data_folder)
    self.test = self._get_test_loader(data_folder)

  def _get_data_files(self,
                      data_folder: str,
                      data_split: str = 'train') -> List[str]:

    file_name_pattern = os.path.join(data_folder, 'contact_map_' + data_split + '*.tfrecord')
    data_files = glob(file_name_pattern)

    if len(data_files) == 0:
      raise FileNotFoundError(data_files)

    return data_files

  def _build_data_loader(self,
                         data_files: List[str],
                         batch_size: int,
                         shuffle: bool,
                         bucket_batch: bool = False) -> tf.data.Dataset:

    dataset = tf.data.TFRecordDataset(data_files)
    dataset = dataset.map(self._tf_parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(lambda d: d['protein_length'] < self.max_protein_length)
    dataset = dataset.shuffle(1024) if shuffle else dataset.prefetch(1024)

    if bucket_batch:
      batch_size = [batch_size] * (len(self.bucket_boundaries) + 1)
      batch_func = tf.data.experimental.bucket_by_sequence_length(
          lambda d: d['protein_length'],
          self.bucket_boundaries,
          batch_size)
      dataset = dataset.apply(batch_func)
    else:
      dataset = dataset.padded_batch(batch_size)

    return dataset

  def _get_data_loader(self, data_folder: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_files = self._get_data_files(data_folder, 'train')
    valid_files = self._get_data_files(data_folder, 'valid')

    train_data = self._build_data_loader(train_files, self.batch_size, shuffle=self.shuffle, bucket_batch=True)
    valid_data = self._build_data_loader(valid_files, self.test_batch_size, shuffle=False)

    return train_data, valid_data

  def _get_test_loader(self, data_folder: str) -> tf.data.Dataset:
    test_files = self._get_data_files(data_folder, 'test')
    test_data = self._build_data_loader(test_files, self.test_batch_size, shuffle=False)

    return test_data
