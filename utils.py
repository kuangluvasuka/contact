from datetime import datetime
from typing import List
import numpy as np


def get_range_weighted_matrix(range_wt: List, max_len: int) -> np.ndarray:
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
  return res


def time_string():
  return datetime.now().strftime("%y%m%d_%H%M")
