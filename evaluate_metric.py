"""
  Credit: https://github.com/songlab-cal/tape-neurips2019/blob/master/tape/analysis/contact_prediction/evaluate_contact_prediction_metrics.py
"""
from typing import List, Callable

import math
import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.metrics import (precision_score, recall_score, f1_score, average_precision_score)


#def precision_cutoff(true_labels: List[int],
#                     predictions: List[int],
#                     cutoff_idx: int):
#  # Sort probabilities
#  high_to_low_idx = np.argsort(predictions).tolist().reverse()
#  sorted_predictions = [predictions[i] for i in high_to_low_idx]
#  sorted_true_labels = [true_labels[i] for i in high_to_low_idx]
#
#  # Cutoff outputs at cutoff_idx and compute precision
#  cutoff_predictions = sorted_predictions[:cutoff_idx]
#  cutoff_true_labels = sorted_true_labels[:cutoff_idx]
#  precision = precision_score(cutoff_true_labels, cutoff_predictions)
#
#  return precision

def apply_to_full_data(labels: List[int],
                       predictions: List[int],
                       metric: Callable):
  metrics = []
  for label, prediction in zip(labels, predictions):
    metrics.append(metric(label, prediction))

  return np.array(metrics)


def sigmoid(x):
  """ 1-D Sigmoid that is more stable.
  """
  return math.exp(-np.logaddexp(0, -x))


def convert_contact_map(logits: np.ndarray):
  """ Applies element-wise sigmoid, using stable version.
  """
  predicted_probs = np.zeros_like(logits)
  for i, row in enumerate(logits):
    for j, p in enumerate(row):
      predicted_probs[i, j] = sigmoid(p)
  return predicted_probs


def load_predictions(model: str):
  """Load all predictions for a given model.
  Returns
          results: numpy array of dim [n_val_set, max_length, 2]
           last dimension has [prediction]
  """
  model_path = 'test_predictions/contact_prediction/' + model + '_outputs.pkl'
  with open(model_path, 'rb') as f:
    data = pkl.load(f)
  sequences = data['primary']
  lengths = data['protein_length']
  true_maps = [cmap[:length, :length] for cmap, length in zip(data['contact_map'], lengths)]  # depad
  logits = [logit[:length, :length, 0] for logit, length in zip(data['contact_prob'], lengths)]
  pdb_ids = data['id']
  predicted_probs = [convert_contact_map(logit) for logit in logits]
  return sequences, predicted_probs, true_maps, lengths


def partition_contacts(full_contact_map: np.ndarray):
  """ Returns lists of short, medium, and long-range contacts.
  """
  length = full_contact_map.shape[0]
  short_contacts = [np.diagonal(full_contact_map, i) for i in range(6, 12)]
  medium_contacts = [np.diagonal(full_contact_map, i) for i in range(12, min(length, 24))]
  if length >= 24:
    long_contacts = [np.diagonal(full_contact_map, i) for i in range(24, length)]
  else:
    return np.concatenate(short_contacts), np.concatenate(medium_contacts), []
  return np.concatenate(short_contacts), np.concatenate(medium_contacts), np.concatenate(long_contacts)


def precision_cutoff(true_labels: List[int],
                     predictions: List[int],
                     cutoff_idx: int):
  # Sort probabilities
  high_to_low_idx = np.argsort(predictions).tolist()[::-1]
  sorted_predictions = [predictions[i] for i in high_to_low_idx]
  sorted_true_labels = [true_labels[i] for i in high_to_low_idx]

  # Cutoff outputs at cutoff_idx and compute precision
  cutoff_predictions = sorted_predictions[:cutoff_idx]
  binarized_cutoff_predictions = [1 if i > 0.5 else 0 for i in cutoff_predictions]
  cutoff_true_labels = sorted_true_labels[:cutoff_idx]
  score = np.sum([(t == p) & (t == 1) for t, p in zip(cutoff_true_labels, binarized_cutoff_predictions)])
  num_true_positives = np.sum([t == 1 for t in cutoff_true_labels])
  if num_true_positives > 0:
    score /= num_true_positives
  else:
    score /= cutoff_idx

  return score


def collect_metrics(true_maps: List[int],
                    predicted_probs: List[float]):
  """ Returns all metrics for contact prediction.

  Takes in list of true maps and contact probs. Sorts in decreasing order
  for the precision metrics.
  """
  binarized_predictions = [[1 if i > 0.5 else 0 for i in predicted_prob] for predicted_prob in predicted_probs]

  # Simple metrics on binarized predictions
  precision = apply_to_full_data(true_maps, binarized_predictions, lambda x, y: precision_score(x, y, zero_division=0))
  recall = apply_to_full_data(true_maps, binarized_predictions, lambda x, y: recall_score(x, y, zero_division=0))
  f1 = apply_to_full_data(true_maps, binarized_predictions, lambda x, y: f1_score(x, y, zero_division=0))

  # Need to ensure any runs with only 0's are dropped
  aupr = []
  for label, prediction in zip(true_maps, predicted_probs):
    if np.sum(label) == 0:
      continue
    else:
      aupr.append(average_precision_score(label, prediction))

  precision_L = apply_to_full_data(true_maps, predicted_probs, lambda x, y: precision_cutoff(x, y, x.shape[0]))
  precision_L_2 = apply_to_full_data(true_maps, predicted_probs, lambda x, y: precision_cutoff(x, y, int(x.shape[0] / 2)))
  precision_L_5 = apply_to_full_data(true_maps, predicted_probs, lambda x, y: precision_cutoff(x, y, int(x.shape[0] / 5)))

  return precision, recall, f1, aupr, precision_L, precision_L_2, precision_L_5
