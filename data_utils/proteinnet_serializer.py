"""
TF parser for ProteinNet Records. Source available at https://github.com/aqlaboratory/proteinnet
Edited for use with the representation learning benchmark.
"""

__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2018, Harvard Medical School"
__license__ = "MIT"

import os
from glob import glob
import tensorflow as tf
import numpy as np
from .vocabs import AA_DICT, PFAM_VOCAB

NUM_AAS = 20
NUM_DIMENSIONS = 3

"""
Inputs
    primary: The amino acid sequence of the protein.
    protein_length: The length of the actual sequence, used for padding and masking.
    valid_mask: Binary mask indicating which positions should not be predicted due to experimental challenges.
    evolutionary (Optional): The alignment-based features, in this produced by a PSSM.
Outputs
    contact_map: Symmetric binary matrix of contacts
Metadata
    id: An id that is either SCOPe id, PDB id, or CASP12 id, depending on the source.
NOTE
    There is more data available in ProteinNet than what we directly use here. See the ProteinNet repository
    for more details.
"""


def masking_matrix(mask, name=None):
    """ Constructs a masking matrix to zero out pairwise distances due to missing residues or padding.
    Args:
        mask: 0/1 vector indicating whether a position should be masked (0) or not (1)
    Returns:
        A square matrix with all 1s except for rows and cols whose corresponding indices in mask are set to 0.
        [MAX_SEQ_LENGTH, MAX_SEQ_LENGTH]
    """

    with tf.name_scope(name, 'masking_matrix', [mask]):
        mask = tf.convert_to_tensor(mask, name='mask')

        mask = tf.expand_dims(mask, 0)
        base = tf.ones([tf.size(mask), tf.size(mask)])
        matrix_mask = base * mask * tf.transpose(mask)

        return matrix_mask


def compute_pairwise_squared_distance(matrix):
    pairwise_dot = tf.matmul(matrix, matrix, transpose_b=True)
    squared_norm = tf.linalg.diag_part(pairwise_dot)
    sq_dist = squared_norm[:, None] + squared_norm[None, :] - 2 * pairwise_dot

    return sq_dist


def convert_to_pfam_vocab(sequence):

    AA_TO_PFAM = {aa_enc: PFAM_VOCAB[aa] for aa, aa_enc in AA_DICT.items()}

    def convert_seq(sequence_numpy):
        re_encoded = np.zeros_like(sequence_numpy)
        for aa_enc, pfam_enc in AA_TO_PFAM.items():
            #re_encoded[sequence_numpy == aa_enc] = pfam_enc      # NOTE: wierd bug, only crash when len(sequence_num) <= 31. Env: pythan 3.7.4, tf 2.3.0
            re_encoded[np.array(sequence_numpy == aa_enc)] = pfam_enc
        return re_encoded

    reencoded_tensor = tf.py_function(convert_seq, [sequence], sequence.dtype)
    reencoded_tensor.set_shape(sequence.shape)
    return reencoded_tensor


def deserialize_proteinnet_sequence(serialized_example, pretrained_embedding=None, embedding_dim=None):
    """ Reads and parses a ProteinNet TF Record.
        Primary sequences are mapped onto 20-dimensional one-hot vectors.
        Evolutionary sequences are mapped onto num_evo_entries-dimensional real-valued vectors.
        Secondary structures are mapped onto ints indicating one of 8 class labels.
        Tertiary coordinates are flattened so that there are 3 times as many coordinates as
        residues.
        Evolutionary, secondary, and tertiary entries are optional.
    Args:
        filename_queue: TF queue for reading files
        max_length:     Maximum length of sequence (number of residues) [MAX_LENGTH]. Not a
                        TF tensor and is thus a fixed value.
    Returns:
        id: string identifier of record
        primary: AA sequence
        protein_length: Length of amino acid sequence
    Other Notes:
        There are several other features that we do not return. These could be used for multi-
        task training or for other forms of training in the future.
    """

    context, features = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features={'id': tf.io.FixedLenFeature((1,), tf.string)},
        sequence_features={
            'primary': tf.io.FixedLenSequenceFeature((1,), tf.int64),
            'evolutionary': tf.io.FixedLenSequenceFeature((21,), tf.float32, allow_missing=True),
            'secondary': tf.io.FixedLenSequenceFeature((1,), tf.int64, allow_missing=True),
            'tertiary': tf.io.FixedLenSequenceFeature((NUM_DIMENSIONS,), tf.float32, allow_missing=True),
            'mask': tf.io.FixedLenSequenceFeature((1,), tf.float32, allow_missing=True)})

    id_ = context['id'][0]
    primary = tf.cast(features['primary'][:, 0], tf.int32)
    evolutionary = features['evolutionary']  # noqa: F841
    secondary = tf.cast(features['secondary'][:, 0], tf.int32)  # noqa: F841
    tertiary = features['tertiary'] / 100  # tertiary returned as hundredths of anstroms
    tertiary = tertiary[1::3]  # Select only Carbon-alpha atoms
    mask = features['mask'][:, 0]

    pairwise_squared_distance = compute_pairwise_squared_distance(tertiary)
    contact_map = tf.cast(tf.less_equal(pairwise_squared_distance, 8 ** 2), tf.float32)

    # TODO: convert alphabet
    primary = convert_to_pfam_vocab(primary)

    protein_length = tf.size(primary)

    # Generate tertiary masking matrix--if mask is missing then assume all residues are present
    # and convert mask into 2-dimensional
    mask_1d = tf.cond(tf.not_equal(tf.size(mask), 0), lambda: mask, lambda: tf.ones(protein_length))
    mask_1d = tf.cast(mask_1d, tf.float32)
    mask_2d = tf.multiply(tf.expand_dims(mask_1d, axis=1), tf.expand_dims(mask_1d, axis=0))

    #TODO:
    #NOTE: bug workaround: the 110-th contact map in feeder.valid has bug
    contact_map = tf.multiply(contact_map, mask_2d)

    ret = {'id': id_,
           'primary': primary,
           'contact_map': contact_map,
           'evolutionary': evolutionary,
           'protein_length': protein_length,
           'mask_2d': mask_2d}

    if pretrained_embedding:
        pretrained_seq = tf.py_function(lambda s: pretrained_embedding[s.numpy()], inp=[id_], Tout=tf.float32)
        pretrained_seq.set_shape([primary.shape[0], embedding_dim])
        ret.update({'pretrained_sequence': pretrained_seq})

    return ret


def get_proteinnet_length_statistics(data_folder):
    """One can plot the length histogram via the returned value"""
    file_name_pattern = os.path.join(data_folder, 'contact_map_train*.tfrecord')
    data_files = glob(file_name_pattern)
    dataset = tf.data.TFRecordDataset(data_files)
    dataset = dataset.map(deserialize_proteinnet_sequence)
    dataset = dataset.filter(lambda d: d['protein_length'] < 1000)

    lengths = []
    for (i, example) in enumerate(dataset):
        lengths.append(example['protein_length'].numpy())

    # hist, bin_edges = np.histogram(lengths, bins='auto')
    return lengths
