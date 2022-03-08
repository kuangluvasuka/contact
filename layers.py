from typing import Dict, List, Optional, Callable, Any, Tuple

import tensorflow as tf
import tensorflow.keras as K


# TODO: redifine embedding and BiLSTM layer
class AAEmbedding(K.layers.Layer):
  """Embedding of amino acids."""
  def __init__(self,
               units: int,
               vocab_size: int,
               embedding_dim: int,
               activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
               name='AAEmbedding'):

    super().__init__(name=name)

    self._vocab_size = vocab_size
    self.embed_mat = tf.Variable(tf.random.normal([vocab_size, embedding_dim]))

  def get_embedding_shape(self) -> tf.TensorShape:
    return self.embed_mat.shape

  def get_embedding(self, x: tf.Tensor) -> tf.Tensor:
    return tf.nn.embedding_lookup(self.embed_mat, x)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    h_emb = self.get_embedding(x)
    h_onehot = tf.one_hot(x, depth=self._vocab_size)

    h_out = tf.concat([h_emb, h_onehot], axis=-1)
    return h_out


class BiLSTM(K.layers.Layer):
  def __init__(self,
               units: int,
               num_stacks: int,
               bidirectional: bool,
               name='BiLSTM'):
    """
      Args:
        units: int, dimensionality of the output tensor
        num_layers: int, number of stacked lstm layers in each forward/backward direction
        bidirectional: bool, whether to use bidirectional lstm

      Input:
        x: embeddings of AA in shape=[B, L, embedding_out]
      Output:
        h: vectors in shape=[B, L, units]
    """
    super().__init__(name=name)

    if bidirectional:
      forward_layer = K.layers.RNN(
          K.layers.StackedRNNCells([K.layers.LSTMCell(units) for _ in range(num_stacks)]),
          return_sequences=True,
          return_state=False,
          time_major=False)
      backward_layer = K.layers.RNN(
          K.layers.StackedRNNCells([K.layers.LSTMCell(units) for _ in range(num_stacks)]),
          return_sequences=True,
          return_state=False,
          time_major=False,
          go_backwards=True)
      self.lstm = K.layers.Bidirectional(
          forward_layer,
          backward_layer=backward_layer,
          merge_mode='concat')
    else:
      self.lstm = K.layers.RNN(K.layers.StackedRNNCells([K.layers.LSTMCell(units) for _ in range(num_stacks)]))

    self.linear = K.layers.Dense(units)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    h = self.lstm(x)
    return self.linear(h)


class DenseConv(K.layers.Layer):
  def __init__(self,
               fc_dims: List[int],
               filters: List[int],
               kernel_size: List[int],
               pool_size: int,
               name='DenseConv'):
    """
      Args:
        fc_hidden_dim: List[int], dimensions of the output of the fully-connected layers
        filters: List[int], number of filters for two convolutional layers
        kernel_size: List[int], sizes of the 2D convolution filters
        pool_size: int, size of pooling filters

      Input:
        x: shape=[B, L, dim]
      Output:
        h: shape=[B, L, L, 1]
    """

    super().__init__(name=name)

    self._num_fc_layers = len(fc_dims)

    self.fc = []
    for i in range(self._num_fc_layers - 1):
      self.fc.append(K.layers.Dense(fc_dims[i], activation='relu'))
    self.fc.append(K.layers.Dense(fc_dims[-1]))
    self.conv1 = K.layers.Conv2D(filters[0], kernel_size[0], padding='same', activation='relu')
    self.pool1 = K.layers.MaxPooling2D(pool_size=(pool_size, pool_size), strides=1, padding='same')
    self.conv2 = K.layers.Conv2D(filters[1], kernel_size[1], padding='same')

  def call(self, x: tf.Tensor) -> tf.Tensor:
    fc_hidden = [x]
    for i in range(self._num_fc_layers):
      h = self.fc[i](fc_hidden[-1])
      fc_hidden.append(h)                             # fc_hidden[-1] = [B, L, L, H]

    conv1 = self.conv1(fc_hidden[-1])
    pool1 = self.pool1(conv1)
    return self.conv2(pool1)


class ResidualBlock(K.layers.Layer):
  def __init__(self,
               filters: int,
               kernel_size: int,
               dilation_rate: int = 1,
               layer_norm: bool = False,
               resize_shortcut: bool = False,
               activation: str = 'leaky_relu',
               drop_rate: float = 0.,
               name='ResidualBlock'):
    #super().__init__(name=name)
    super().__init__()

    if layer_norm:
      self.layer_norm = K.layers.LayerNormalization(axis=-1)
    else:
      self.layer_norm = K.layers.Lambda(lambda x: x)

    self.batch_norm1 = K.layers.BatchNormalization(axis=-1)
    self.batch_norm2 = K.layers.BatchNormalization(axis=-1)

    if activation == 'relu':
      self.act1 = K.layers.ReLU()
      self.act2 = K.layers.ReLU()
    elif activation == 'leaky_relu':
      self.act1 = K.layers.LeakyReLU()
      self.act2 = K.layers.LeakyReLU()
    else:
      raise ValueError("Unknown activation option, please select from 'relu' or 'leaky_relu'.")

    if resize_shortcut:
      self.linear = K.layers.Dense(filters)
    else:
      self.linear = K.layers.Lambda(lambda x: x)

    self.dropout = K.layers.Dropout(drop_rate)

    self.conv1 = K.layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=True,
                                 activation='linear', dilation_rate=dilation_rate)
    self.conv2 = K.layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=True,
                                 activation='linear', dilation_rate=dilation_rate)

  def call(self, inputs, training=False):
    x = inputs
    x = self.layer_norm(inputs)

    x = self.conv1(x)
    x = self.batch_norm1(x, training=training)
    x = self.act1(x)

    x = self.conv2(x)
    x = self.batch_norm2(x, training=training)

    x = self.dropout(x, training=training)
    x += self.linear(inputs)

    return self.act2(x)


class GraphBlock(K.layers.Layer):
  """
    Single layer of graph neural networks that updates edge/node hidden representations.

  """
  def __init__(self,
               dim_e: int,
               dim_h: int, 
               name='GraphBlock'):
    super().__init__()
    # NOTE: dim_e = 2 * dim_h
    self.w_e1 = K.layers.Dense(dim_e, activation='relu')
    self.w_e2 = K.layers.Dense(dim_h, activation='relu')
    self.w_e3 = K.layers.Dense(dim_h, activation='relu')

    self.w_h1 = K.layers.Dense(dim_h, activation='relu')
    self.w_h2 = K.layers.Dense(dim_h, activation='relu')

  def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False):
    """
      Args:
        inputs[0]: edge feature (coarse map), [B, L, L, edge_dim]
        inputs[1]: node feature (AA encoding), [B, L, node_dim]

      Returns:
        new edge feature [B, L, L, edge_dim]
        new node feature [B, L, node_dim]
    """
    e, h = inputs

    seq_len = tf.shape(h)[1]
    h_u = tf.tile(tf.expand_dims(h, axis=1), [1, seq_len, 1, 1])        # [B, 1, L, dim]
    h_v = tf.tile(tf.expand_dims(h, axis=2), [1, 1, seq_len, 1])        # [B, L, 1, dim]
    h_concat = tf.concat([h_u, h_v], -1)                                # [B, L, 1, 2dim]
    e_t = self.w_e1(tf.concat([self.w_e2(e), self.w_e3(h_concat)], -1))  # [B, L, L 2dim]

    e_t_mean = tf.reduce_mean(e_t, axis=2)                              # [B, L, 2dim]
    h_t = self.w_h1(tf.concat([self.w_h2(h), e_t_mean], -1))            # [B, L, dim]
    return [e_t, h_t]


