from typing import Dict, List, Optional, Callable, Any, Tuple

import tensorflow as tf
import tensorflow.keras as K


class AAEmbedding(K.layers.Layer):
  """Embedding of amino acids."""
  def __init__(self,
               units: int,
               vocab_size: int,
               embedding_dim: int,
               activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
               name='AAEmbedding'):

    super().__init__(name=name)

    self.embed_mat = tf.Variable(tf.random.normal([vocab_size, embedding_dim]))

    # initialize transformation weights
    self.weight_emb = tf.Variable(tf.random.normal([embedding_dim, units]))
    self.weight_onehot = tf.Variable(tf.random.normal([vocab_size, units]))
    self.bias = tf.Variable(tf.random.normal([1, units]))

    self.act = activation

  def get_embedding_shape(self) -> tf.TensorShape:
    return self.embed_mat.shape

  def get_embedding(self, x: tf.Tensor) -> tf.Tensor:
    return tf.nn.embedding_lookup(self.embed_mat, x)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    h_emb = tf.matmul(self.get_embedding(x), self.weight_emb)
    h_onehot = tf.nn.embedding_lookup(self.weight_onehot, x)

    # TODO: concatenate?
    h_out = h_emb + h_onehot + self.bias
    return self.act(h_out)


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

    lstm_cells = []
    for _ in range(num_stacks):
      lstm_cells.append(K.layers.LSTMCell(units))
    stacked_lstm = K.layers.StackedRNNCells(lstm_cells)

    if bidirectional:
      forward_layer = K.layers.RNN(
          stacked_lstm,
          return_sequences=True,
          return_state=False,
          time_major=False)
      backward_layer = K.layers.RNN(
          stacked_lstm,
          return_sequences=True,
          return_state=False,
          time_major=False,
          go_backwards=True)
      self.lstm = K.layers.Bidirectional(
          forward_layer,
          backward_layer=backward_layer,
          merge_mode=None)
    else:
      self.lstm = K.layers.RNN(stacked_lstm)

    self.linear = K.layers.Dense(units)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    [forward, backward] = self.lstm(x)
    h = tf.concat([forward, backward], axis=-1)
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
    super().__init__(name=name)

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

