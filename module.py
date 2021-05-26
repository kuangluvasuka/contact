from typing import Dict, List, Optional, Callable

import tensorflow as tf
from tensorflow.keras import layers


class AAEmbedding(layers.Layer):
  """Embedding of amino acids."""
  def __init__(self,
               units: int,
               vocab_size: int,
               embedding_dim: Optional[int] = None,
               pretrained_embedding: Optional[tf.Tensor] = None,
               activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu):

    super().__init__(self)

    # initialize embedding
    if pretrained_embedding is not None:

      # TODO: trainable?
      self.embedding = tf.Variable(initial_value=pretrained_embedding, trainable=True)
    else:
      self.embedding = tf.Variable(tf.random.normal([vocab_size, embedding_dim]))

    # initialize transformation weights
    self.weight_emb = tf.Variable(tf.random.normal([embedding_dim, units]))
    self.weight_onehot = tf.Variable(tf.random.normal([vocab_size, units]))
    self.bias = tf.Variable(tf.random.normal([1, units]))

    self.act = activation

  def get_embedding_shape(self) -> tf.TensorShape:
    return self.embedding.shape

  def get_embedding(self, x: tf.Tensor) -> tf.Tensor:
    return tf.nn.embedding_lookup(self.embedding, x)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    h_emb = tf.matmul(self.get_embedding(x), self.weight_emb)
    h_onehot = tf.nn.embedding_lookup(self.weight_onehot, x)

    # TODO: concatenate?
    h_out = h_emb + h_onehot + self.bias
    return self.act(h_out)



class BiLSTM(layers.Layer):
  def __init__(self,
               units: int,
               num_stacks: int,
               bidirectional: bool,
               ):
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
    super().__init__(self)

    lstm_cells = []
    for _ in range(num_stacks):
      lstm_cells.append(layers.LSTMCell(units))
    stacked_lstm = layers.StackedRNNCells(lstm_cells)

    if bidirectional:
      forward_layer = layers.RNN(
          stacked_lstm,
          return_sequences=True,
          return_state=False,
          time_major=False)
      backward_layer = layers.RNN(
          stacked_lstm,
          return_sequences=True,
          return_state=False,
          time_major=False,
          go_backwards=True)
      self.lstm = layers.Bidirectional(
          forward_layer,
          backward_layer=backward_layer,
          merge_mode=None)
    else:
      self.lstm = layers.RNN(stacked_lstm)

    self.linear = layers.Dense(units)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    [forward, backward] = self.lstm(x)
    h = tf.concat([forward, backward], axis=-1)
    return self.linear(h)


class DenseConv(layers.Layer):
  def __init__(self,
               fc_dims: List[int],
               filters: int,
               kernel_size: int = 7):
    """
      Args:
        fc_hidden_dim: List[int], dimensions of the output of the fully-connected layers
        filters: int, dimension of the output space
        kernel_size: int, size of the 2D convolution filters

      Input:
        x: shape=[B, L, dim]
      Output:
        h: shape=[B, L, L, 1]
    """

    super().__init__(self)

    self._num_fc_layers = len(fc_dims)

    self.fc = []
    for i in range(self._num_fc_layers - 1):
      self.fc.append(layers.Dense(fc_dims[i], activation='relu'))
    self.fc.append(layers.Dense(fc_dims[-1]))
    self.conv = layers.Conv2D(filters, kernel_size, padding='same')

  def call(self, x: tf.Tensor) -> tf.Tensor:
    # concat pairs to get pair-wise feature vector
    x_expand_1 = tf.expand_dims(x, axis=1)            # [B, 1, L, dim]
    x_expand_2 = tf.expand_dims(x, axis=2)            # [B, L, 1, dim]
    x_abs = tf.math.abs(x_expand_1 - x_expand_2)      # [B, L, L, dim]
    x_mul = tf.math.multiply(x_expand_1, x_expand_2)
    pair_concat = tf.concat([x_abs, x_mul], axis=-1)  # [B, L, L, 2dim]

    fc_hidden = [pair_concat]
    for i in range(self._num_fc_layers):
      h = self.fc[i](fc_hidden[-1])
      fc_hidden.append(h)                             # fc_hidden[-1] = [B, L, L, H]

    out = self.conv(fc_hidden[-1])

    return tf.squeeze(out, axis=3)


class ConvModel(tf.keras.Model):
  def __init__(self, hparams: Dict):
    super().__init__(self)
    hp = hparams

    pretrained = None
    if hp['pretrained']:
      pass

    self.layer_embedding = AAEmbedding(
        units=hp['embedding_units'],
        vocab_size=hp['vocab_size'],
        embedding_dim=hp['embedding_dim'],
        pretrained_embedding=pretrained)

    self.layer_lstm = BiLSTM(
        units=hp['lstm_units'],
        num_stacks=hp['lstm_stacks'],
        bidirectional=hp['bidirectional'])

    self.layer_conv = DenseConv(
        fc_dims=hp['fc_dims'],
        filters=hp['filters'],
        kernel_size=hp['kernel_size'])


  def _load_pretrained(self, ) -> None:
    pass


  def call(self, x: tf.Tensor) -> tf.Tensor:
    embedding = self.layer_embedding(x)
    lstm_hidden = self.layer_lstm(embedding)
    contact_map = self.layer_conv(lstm_hidden)
    return contact_map

