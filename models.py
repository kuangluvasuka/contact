from typing import Dict
import tensorflow as tf
import tensorflow.keras as K

from layers import AAEmbedding, BiLSTM, DenseConv, ResidualBlock, GraphBlock


#TODO: base class?


class ConvModel(K.Model):
  def __init__(self, hparams: Dict, name='MLPConv'):
    super().__init__(name=name)
    hp = hparams

    self.encoder = K.Sequential(name='encoder')
    if hp['use_pretrain']:
      self.encoder.add(K.layers.LayerNormalization(axis=-1))
      self._input_source = 'pretrained_sequence'
    else:
      self.encoder.add(AAEmbedding(hp['embedding_units'], hp['vocab_size'], embedding_dim=hp['embedding_dim']))
      self.encoder.add(BiLSTM(hp['lstm_units'], hp['lstm_stacks'], hp['bidirectional']))
      self._input_source = 'primary'

    #TODO: try differnet pairing approach
    def sequence_to_map(x: tf.Tensor):
      x_expand_1 = tf.expand_dims(x, axis=1)            # [B, 1, L, dim]
      x_expand_2 = tf.expand_dims(x, axis=2)            # [B, L, 1, dim]
      x_abs = tf.math.abs(x_expand_1 - x_expand_2)      # [B, L, L, dim]
      x_mul = tf.math.multiply(x_expand_1, x_expand_2)
      return tf.concat([x_abs, x_mul], axis=-1)         # [B, L, L, 2dim

    #TODO: delete one of the pairing func
    def concat_pairs(tensor):
      seqlen = tf.shape(tensor)[1]
      input_left = tf.tile(tensor[:, :, None], (1, 1, seqlen, 1))
      input_right = tf.tile(tensor[:, None, :], (1, seqlen, 1, 1))
      output = tf.concat((input_left, input_right), -1)
      return output


    self.decoder = K.Sequential(name='decoder')
    self.decoder.add(K.layers.Lambda(sequence_to_map))
    self.decoder.add(DenseConv(hp['fc_dims'], hp['filters'], hp['kernel_size'], hp['pool_size']))

  def call(self, inputs: Dict[str, tf.Tensor], training=False) -> tf.Tensor:
    encoded_seq = self.encoder(inputs[self._input_source])
    asymmetric_map = self.decoder(encoded_seq, training=training)
    contact_map_logit = (asymmetric_map + tf.transpose(asymmetric_map, (0, 2, 1, 3))) / 2
    return tf.squeeze(contact_map_logit, axis=3)


class Resnet(K.Model):
  def __init__(self, hparams: Dict, name='Resnet'):
    super().__init__(name=name)
    hp = hparams

    self.encoder = K.Sequential(name='encoder')
    if hp['use_pretrain']:
      self.encoder.add(K.layers.LayerNormalization(axis=-1))
      self._input_source = 'pretrained_sequence'
    else:
      self.encoder.add(AAEmbedding(hp['embedding_units'], hp['vocab_size'], embedding_dim=hp['embedding_dim']))
      self.encoder.add(BiLSTM(hp['lstm_units'], hp['lstm_stacks'], hp['bidirectional']))
      self._input_source = 'primary'

    def sequence_to_map(x: tf.Tensor):
      x_expand_1 = tf.expand_dims(x, axis=1)
      x_expand_2 = tf.expand_dims(x, axis=2)
      x_abs = tf.math.abs(x_expand_1 - x_expand_2)
      x_mul = tf.math.multiply(x_expand_1, x_expand_2)
      return tf.concat([x_abs, x_mul], axis=-1)

    #TODO: delete one of the pairing func
    def concat_pairs(tensor):
      seqlen = tf.shape(tensor)[1]
      input_left = tf.tile(tensor[:, :, None], (1, 1, seqlen, 1))
      input_right = tf.tile(tensor[:, None, :], (1, seqlen, 1, 1))
      output = tf.concat((input_left, input_right), -1)
      return output


    self.decoder = K.Sequential(name='decoder')
    self.decoder.add(K.layers.Lambda(sequence_to_map))
    self.decoder.add(ResidualBlock(128, 7, layer_norm=False, resize_shortcut=True, drop_rate=0.1))
    for i in range(5):
      self.decoder.add(ResidualBlock(128, 7, drop_rate=0.1))

    self.decoder.add(ResidualBlock(64, 3, layer_norm=False, resize_shortcut=True, drop_rate=0.1))
    for i in range(5):
      self.decoder.add(ResidualBlock(64, 3, drop_rate=0.1))

    #TODO: add weight normalization?
    self.decoder.add(K.layers.Dense(1))

  def call(self, inputs: Dict[str, tf.Tensor], training=False):
    encoded_seq = self.encoder(inputs[self._input_source])
    asymmetric_map = self.decoder(encoded_seq, training=training)
    contact_map_logit = (asymmetric_map + tf.transpose(asymmetric_map, (0, 2, 1, 3))) / 2
    return tf.squeeze(contact_map_logit, axis=3)


class GraphModel(K.Model):
  def __init__(self, hparams: Dict, name='Graph'):
    super().__init__(name=name)
    hp = hparams

    self.encoder = K.Sequential(name='encoder')
    if hp['use_pretrain']:
      self.encoder.add(K.layers.LayerNormalization(axis=-1))
      self._input_source = 'pretrained_sequence'
    else:
      self.encoder.add(AAEmbedding(hp['embedding_units'], hp['vocab_size'], embedding_dim=hp['embedding_dim']))
      self.encoder.add(BiLSTM(hp['lstm_units'], hp['lstm_stacks'], hp['bidirectional']))
      self._input_source = 'primary'

    def sequence_to_map(x: tf.Tensor):
      x_expand_1 = tf.expand_dims(x, axis=1)
      x_expand_2 = tf.expand_dims(x, axis=2)
      x_abs = tf.math.abs(x_expand_1 - x_expand_2)
      x_mul = tf.math.multiply(x_expand_1, x_expand_2)
      return [tf.concat([x_abs, x_mul], axis=-1), x]

    # multi-in/output not available for Sequential, so use functional api
    self.s2m_layer = K.layers.Lambda(sequence_to_map)

    self.graph_layers = []
    self.graph_layers.append(GraphBlock(2 * hp['lstm_units'], hp['lstm_units']))
    self.graph_layers.append(GraphBlock(2 * hp['lstm_units'], hp['lstm_units']))
    self.linear = K.layers.Dense(1)

  def call(self, inputs: Dict[str, tf.Tensor], training=False):
    encoded_seq = self.encoder(inputs[self._input_source])
    hidden_inputs = []
    hidden_inputs.append(self.s2m_layer(encoded_seq))
    for graph in self.graph_layers:
      hidden_inputs.append(graph(hidden_inputs[-1]))
    asymmetric_map = self.linear(hidden_inputs[-1][0])
    contact_map_logit = (asymmetric_map + tf.transpose(asymmetric_map, (0, 2, 1, 3))) / 2
    return tf.squeeze(contact_map_logit, axis=3)
