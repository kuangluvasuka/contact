from typing import Dict, List, Optional, Callable
import tensorflow as tf

from layers import AAEmbedding, BiLSTM, DenseConv


class ConvModel(tf.keras.Model):
  def __init__(self, hparams: Dict):
    super().__init__(self)
    hp = hparams

    self._use_pretrain = hp['use_pretrain']
    if not self._use_pretrain:
      self.encoder = tf.keras.Sequential(name='encoder')
      self.encoder.add(AAEmbedding(hp['embedding_units'], hp['vocab_size'], embedding_dim=hp['embedding_dim']))

      self.encoder.add(BiLSTM(hp['lstm_units'], hp['lstm_stacks'], hp['bidirectional']))

    self.decoder = tf.keras.Sequential(name='decoder')
    self.decoder.add(DenseConv(hp['fc_dims'], hp['filters'], hp['kernel_size']))

  def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
    if self._use_pretrain:
      encoded_seq = inputs['pretrained_sequence']
    else:
      encoded_seq = self.encoder(inputs['primary'])
    asymmetric_map = self.decoder(encoded_seq)
    contact_map_logit = (asymmetric_map + tf.transpose(asymmetric_map, (0, 2, 1))) / 2
    return contact_map_logit



