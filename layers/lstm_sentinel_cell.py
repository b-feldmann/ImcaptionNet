import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import initializers, activations, regularizers
from tensorflow.python.keras._impl.keras.engine.base_layer import shape_type_conversion

class LSTMSentinelCell(Layer):
  """
  LSTM with visual sentinel gate, from the paper:
  Lu et al. Knowing When to Look:
            Adaptive Attention via A Visual Sentinel for Image Captioning
  https://arxiv.org/abs/1612.01887
  """

  def __init__(self, units,
               init='glorot_uniform',
               inner_init='orthogonal',
               forget_bias_init='one',
               activation='tanh',
               inner_activation='hard_sigmoid',
               W_regularizer=None,
               U_regularizer=None,
               b_regularizer=None,
               dropout_W=0.,
               dropout_U=0.,
               sentinel=True,
               **kwargs):
    super(LSTMSentinelCell, self).__init__(**kwargs)
    self.units = units
    self.W_initializer = initializers.get(init)
    self.U_initializer = initializers.get(inner_init)
    self.bias_initializer = initializers.get('zeros')
    self.forget_bias_init = initializers.get(forget_bias_init)
    self.activation = activations.get(activation)
    self.inner_activation = activations.get(inner_activation)
    self.W_regularizer = regularizers.get(W_regularizer)
    self.U_regularizer = regularizers.get(U_regularizer)
    self.b_regularizer = regularizers.get(b_regularizer)
    self.dropout_W = dropout_W
    self.dropout_U = dropout_U
    self.sentinel = sentinel
    if self.dropout_W or self.dropout_U:
      self.uses_learning_phase = True
    self.state_size = [units, units]
    # self.state_size = units

  @shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[1]

    self.W_i = self.add_weight(
        shape=(input_dim, self.units), name='{}_W_i'.format(self.name),
        initializer=self.W_initializer)
    self.U_i = self.add_weight(
        shape=(self.units, self.units), name='{}_U_i'.format(self.name),
        initializer=self.U_initializer)
    self.b_i = self.add_weight(
        shape=(self.units,), name='{}_b_i'.format(self.name),
        initializer=self.bias_initializer)

    self.W_f = self.add_weight(
        shape=(input_dim, self.units), name='{}_W_f'.format(self.name),
        initializer=self.W_initializer)
    self.U_f = self.add_weight(
        shape=(self.units, self.units), name='{}_U_f'.format(self.name),
        initializer=self.U_initializer)
    self.b_f = self.add_weight(
        shape=(self.units,), name='{}_b_f'.format(self.name),
        initializer=self.bias_initializer)

    self.W_c = self.add_weight(
        shape=(input_dim, self.units), name='{}_W_c'.format(self.name),
        initializer=self.W_initializer)
    self.U_c = self.add_weight(
        shape=(self.units, self.units), name='{}_U_c'.format(self.name),
        initializer=self.U_initializer)
    self.b_c = self.add_weight(
        shape=(self.units,), name='{}_b_c'.format(self.name),
        initializer=self.bias_initializer)

    self.W_o = self.add_weight(
        shape=(input_dim, self.units), name='{}_W_o'.format(self.name),
        initializer=self.W_initializer)
    self.U_o = self.add_weight(
        shape=(self.units, self.units), name='{}_U_o'.format(self.name),
        initializer=self.U_initializer)
    self.b_o = self.add_weight(
        shape=(self.units,), name='{}_b_o'.format(self.name),
        initializer=self.bias_initializer)

    if self.sentinel:
      # sentinel gate
      self.W_g = self.add_weight(
          shape=(input_dim, self.units), name='{}_W_g'.format(self.name),
          initializer=self.W_initializer)
      self.U_g = self.add_weight(
          shape=(self.units, self.units), name='{}_U_g'.format(self.name),
          initializer=self.U_initializer)
      self.b_g = self.add_weight(
          shape=(self.units,), name='{}_b_g'.format(self.name),
          initializer=self.bias_initializer)

    self.built = True

  def call(self, x, states, training=None, constants=None):
    h_tm1 = states[0]
    c_tm1 = states[1]
    B_U = constants[0]
    B_W = constants[1]

    x_i = K.dot(x, self.W_i) + self.b_i
    x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
    x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
    x_o = K.dot(x * B_W[3], self.W_o) + self.b_o
    if self.sentinel:
      x_g = K.dot(x * B_W[4], self.W_g) + self.b_g

    i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))
    f = self.inner_activation(x_f + K.dot(h_tm1 * B_U[1], self.U_f))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1 * B_U[2], self.U_c))
    o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[3], self.U_o))
    h = o * self.activation(c)
    if self.sentinel:
      g = self.inner_activation(x_g + K.dot(h_tm1 * B_U[4], self.U_g))
      s = g * self.activation(c)
      return tf.concat([h, s], -1), [h, c]
    else:
      return h, [h, c]

  def get_config(self):
    config = {
        "output_dim": self.units,
        "init": self.init.__name__,
        "inner_init": self.inner_init.__name__,
        "forget_bias_init": self.forget_bias_init.__name__,
        "activation": self.activation.__name__,
        "inner_activation": self.inner_activation.__name__,
        "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
        "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
        "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
        "dropout_W": self.dropout_W,
        "dropout_U": self.dropout_U,
        "sentinel": self.sentinel}
    base_config = super(LSTM_sent, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
