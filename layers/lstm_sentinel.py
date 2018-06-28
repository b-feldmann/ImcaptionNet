from tensorflow.python.keras.layers import RNN
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer
from layers.lstm_sentinel_cell import LSTMSentinelCell
import numpy as np


class LSTMSentinel(RNN):

  def __init__(self, units,
               init='glorot_uniform', inner_init='orthogonal',
               forget_bias_init='one', activation='tanh',
               inner_activation='hard_sigmoid',
               W_regularizer=None, U_regularizer=None, b_regularizer=None,
               dropout_W=0.,
               dropout_U=0.,
               sentinel=True,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    self.sentinel = sentinel
    self.dropout_U = dropout_U
    self.dropout_W = dropout_W
    cell = LSTMSentinelCell(
        units,
        init=init,
        inner_init=inner_init,
        forget_bias_init=forget_bias_init,
        activation=activation,
        inner_activation=inner_activation,
        W_regularizer=W_regularizer,
        U_regularizer=U_regularizer,
        b_regularizer=b_regularizer,
        dropout_U=dropout_U,
        dropout_W=dropout_W,
        sentinel=sentinel,
        **kwargs)
    super(LSTMSentinel, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)

  def get_constants(self, x):
    constants = []
    if self.sentinel:
      Ngate = 5
    else:
      Ngate = 4
    if 0 < self.dropout_U < 1:
      ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
      ones = K.concatenate([ones] * self.units, 1)
      B_U = [K.dropout(ones, self.dropout_U) for _ in range(Ngate)]
      constants.append(B_U)
    else:
      constants.append([K.cast_to_floatx(1.) for _ in range(Ngate)])

    constants.append([K.cast_to_floatx(1.) for _ in range(Ngate)])
    return constants

  def call(self, inputs, mask=None,
           training=None,
           initial_state=None,
           constants=None):
    constants = self.get_constants(inputs)
    self._num_constants = 2
    return super(LSTMSentinel, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state,
        constants=constants)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  @property
  def reset_after(self):
    return self.cell.reset_after

  def get_config(self):
    config = {'return_sequences': self.return_sequences,
              'return_state': self.return_state,
              'go_backwards': self.go_backwards,
              'stateful': self.stateful,
              'unroll': self.unroll}
    if self._num_constants is not None:
      config['num_constants'] = self._num_constants

    cell_config = self.cell.get_config()
    config['cell'] = {'class_name': self.cell.__class__.__name__,
                      'config': cell_config}
    base_config = super(RNN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_mask(self, input, mask):
    if self.return_sequences:
      if self.sentinel:
        return [mask, mask]
      else:
        return mask
    else:
      if self.sentinel:
        return [None, None]
      else:
        return None
