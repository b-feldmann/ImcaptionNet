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

  def reset_states(self):
    assert self.stateful, 'Layer must be stateful.'
    input_shape = self.input_spec[0].shape
    if not input_shape[0]:
      raise Exception('If a RNN is stateful, a complete ' +
                      'input_shape must be provided (including batch size).')
    if hasattr(self, 'states') and self.states[0] is not None:
      K.set_value(self.states[0],
                  np.zeros((input_shape[0], self.units)))
      K.set_value(self.states[1],
                  np.zeros((input_shape[0], self.units)))
    else:
      self.states = [K.zeros((input_shape[0], self.units)),
                     K.zeros((input_shape[0], self.units))]

  def build(self, input_shape):
    self.input_spec = [InputSpec(shape=input_shape)]

    # Note input_shape will be list of shapes of initial states and
    # constants if these are passed in __call__.
    if self._num_constants is not None:
      constants_shape = input_shape[-self._num_constants:]
    else:
      constants_shape = None

    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    batch_size = input_shape[0] if self.stateful else None
    input_dim = input_shape[-1]

    # allow cell (if layer) to build before we set or validate state_spec
    if isinstance(self.cell, Layer):
      # step_input_shape = (input_shape[0],) + input_shape[2:]
      if constants_shape is not None:
        # self.cell.build([step_input_shape] + constants_shape)
        self.cell.build(input_shape)
      else:
        # self.cell.build(step_input_shape)
        self.cell.build(input_shape)

    # set or validate state_spec
    if hasattr(self.cell.state_size, '__len__'):
      state_size = list(self.cell.state_size)
    else:
      state_size = [self.cell.state_size]

    if self.state_spec is not None:
      # initial_state was passed in call, check compatibility
      if [spec.shape[-1] for spec in self.state_spec] != state_size:
        raise ValueError(
            'An `initial_state` was passed that is not compatible with '
            '`cell.state_size`. Received `state_spec`={}; '
            'however `cell.state_size` is '
            '{}'.format(self.state_spec, self.cell.state_size))
    else:
      self.state_spec = [InputSpec(shape=(None, dim))
                         for dim in state_size]

    if self.stateful:
      self.reset_states()
    else:
      if self.sentinel:
        # initial states: 2 all-zero tensors of shape (units)
        self.states = [None, None]
      else:
        self.states = [None]

    self.built = True

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
    # input shape: (nb_samples, time (padded with zeros), input_dim)
    # note that the .build() method of subclasses MUST define
    # self.input_sepc with a complete input shape.
    input_shape = self.input_spec[0].shape

    if self.stateful:
      initial_states = self.states
    else:
      initial_states = self.get_initial_states(inputs)
    constants = self.get_constants(inputs)

    def step(inputs, states):
      return self.cell.call(inputs, states)

    last_output, outputs, states = K.rnn(step,
                                         inputs,
                                         initial_states,
                                         go_backwards=self.go_backwards,
                                         mask=mask,
                                         constants=constants,
                                         unroll=self.unroll,
                                         input_length=input_shape[1])

    if self.stateful:
      updates = []
      for i in range(len(states)):
          updates.append((self.states[i], states[i]))
      self.add_update(updates, inputs)

    # we need to reorder the batch position, as the default K.rnn()
    # assumes that the output is a single tensor.
    if self.sentinel:
      outputs = K.permute_dimensions(outputs, [0, 2, 1, 3])
      # returns a list where the first element
      # is the hidden state and the second sentinel
      if self.return_sequences:
        return [outputs[0], outputs[1]]
      else:
        return [last_output[0], last_output[1]]
    else:
      if self.return_sequences:
        return outputs
      else:
        return last_output

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
