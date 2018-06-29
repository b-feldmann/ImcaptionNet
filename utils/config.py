from tensorflow.python.keras.optimizers import (
    SGD, RMSprop, Adam, Adadelta, Adagrad
)
from tensorflow.python.keras.callbacks import Callback


def get_optimizer(args):
  opt_name = args.optimizer
  if opt_name is 'SGD':
    opt = SGD(lr=args.lr, decay=args.decay, momentum=0.9, nesterov=True,
              clipvalue=args.clip)
  elif opt_name is 'adam':
    opt = Adam(lr=args.lr, beta_1=args.alpha,
               beta_2=args.beta,
               decay=args.decay, clipvalue=args.clip)
  elif opt_name is 'adadelta':
    opt = Adadelta(lr=args.lr, decay=args.decay,
                   clipvalue=args.clip)
  elif opt_name is 'adagrad':
    opt = Adagrad(lr=args.lr, decay=args.decay,
                  clipvalue=args.clip)
  elif opt_name is 'rmsprop':
    opt = RMSprop(lr=args.lr, decay=args.decay,
                  clipvalue=args.clip)
  else:
    print("Unknown optimizer! Using Adam by default...")
    opt = Adam(lr=args.lr, decay=args.decay,
               clipvalue=args.clip)

  return opt


class ResetStatesCallback(Callback):
  def on_batch_end(self, batch, logs={}):
    self.model.reset_states()
