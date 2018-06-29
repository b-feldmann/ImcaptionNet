from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import preprocess_input
import numpy as np
import model as M
import args as P
from utils import config, dataloader


def trainloop(args, model, validation_model=None, epoch_start=0, suffix=''):
  pass


def init_models(args):
  model = M.get_model(args)
  optimizer = config.get_optimizer(args)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                sample_weight_mode="temporal")

  # Validation Model for EarlyStopping
  if not args.es_metric == 'loss':
    args.mode = 'test'
    validation_model = M.get_model(args)
    validation_model.compile(
        optimizer=optimizer, loss='categorical_crossentropy',
        sample_weight_mode="temporal")
    args.mode = 'train'
  else:
    validation_model = None

  return model, validation_model


if __name__ == '__main__':
  parser = P.get_parser()
  args = parser.parse_args()

  epoch_start = 0
  if not args.model_file:
    # No already trained model specified
    model, validation_model = init_models(args)
    _, model_name = trainloop(args, model, validation_model)
    epoch_start = args.nepochs

  model = M.get_model(args)
  optimizer = config.get_optimizer(args)

  if args.model_file:
    print('Loading model weights from snapshot: {}'.format(args.model_file))
    model.load_weights(args.model_file)
  else:
    model.load_weights(model_name)

  for i, layer in enumerate(model.layers[1].layers):
    if i > args.finetune_start_layer:
      layer.trainable = True

  model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                sample_weight_mode="temporal")

  # Train last N (=finetune_start_layer) of Encoder
  args.cnn_train = True
  args.mode = 'test'
  validation_model = M.get_model(args)
  validation_model.compile(
      optimizer=optimizer, loss='categorical_crossentropy',
      sample_weight_mode="temporal")
  args.mode = 'train'
  model, model_name = trainloop(
      args, model, suff_name='_cnn_train', model_val=validation_model,
      epoch_start=epoch_start)
