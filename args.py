import argparse


def get_parser():
  p = argparse.ArgumentParser(description='ImcaptionNet (SpatialAttention)')

  # Model Parameter #
  p.add_argument('-mode', default='train', choices=['train', 'test'])
  p.add_argument('-bs', default=32, type=int)
  p.add_argument('-seqlen', default=18, type=int)
  p.add_argument('-vocab_size', default=9570, type=int)
  p.add_argument('-emb_dim', default=512, type=int)
  p.add_argument('-imsize', default=256, type=int)
  p.add_argument('-lstm_dim', default=512, type=int)
  p.add_argument('-dr_ratio', default=0.5, type=float)
  p.add_argument('-z_dim', default=512, type=int)
  p.add_argument('-finetune_start_layer', default=6, type=int)

  p.add_argument('--dr', default=False, action='store_true')
  p.add_argument('--bn', default=False, action='store_true')
  p.add_argument('--sgate', default=False, action='store_true')
  p.add_argument('--attlstm', default=True, action='store_true')
  p.add_argument('--cnn_train', default=False, action='store_true')

  # Optimizer Parameter #
  p.add_argument('-optimizer', default='adam',
                 choices=['adam', 'SGD', 'adadelta', 'adagrad', 'rmsprop'])
  p.add_argument('-lr', default=5e-4, type=float)
  p.add_argument('-decay', default=0.0, type=float)
  p.add_argument('-clip', default=5, type=float)
  p.add_argument('-alpha', default=0.9, type=float)
  p.add_argument('-beta', default=0.999, type=float)

  # Callback / Validation Parameter #
  p.add_argument('-es_metric', default='CIDEr',
                 choices=['loss', 'CIDEr', 'Bleu_4', 'Bleu_3', 'Bleu_2',
                          'Bleu_1', 'ROUGE_L', 'METEOR'])

  # Filesystem Parameter #
  p.add_argument('-model_file', default=None)

  # Train Parameter #
  p.add_argument('-nepochs', default=20, type=int)

  return p
