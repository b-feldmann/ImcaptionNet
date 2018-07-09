import argparse


def get_parser():
  p = argparse.ArgumentParser(description='ImcaptionNet (SpatialAttention)')

  # Model Parameter #
  p.add_argument('-mode', default='train', choices=['train', 'test'])
  p.add_argument('-bs', default=32, type=int)
  p.add_argument('-seqlen', default=18, type=int)
  p.add_argument('-emb_dim', default=512, type=int)
  p.add_argument('-lstm_dim', default=512, type=int)
  p.add_argument('-dr_ratio', default=0.5, type=float)
  p.add_argument('-z_dim', default=512, type=int)
  p.add_argument('-finetune_start_layer', default=18, type=int,
                 help='6 for Resnet50, 18 for InceptionResNetV2')

  p.add_argument('-imgw', default=256, type=int)
  p.add_argument('-imgh', default=256, type=int)

  p.add_argument('--dr', default=False, action='store_true')
  p.add_argument('--bn', default=False, action='store_true')
  p.add_argument('--sgate', default=True, action='store_true')
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
  p.add_argument('-es_prev_words', default='gen',
                 choices=['gt', 'gen'])

  # Model Save/Restoer Parameter #
  p.add_argument('-model_file', default=None)
  p.add_argument('-model_name', default='model')

  # Filesystem Parameter #
  p.add_argument('-data_folder', default='../imcaptionnet-model/')
  p.add_argument('-train_img_dir',
                 default='/data/dl_lecture_data/TrainVal/train2014')
  p.add_argument('-train_coco_file',
                 default='/data/dl_lecture_data/TrainVal/annotations/captions_train2014.json')
  p.add_argument(
      '-val_img_dir', default='/data/dl_lecture_data/TrainVal/val2014')
  p.add_argument('-val_coco_file',
                 default='/data/dl_lecture_data/TrainVal/annotations/captions_val2014.json')
  p.add_argument('-preprocessed', default=False, action='store_true')

  # Train Parameter #
  p.add_argument('-current_lang_epoch', default=0, type=int)
  p.add_argument('-lang_epochs', default=20, type=int)
  p.add_argument('-current_cnn_epoch', default=0, type=int)
  p.add_argument('-cnn_epochs', default=20, type=int)
  p.add_argument('-patience', default=5, type=int)
  p.add_argument('-val_samples', default=640, type=int)
  p.add_argument('-train_samples', default=-1, type=int)
  p.add_argument('-gpus', default=1, type=int)

  # Vocabulary Parameter #
  p.add_argument('-vocab_size', default=9591, type=int)
  p.add_argument('-word_count_threshold', default=5, type=int)
  p.add_argument('-tokenizer', default='nltk',
                 choices=['nltk', 'neuraltalk'])

  return p
