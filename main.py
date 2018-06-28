import tensorflow as tf

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense, Activation, Permute, Concatenate, Add, Lambda, Multiply
from tensorflow.python.keras.layers import RepeatVector, Dropout, Reshape
from tensorflow.python.keras.layers import Conv1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras import backend as K
from layers.lstm_sentinel import LSTMSentinel
import numpy as np


class Args(object):
  pass


def get_encoder_model(inp, inp_shape):
  base_resnet50 = tf.keras.applications.resnet50.ResNet50(
      include_top=False, weights='imagenet',
      input_tensor=inp,
      input_shape=inp_shape)
  # EXPERIMENT: Testen ob wir die Activation (-2) oder das Pooling (-1) benutzen
  encoder = Model(
      # name='Encoder-Resnet50',
      inputs=base_resnet50.input,
      outputs=[base_resnet50.layers[-2].output])
  return encoder


def get_decoder_model(args_dict, wh, dim, convfeats, prev_words):
  num_classes = args_dict.vocab_size
  seqlen = args_dict.seqlen

  # imfeats need to be "flattened" eg 15x15x512 --> 225x512
  V = Reshape((wh * wh, dim), name='conv_feats')(convfeats)  # 225x512

  # input is the average of conv feats
  Vg = GlobalAveragePooling1D(name='Vg')(V)
  # embed average imfeats
  Vg = Dense(args_dict.emb_dim, activation='relu', name='Vg_')(Vg)
  if args_dict.dr:
    Vg = Dropout(args_dict.dr_ratio)(Vg)

  # we keep spatial image feats to compute context vector later
  # project to z_space
  Vi = Conv1D(args_dict.z_dim, 1, padding='same',
              activation='relu', name='Vi')(V)

  if args_dict.dr:
    Vi = Dropout(args_dict.dr_ratio)(Vi)
  # embed
  Vi_emb = Conv1D(args_dict.emb_dim, 1, padding='same',
                  activation='relu', name='Vi_emb')(Vi)

  # repeat average feat as many times as seqlen to infer output size
  x = RepeatVector(seqlen)(Vg)  # seqlen,512

  # embedding for previous words
  wemb = Embedding(num_classes, args_dict.emb_dim, input_length=seqlen)
  emb = wemb(prev_words)
  emb = Activation('relu')(emb)
  if args_dict.dr:
    emb = Dropout(args_dict.dr_ratio)(emb)

  x = Concatenate(name='lstm_in')([x, emb])
  if args_dict.dr:
    x = Dropout(args_dict.dr_ratio)(x)
  if args_dict.sgate:
    # lstm with two outputs
    lstm_ = LSTMSentinel(args_dict.lstm_dim,
                         return_sequences=True, stateful=True,
                         dropout_W=args_dict.dr_ratio,
                         dropout_U=args_dict.dr_ratio,
                         sentinel=True, name='hs')
    h_ = lstm_(x)
    h = Lambda(lambda x: x[:, :, :args_dict.lstm_dim])(h_)
    s = Lambda(lambda x: x[:, :, args_dict.lstm_dim:])(h_)
  else:
    # regular lstm
    lstm_ = LSTMSentinel(args_dict.lstm_dim,
                         return_sequences=True, stateful=True,
                         dropout_W=args_dict.dr_ratio,
                         dropout_U=args_dict.dr_ratio,
                         sentinel=False, name='h')
    h = lstm_(x)

  num_vfeats = wh * wh
  if args_dict.sgate:
    num_vfeats = num_vfeats + 1

  if args_dict.attlstm:

    # embed ht vectors.
    # linear used as input to final classifier, embedded ones are used to compute attention
    h_out_linear = Conv1D(
        args_dict.z_dim, 1, activation='tanh', name='zh_linear', padding='same')(h)
    if args_dict.dr:
      h_out_linear = Dropout(args_dict.dr_ratio)(h_out_linear)
    h_out_embed = Conv1D(
        args_dict.emb_dim, 1, name='zh_embed', padding='same')(h_out_linear)
    # repeat all h vectors as many times as local feats in v
    # ERROR
    z_h_embed = TimeDistributed(RepeatVector(num_vfeats))(h_out_embed)

    # repeat all image vectors as many times as timesteps (seqlen)
    # linear feats are used to apply attention, embedded feats are used to compute attention
    z_v_linear = TimeDistributed(RepeatVector(seqlen), name='z_v_linear')(Vi)
    z_v_embed = TimeDistributed(
        RepeatVector(seqlen), name='z_v_embed')(Vi_emb)

    z_v_linear = Permute((2, 1, 3))(z_v_linear)
    z_v_embed = Permute((2, 1, 3))(z_v_embed)

    if args_dict.sgate:

      # embed sentinel vec
      # linear used as additional feat to apply attention, embedded used as add. feat to compute attention
      fake_feat = Conv1D(
          args_dict.z_dim, 1, activation='relu', name='zs_linear', padding='same')(s)
      if args_dict.dr:
        fake_feat = Dropout(args_dict.dr_ratio)(fake_feat)

      fake_feat_embed = Conv1D(
          args_dict.emb_dim, 1, name='zs_embed', padding='same')(fake_feat)
      # reshape for merging with visual feats
      z_s_linear = Reshape((seqlen, 1, args_dict.z_dim))(fake_feat)
      z_s_embed = Reshape((seqlen, 1, args_dict.emb_dim))(fake_feat_embed)

      # concat fake feature to the rest of image features
      z_v_linear = Concatenate(-2)([z_v_linear, z_s_linear])
      z_v_embed = Concatenate(-2)([z_v_embed, z_s_embed])

    # sum outputs from z_v and z_h
    z = Add(name='merge_v_h')([z_h_embed, z_v_embed])
    if args_dict.dr:
      z = Dropout(args_dict.dr_ratio)(z)
    z = TimeDistributed(Activation('tanh', name='merge_v_h_tanh'))(z)
    # compute attention values
    att = TimeDistributed(Conv1D(1, 1, padding='same'), name='att')(z)

    att = Reshape((seqlen, num_vfeats), name='att_res')(att)
    # softmax activation
    att = TimeDistributed(Activation('softmax'), name='att_scores')(att)
    att = TimeDistributed(RepeatVector(args_dict.z_dim), name='att_rep')(att)
    att = Permute((1, 3, 2), name='att_rep_p')(att)

    # get context vector as weighted sum of image features using att
    w_Vi = Multiply()([att, z_v_linear])
    sumpool = Lambda(lambda x: K.sum(x, axis=-2),
                     output_shape=(args_dict.z_dim,))
    c_vec = TimeDistributed(sumpool, name='c_vec')(w_Vi)
    print(h_out_linear)
    print(c_vec)
    # c_vec = [?,18,64,512] Expected: [1,18,512]
    atten_out = Add(name='sum_mlp_in')([h_out_linear, c_vec])
    # atten_out = Add(name='sum_mlp_in')([h_out_linear, w_Vi])
    h = TimeDistributed(
        Dense(args_dict.emb_dim, activation='tanh'))(atten_out)
    if args_dict.dr:
      h = Dropout(args_dict.dr_ratio, name='mlp_in_tanh_dp')(h)

  predictions = TimeDistributed(
      Dense(num_classes, activation='softmax'), name='out')(h)

  model = Model(inputs=[convfeats, prev_words], outputs=predictions)
  return model


def get_model(args_dict):
  encoder_input_shape = (args_dict.imsize, args_dict.imsize, 3)

  encoder_input = Input(batch_shape=(
      args_dict.bs, args_dict.imsize, args_dict.imsize, 3), name='image')
  encoder = get_encoder_model(encoder_input, encoder_input_shape)

  wh = encoder.output_shape[1]  # size of conv5
  dim = encoder.output_shape[3]  # number of channels

  if not args_dict.cnn_train:
    for i, layer in enumerate(encoder.layers):
      if i > args_dict.finetune_start_layer:
        layer.trainable = False

  encoder_output = encoder(encoder_input)
  convfeats = Input(batch_shape=(args_dict.bs, wh, wh, dim), name='convfeats')
  prev_words = Input(batch_shape=(
      args_dict.bs, args_dict.seqlen), name='prev_words')
  decoder = get_decoder_model(args_dict, wh, dim, convfeats, prev_words)

  decoder_output = decoder([encoder_output, prev_words])

  model = Model(inputs=[encoder_input, prev_words], outputs=decoder_output)

  return model


class MyModel(Model):

  def __init__(self, encoder, decoder):
    super(MyModel, self).__init__()
    self.encoder = encoder
    # print(encoder.summary())
    self.decoder = decoder
    # print(decoder.summary())

  def call(self, inputs):
    return self.decoder([self.encoder(inputs[0]), inputs[1]])


if __name__ == '__main__':
  # Hardcoded commandline parameter
  args_dict = Args()
  args_dict.mode = 'train'
  args_dict.bs = 2
  args_dict.seqlen = 18
  args_dict.vocab_size = 9570
  args_dict.emb_dim = 512
  args_dict.imsize = 256
  args_dict.lstm_dim = 512
  args_dict.dr = False
  args_dict.dr_ratio = 0.5
  args_dict.bn = False
  args_dict.sgate = True
  args_dict.attlstm = True
  args_dict.z_dim = 512
  args_dict.cnn_train = False
  args_dict.finetune_start_layer = 6

  img_path = '.data/train2014/COCO_train2014_000000000009.jpg'
  img = image.load_img(img_path, target_size=(256, 256))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  model = get_model(args_dict)
  print(model.summary())

  # example_text = np.array([1] * args_dict.seqlen)
  # example_text = np.expand_dims(example_text, axis=0)

  # print(x.shape)
  # print(example_text.shape)

  example_text = np.ones([args_dict.bs, 18])
  img = np.ones([args_dict.bs, 256, 256, 3])

  features = model.predict([img, example_text])
  print([v.shape for v in features])
