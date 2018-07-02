import tensorflow as tf
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import (
    Input, Dense, Activation, Permute, Concatenate, Add, Lambda, Multiply,
    RepeatVector, Dropout, Reshape, Conv1D, GlobalAveragePooling1D, Embedding,
    TimeDistributed
)
from tensorflow.python.keras import backend as K
from utils.multi_gpu_utils import multi_gpu_model
from layers.lstm_sentinel import LSTMSentinel


def get_encoder_model(inp, inp_shape):
  base_resnet50 = ResNet50(
      include_top=False, weights='imagenet',
      input_tensor=inp,
      input_shape=inp_shape)
  # EXPERIMENT: Testen ob wir die Activation (-2) oder das Pooling (-1) benutzen
  encoder = Model(
      # name='Encoder-Resnet50',
      inputs=base_resnet50.input,
      outputs=[base_resnet50.layers[-2].output])
  return encoder


def get_decoder_model(args, wh, dim, convfeats, prev_words):
  num_classes = args.vocab_size
  if args.mode == 'train':
    seqlen = args.seqlen
  else:
    seqlen = 1

  # imfeats need to be "flattened" eg 15x15x512 --> 225x512
  V = Reshape((wh * wh, dim), name='conv_feats')(convfeats)  # 225x512

  # input is the average of conv feats
  Vg = GlobalAveragePooling1D(name='Vg')(V)
  # embed average imfeats
  Vg = Dense(args.emb_dim, activation='relu', name='Vg_')(Vg)
  if args.dr:
    Vg = Dropout(args.dr_ratio)(Vg)

  # we keep spatial image feats to compute context vector later
  # project to z_space
  Vi = Conv1D(args.z_dim, 1, padding='same',
              activation='relu', name='Vi')(V)

  if args.dr:
    Vi = Dropout(args.dr_ratio)(Vi)
  # embed
  Vi_emb = Conv1D(args.emb_dim, 1, padding='same',
                  activation='relu', name='Vi_emb')(Vi)

  # repeat average feat as many times as seqlen to infer output size
  x = RepeatVector(seqlen)(Vg)  # seqlen,512

  # embedding for previous words
  wemb = Embedding(num_classes, args.emb_dim, input_length=seqlen)
  emb = wemb(prev_words)
  emb = Activation('relu')(emb)
  if args.dr:
    emb = Dropout(args.dr_ratio)(emb)

  x = Concatenate(name='lstm_in')([x, emb])
  if args.dr:
    x = Dropout(args.dr_ratio)(x)
  if args.sgate:
    # lstm with two outputs
    lstm_ = LSTMSentinel(args.lstm_dim,
                         return_sequences=True, stateful=True,
                         dropout_W=args.dr_ratio,
                         dropout_U=args.dr_ratio,
                         sentinel=True, name='hs')
    h_ = lstm_(x)
    h = Lambda(lambda x: x[:, :, :args.lstm_dim])(h_)
    s = Lambda(lambda x: x[:, :, args.lstm_dim:])(h_)
  else:
    # regular lstm
    lstm_ = LSTMSentinel(args.lstm_dim,
                         return_sequences=True, stateful=True,
                         dropout_W=args.dr_ratio,
                         dropout_U=args.dr_ratio,
                         sentinel=False, name='h')
    h = lstm_(x)

  num_vfeats = wh * wh
  if args.sgate:
    num_vfeats = num_vfeats + 1

  if args.attlstm:

    # embed ht vectors.
    # linear used as input to final classifier, embedded ones are used to compute attention
    h_out_linear = Conv1D(
        args.z_dim, 1, activation='tanh', name='zh_linear', padding='same')(h)
    if args.dr:
      h_out_linear = Dropout(args.dr_ratio)(h_out_linear)
    h_out_embed = Conv1D(
        args.emb_dim, 1, name='zh_embed', padding='same')(h_out_linear)
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

    if args.sgate:

      # embed sentinel vec
      # linear used as additional feat to apply attention, embedded used as add. feat to compute attention
      fake_feat = Conv1D(
          args.z_dim, 1, activation='relu', name='zs_linear',
          padding='same')(s)
      if args.dr:
        fake_feat = Dropout(args.dr_ratio)(fake_feat)

      fake_feat_embed = Conv1D(
          args.emb_dim, 1, name='zs_embed', padding='same')(fake_feat)
      # reshape for merging with visual feats
      z_s_linear = Reshape((seqlen, 1, args.z_dim))(fake_feat)
      z_s_embed = Reshape((seqlen, 1, args.emb_dim))(fake_feat_embed)

      # concat fake feature to the rest of image features
      z_v_linear = Concatenate(-2)([z_v_linear, z_s_linear])
      z_v_embed = Concatenate(-2)([z_v_embed, z_s_embed])

    # sum outputs from z_v and z_h
    z = Add(name='merge_v_h')([z_h_embed, z_v_embed])
    if args.dr:
      z = Dropout(args.dr_ratio)(z)
    z = TimeDistributed(Activation('tanh', name='merge_v_h_tanh'))(z)
    # compute attention values
    att = TimeDistributed(Conv1D(1, 1, padding='same'), name='att')(z)

    att = Reshape((seqlen, num_vfeats), name='att_res')(att)
    # softmax activation
    att = TimeDistributed(Activation('softmax'), name='att_scores')(att)
    att = TimeDistributed(RepeatVector(args.z_dim), name='att_rep')(att)
    att = Permute((1, 3, 2), name='att_rep_p')(att)

    # get context vector as weighted sum of image features using att
    w_Vi = Multiply()([att, z_v_linear])
    sumpool = Lambda(lambda x: K.sum(x, axis=-2),
                     output_shape=(args.z_dim,))
    c_vec = TimeDistributed(sumpool, name='c_vec')(w_Vi)
    atten_out = Add(name='sum_mlp_in')([h_out_linear, c_vec])
    h = TimeDistributed(
        Dense(args.emb_dim, activation='tanh'))(atten_out)
    if args.dr:
      h = Dropout(args.dr_ratio, name='mlp_in_tanh_dp')(h)

  predictions = TimeDistributed(
      Dense(num_classes, activation='softmax'), name='out')(h)

  model = Model(inputs=[convfeats, prev_words], outputs=predictions)
  return model


def get_model(args):
  if args.mode == 'train':
    seqlen = args.seqlen
  else:
    seqlen = 1
  encoder_input_shape = (args.imsize, args.imsize, 3)

  encoder_input = Input(batch_shape=(
      args.model_bs, args.imsize, args.imsize, 3), name='image')
  encoder = get_encoder_model(encoder_input, encoder_input_shape)

  wh = encoder.output_shape[1]  # size of conv5
  dim = encoder.output_shape[3]  # number of channels

  if not args.cnn_train:
    for i, layer in enumerate(encoder.layers):
      if i > args.finetune_start_layer:
        layer.trainable = False

  encoder_output = encoder(encoder_input)
  convfeats = Input(batch_shape=(args.model_bs, wh, wh, dim), name='convfeats')
  prev_words = Input(batch_shape=(args.model_bs, seqlen), name='prev_words')
  decoder = get_decoder_model(args, wh, dim, convfeats, prev_words)

  decoder_output = decoder([encoder_output, prev_words])

  if args.gpus > 1:
    with tf.device('/cpu:0'):
      model = Model(inputs=[encoder_input, prev_words], outputs=decoder_output)

    model = multi_gpu_model(model, args.gpus)
  else:
    model = Model(inputs=[encoder_input, prev_words], outputs=decoder_output)

  return model
