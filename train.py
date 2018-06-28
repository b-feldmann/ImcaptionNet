import pickle

import math

import os
import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from build_vocab import Vocabulary
from adaptiveModel import Encoder2Decoder
from data_load import collate_fn, CocoDataset, get_loader
from utils import to_var


def train_model(image_dir, caption_path, vocab_path, learning_rate, num_epochs, lrd, lrd_every, alpha, beta, clip,
                logger_step, model_path, crop_size, batch_size, num_workers, cnn_learning_rate, max_steps=None):

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

        # Image Preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    data_loader = get_loader(image_dir, caption_path, vocab,transform, batch_size, shuffle=True, num_workers=num_workers)

    adaptive = Encoder2Decoder(256, len(vocab), 512)

    # Constructing CNN parameters for optimization, only fine-tuning higher layers
    cnn_subs = list(adaptive.encoder.resnet_conv.children())[5:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]

    cnn_optimizer = torch.optim.Adam(cnn_params, lr=cnn_learning_rate,
                                     betas=(alpha, beta))

    params = list(adaptive.encoder.affine_a.parameters()) + list(adaptive.encoder.affine_b.parameters()) \
             + list(adaptive.decoder.parameters())

    start_epoch = 1

    LMcriterion = nn.CrossEntropyLoss()

    # Change to GPU mode if available
    if torch.cuda.is_available():
        adaptive.cuda()
        LMcriterion.cuda()

    num_steps = len(data_loader)

    for epoch in range(start_epoch, num_epochs + 1):
        if epoch > lrd:
            frac = float(epoch - lrd) / lrd_every
            decay_factor = math.pow(0.5, frac)

            learning_rate = lrd * decay_factor

        print(f'Learning Rate Epoch {epoch}: {"{0:.6f}".format(learning_rate)}')
        optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(alpha, beta))

        print(f'Training for Epoch {epoch}')

        for i, (images, captions, lengths, _, _) in enumerate(data_loader):
            if max_steps is not None:
                if i > max_steps:
                    break
            images = to_var(images)
            captions = to_var(captions)
            lengths = [cap_len - 1 for cap_len in lengths]
            targets = pack_padded_sequence(captions[:, 1:], lengths, batch_first=True)[0]

            adaptive.train()
            adaptive.zero_grad()

            packed_scores = adaptive(images, captions, lengths)

            loss = LMcriterion(packed_scores[0], targets)
            loss.backward()

            for p in adaptive.decoder.LSTM.parameters():
                p.data.clamp_(-clip, clip)

            optimizer.step()

            if epoch > 20:
                cnn_optimizer.step()

        if i % logger_step == 0:
            print(f'Epoch {epoch}/{num_epochs}, Step {i}/{num_steps}, CrossEntropy Loss: {loss.item()}, Perplexity: {np.exp(loss.item())}')

        torch.save(adaptive.state_dict(), os.path.join(model_path, f'adaptive-{epoch}.pkl'))
