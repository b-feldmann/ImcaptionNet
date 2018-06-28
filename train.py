import pickle

import math

import os
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from build_vocab import Vocabulary
from adaptiveModel import Encoder2Decoder
from data_load import collate_fn
from utils import to_var


def train_model(caption_path, vocab_path, learning_rate, num_epochs, lrd, lrd_every, alpha, beta, clip, logger_step,
                model_path):
    data_loader = DataLoader(caption_path, batch_size=52, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    adaptive = Encoder2Decoder(256, len(vocab), 512)

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

        print(f'Learning Rate Epoch {epoch}: {"{0:.2f}".format(learning_rate)}')
        optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(alpha, beta))

        print(f'Training for Epoch {epoch}')

        for i, (images, captions, lengths, _, _) in enumerate(data_loader):
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

            if i % logger_step == 0:
                print(f'Epoch {epoch}/{num_epochs}, Step {i}/{num_steps}, CrossEntropy Loss: {loss.data[0]}, Perplexity: {np.exp( loss.data[0])}')

            torch.save(adaptive.state_dict(), os.path.join(model_path, f'adaptive-{epoch}.pkl'))
