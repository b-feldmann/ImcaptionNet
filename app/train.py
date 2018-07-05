import json
import pickle

import math

import os
import torch
import numpy as np
from pycocoevalcap.eval import COCOEvalCap
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from build_vocab import Vocabulary
from adaptiveModel import Encoder2Decoder
from cocoapi2.PythonAPI.pycocotools.coco import COCO
from data_load import collate_fn, CocoDataset, get_loader, CocoEvalLoader
from evaluation import predict_captions, coco_metrics
from utils import to_var


def train_model(image_dir, caption_path, val_caption_path, vocab_path, learning_rate, num_epochs, lrd, lrd_every, alpha,
                beta, clip, logger_step, model_path, crop_size, batch_size, num_workers, cnn_learning_rate, shuffle,
                eval_size, evaluation_result_root, pretrained, max_steps=None):

    cider_scores = []
    best_epoch = 0
    best_cider_score = 0

    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

        # Image Preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    data_loader = get_loader(image_dir, caption_path, vocab, transform, batch_size, shuffle=shuffle, num_workers=num_workers)

    adaptive = Encoder2Decoder(256, len(vocab), 512)

    if pretrained is not None:
        print('Load pretrained')
        adaptive.load_state_dict(torch.load(pretrained))
        start_epoch = int(pretrained.split('/')[-1].split('-')[1].split('.')[0]) + 1
    else:
        start_epoch = 1
    # Constructing CNN parameters for optimization, only fine-tuning higher layers
    cnn_subs = list(adaptive.encoder.resnet_conv.children())[5:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]

    cnn_optimizer = torch.optim.Adam(cnn_params, lr=cnn_learning_rate,
                                     betas=(alpha, beta))

    params = list(adaptive.encoder.affine_a.parameters()) + list(adaptive.encoder.affine_b.parameters()) \
             + list(adaptive.decoder.parameters())


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

        # print('Start Epoch Evaluation')
        # Evaluate Model after epoch
        # epoch_score = evaluate_epoch(adaptive, image_dir, vocab, crop_size, val_caption_path, num_workers, eval_size, evaluation_result_root, epoch)
        # cider_scores.append(epoch_score)

        # print(f'Epoch {epoch}/{num_epochs}: CIDEr Score {epoch_score}')

        # if epoch_score > best_cider_score:
        #     best_cider_score = epoch_score
        #     best_epoch = epoch
        # if epoch > 20:
        #     if len(cider_scores) > 5:
        #         last_6 = cider_scores[-6:]
        #         last_6_max = max(last_6)
        #
        #         if last_6_max != best_cider_score:
        #             print('No improvements in the last 6 epochs')
        #             print(f'Model of best epoch #: {best_epoch} with CIDEr score {best_cider_score}')
        #             break


def evaluate_epoch(model, image_dir, vocab, crop_size, val_caption_path, num_workers, eval_size, evaluation_result_root, epoch):
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    eval_data_loader = torch.utils.data.DataLoader(
        CocoEvalLoader(image_dir, val_caption_path, transform),
        batch_size=eval_size,
        shuffle=False, num_workers=num_workers,
        drop_last=False)

    result_json = predict_captions(model, vocab, eval_data_loader)

    json.dump(result_json, open(evaluation_result_root + f'/evaluate-{epoch}.json', 'w'))

    return coco_metrics(val_caption_path, evaluation_result_root + f'/evaluate-{epoch}.json', 'CIDEr')
