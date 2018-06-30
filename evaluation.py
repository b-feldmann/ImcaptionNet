import json
import pickle
import torch
from PIL import Image
from torchvision import transforms

from adaptiveModel import Encoder2Decoder
from data_load import CocoEvalLoader
from utils import to_var


def predict_captions(model, vocab, data_loader):
    result_json = []
    for i, (images, image_ids, _) in enumerate(data_loader):
        predicted_captions, _, _ = model.sampler(images)
        if torch.cuda.is_available():
            captions = predicted_captions.cpu().data.numpy()
        else:
            captions = predicted_captions.data.numpy()

        for tokens in range(captions.shape[0]):
            token_ids = captions[tokens]

            generated_captions = []

            for word in token_ids:
                word = vocab.idx2word[word]

                if word == '<end>':
                    break
                else:
                    generated_captions.append(word)

            result_json.append({'image_id': int(image_ids[token_ids]), 'sentence': " ".join(generated_captions)})

    return result_json


def generate_result_json(model_path, vocab_path, image_root, val_caption_path, result_path, crop_size, eval_size,
                         num_workers):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    model = Encoder2Decoder(256, len(vocab), 512)

    model.load_state_dict(torch.load(model_path))

    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    eval_data_loader = torch.utils.data.DataLoader(
        CocoEvalLoader(image_root, val_caption_path, transform),
        batch_size=eval_size,
        shuffle=False, num_workers=num_workers,
        drop_last=False)

    result_json = predict_captions(model, vocab, eval_data_loader)

    json.dump(result_json, open(result_path, 'w'))
