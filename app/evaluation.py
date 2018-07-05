import json
import pickle
import torch
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap
from torchvision import transforms

from adaptiveModel import Encoder2Decoder
from cocoapi2.PythonAPI.pycocotools.coco import COCO
from data_load import CocoEvalLoader
from utils import to_var


def predict_captions(model, vocab, data_loader):
    result_json = []
    for i, (tensor, image_ids, _) in enumerate(data_loader):
        predicted_captions, _, _ = model.sampler(to_var(tensor))
        if torch.cuda.is_available():
            captions = predicted_captions.cpu().data.numpy()
        else:
            captions = predicted_captions.data.numpy()

        for token_ids in range(captions.shape[0]):
            tokens = captions[token_ids]

            generated_captions = []

            for word in tokens:
                word = vocab.idx2word[word]

                if word == '<end>':
                    break
                else:
                    generated_captions.append(word)

            result_json.append({'image_id': int(image_ids[token_ids]), 'caption': " ".join(generated_captions)})
        if (i + 1) % 10 == 0:
            print(f'[{i+1}/{len(data_loader)}]')

    return result_json


def generate_result_json(model_path, vocab_path, image_root, val_caption_path, result_path, crop_size, eval_size,
                         num_workers):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    model = Encoder2Decoder(256, len(vocab), 512)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))

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


def coco_metrics(val_captions_file, result_captions, metric):
    coco = COCO(val_captions_file)
    cocoRes = coco.loadRes(result_captions)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    return cocoEval.eval[metric]