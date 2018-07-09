import json
import pickle
import torch
from PIL import Image
from torchvision import transforms
from adaptiveModel import Encoder2Decoder


def load_json(root):
    with open(root + '/input.json') as file:
        json_data = json.load(file)
    return json_data['images']


def single_image_predict(image_path, model, vocab, transform, image_size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([image_size, image_size], Image.ANTIALIAS)
    tensor = transform(image)
    predicted_captions, _, _ = model.sampler(tensor.unsqueeze_(0))

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

        sentence = " ".join(generated_captions)

    return sentence


def generate_predicted_json(image_dir, model_path, vocab_path, result_json_path, crop_size, image_size, use_filenames):
    result_json = []

    images = load_json(image_dir)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    model = Encoder2Decoder(256, len(vocab), 512)

    model.eval()

    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    for i, image in enumerate(images):
        filename = image['file_name']
        id = image['id']
        sentence = single_image_predict(image_dir + '/' + filename, model, vocab, transform, image_size)
        if use_filenames:
            result_json.append({'image_id': filename, 'caption': sentence})
        else:
            result_json.append({'image_id': id, 'caption': sentence})

    json.dump(result_json, open(result_json_path, 'w'))

