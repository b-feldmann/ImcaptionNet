import pickle
import torch
from PIL import Image
from torchvision import transforms

from adaptiveModel import Encoder2Decoder
from utils import to_var, show_image


def single_image_predict(image_path, model_path, vocab_path, crop_size):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    model = Encoder2Decoder(256, len(vocab), 512)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image = Image.open(image_path).convert('RGB')
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
        print('PREDICTED CAPTION:')
        print(sentence)
        show_image(image_path)


