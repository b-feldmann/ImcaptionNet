import pickle
import torch
from PIL import Image
from torchvision import transforms

from adaptiveModel import Encoder2Decoder
from utils import to_var


def single_image_predict(image_path, model_path, vocab_path, alpha, beta, cnn_learning_rate, crop_size):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    model = Encoder2Decoder(256, len(vocab), 512)
    model.load_state_dict(torch.load(model_path))

    cnn_subs = list(model.encoder.resnet_conv.children())[5:]
    cnn_params = [list(sub_module.parameters()) for sub_module in cnn_subs]
    cnn_params = [item for sublist in cnn_params for item in sublist]

    transform = transforms.Compose([
        transforms.Scale((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    LMcriterion = torch.nn.CrossEntropyLoss()

    # Change to GPU mode if available
    if torch.cuda.is_available():
        model.cuda()
        LMcriterion.cuda()

    image = Image.open(image_path).convert('RGB')
    image_activation = to_var(transform(image))
    print(model.sampler(image_activation))