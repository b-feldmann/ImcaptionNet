import click
import numpy as np
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing import image

from adaptiveModel import Encoder2Decoder


@click.group()
def cli():
    pass


@cli.command()
@click.option('--imsize', default=224, help='Image Size')
@click.option('--vocab-path', help='Path to vocab')
def train(imsize):
    img_path = 'data/train2014/COCO_train2014_000000000009.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    adaptive = Encoder2Decoder(256, len(vocab), 512)


if __name__ == '__main__':
    cli()
