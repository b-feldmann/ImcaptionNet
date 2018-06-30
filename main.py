import click
from build_vocab import Vocabulary
from evaluation import generate_result_json
from predict import single_image_predict
from resize_images import resize

from train import train_model


@click.group()
def cli():
    pass


@cli.command()
@click.option('--image_path', help='Absolut path to image')
@click.option('--model_path', help='Absolut path to model')
@click.option('--vocab_path', help='Absolut path to vocab')
@click.option('--crop_size', default=226, type=int, help='Data Augmentation Crop Size')
def predict(image_path, model_path, vocab_path, crop_size):
    single_image_predict(image_path, model_path, vocab_path, crop_size)


@cli.command()
@click.option('--caption_path')
@click.option('--image_path', help='Path to images')
@click.option('--vocab_path', help='Path to vocab')
@click.option('--model_path', help='Path to save model')
@click.option('--learning_rate', default=4e-4, type=float, help='Learning_rate for adaptive attention model')
@click.option('--cnn_learning_rate', default=1e-4, type=float, help='Learning_rate for CNN')
@click.option('--batch_size', default=1, type=int, help='Size of batches')
@click.option('--num_epochs', default=50, type=int, help='Number of epochs')
@click.option('--ld', default=20, type=int, help='Learning rate decay')
@click.option('--ld_every', default=50, type=int, help='Learning rate decay every')
@click.option('--alpha', default=0.8, type=int, help='Alpha for Adam')
@click.option('--beta', default=0.999, type=int, help='Beta for Adam')
@click.option('--clip', default=0.1, type=int, help='clip')
@click.option('--logger_step', default=10, type=int, help='Logger step')
@click.option('--num_workers', default=4, type=int, help='Number of workers')
@click.option('--crop_size', default=224, type=int, help='Data Augmentation Crop Size')
@click.option('--max_steps', default=None, type=int, help='Max number of images to train')
@click.option('--shuffle', default=True, type=bool, help='Shuffle dataset')
def train(image_path, caption_path, vocab_path, learning_rate, num_epochs, ld, ld_every, alpha, beta, clip, logger_step,
          model_path, crop_size, batch_size, num_workers, cnn_learning_rate, max_steps, shuffle):
    train_model(image_path, caption_path, vocab_path, learning_rate, num_epochs, ld, ld_every, alpha, beta, clip,
                logger_step, model_path, crop_size, batch_size, num_workers, cnn_learning_rate, max_steps, shuffle)


@cli.command()
@click.option('--image_dir', help='Path to images')
@click.option('--output_dir', help='Path to resized images')
def resize_images(image_dir, output_dir):
    resize(image_dir, output_dir, dataset_type='val', year = '2014', image_size = 256)


@cli.command()
@click.option('--model_path', help='Path to Model')
@click.option('--vocab_path', help='Path to Vocab')
@click.option('--image_root', help='Path to Image Directory')
@click.option('--val_caption_path', help='Path to Validation caption path')
@click.option('--result_path', help='Path for output file')
@click.option('--crop_size', default=224, type=int, help='Crop Size')
@click.option('--eval_size', default=28, type=int, help='Size of evaluation')
@click.option('--num_workers', default=4, type=int, help='Number of workers')
def generate_result_captions(model_path, vocab_path, image_root, val_caption_path, result_path, crop_size, eval_size, num_workers):
    generate_result_json(model_path, vocab_path, image_root, val_caption_path, result_path, crop_size, eval_size,
                         num_workers)


if __name__ == '__main__':
    cli()
