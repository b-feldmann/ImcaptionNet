import click
from build_vocab import Vocabulary

from train import train_model


@click.group()
def cli():
    pass



@cli.command()
@click.option('--caption_path')
@click.option('--vocab_path', help='Path to vocab')
@click.option('--model_path', help='Path to save model')
@click.option('--learning_rate', default=4e-4, help='Learning_rate')
@click.option('--num_epochs', default=50, help='Number of epochs')
@click.option('--ld', default=20, help='Learning rate decay')
@click.option('--ld_every', default=50, help='Learning rate decay every')
@click.option('--alpha', default=0.8, help='Alpha for Adam')
@click.option('--beta', default=0.999, help='Beta for Adam')
@click.option('--clip', default=0.1, help='clip')
@click.option('--logger_step', default=10, help='Logger step')
def train(caption_path, vocab_path, learning_rate, num_epochs, ld, ld_every, alpha, beta, clip, logger_step, model_path):
    train_model(caption_path, vocab_path, learning_rate, num_epochs, ld, ld_every, alpha, beta, clip, logger_step, model_path)


if __name__ == '__main__':
    cli()
