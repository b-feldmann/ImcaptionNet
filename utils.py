import torch
from matplotlib import image, pyplot
from torch.autograd import Variable


def to_var(x, volatile=False):
    '''
    Wrapper torch tensor into Variable
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable( x, volatile=volatile )


def show_image(path):
    img = image.imread(path)
    imgplot = pyplot.imshow(img)
    pyplot.show(imgplot)