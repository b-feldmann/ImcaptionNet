import glob
import os
import math
import nltk
import random
import numpy as np
import time
from PIL import Image
from tensorflow.python.keras.utils import Sequence, to_categorical
from pycocotools.coco import COCO
from tensorflow.python.keras.applications.resnet50 import preprocess_input


class COCOSequence(Sequence):

  def __init__(self, img_directory, coco, vocab_size, seqlen, batch_size,
               imgw=256, imgh=256, preprocessed=False):
    self.img_directory = img_directory
    self.coco = coco
    self.ann_ids = list(self.coco.anns.keys())
    self.seqlen = seqlen
    self.vocab_size = vocab_size
    self.batch_size = batch_size
    self.imgw = imgw
    self.imgh = imgh
    self.return_image_ids = False
    # If True, images will only be loaded from disk
    # If False, resize and transform to RGB
    self.preprocessed = preprocessed

  def __len__(self):
    return math.ceil(len(list(self.ann_ids)) / self.batch_size)

  def __getitem__(self, index):
    start = time.time()
    batch_indices = self.ann_ids[index *
                                 self.batch_size:(index + 1) * self.batch_size]

    batch_img = np.zeros([self.batch_size, self.imgw, self.imgh, 3])
    batch_label_input = np.zeros([self.batch_size, self.seqlen])
    batch_label_expected = np.zeros([self.batch_size, self.seqlen])
    img_ids = []
    for i, idx in enumerate(batch_indices):
      caption = self.coco.anns[idx]['caption']
      img_id = self.coco.anns[idx]['image_id']
      img_ids.append(img_id)
      path = self.coco.loadImgs(img_id)[0]['file_name']
      # Load and Adjust image
      if preprocessed:
        image = Image.open(os.path.join(self.img_directory, path))
      else:
        image = Image.open(os.path.join(self.img_directory, path))
        image = image.convert('RGB')
        image = image.resize((self.imgw, self.imgh), Image.ANTIALIAS)
      # Pad caption
      padded_tokens = np.zeros((self.seqlen,))
      length = min([len(caption), self.seqlen])
      padded_tokens[:length] = caption[:length]
      # With <start>
      batch_label_input[i] = padded_tokens
      # Without <start>
      batch_label_expected[i, :-1] = padded_tokens[1:]
      image = preprocess_input(np.array(image, dtype=np.float), mode='tf')
      batch_img[i, :, :, :] = image[:, :, :]

    batch_sample_weight = np.zeros((self.batch_size, self.seqlen))
    batch_sample_weight[batch_label_expected > 0] = 1
    # Expected to One-Hot encoding
    batch_label_expected = to_categorical(
        batch_label_expected, self.vocab_size)

    print('Batch in: {}'.format(time.time() - start))
    if not self.return_image_ids:
      return [np.array(batch_img), np.array(batch_label_input)], batch_label_expected, batch_sample_weight
    else:
      return [np.array(batch_img), np.array(batch_label_input)], batch_label_expected, batch_sample_weight, np.array(img_ids)

  def once(self, return_image_ids=False, samples=-1):
    """Create a generator that iterate over the Sequence once.
    Args:
      return_image_ids: If True, return the COCO image ids as a fourth output
      samples: The number of samples to return. If -1, return full epoch.
    """
    temp = self.return_image_ids
    self.return_image_ids = return_image_ids
    current_samples = 0
    for item in (self[i] for i in range(len(self))):
      if samples > 0 and current_samples >= samples:
        raise StopIteration()
      current_samples += self.batch_size
      yield item
    self.return_image_ids = temp

  def on_epoch_end(self):
    """Shuffle Dataset"""
    random.shuffle(self.ann_ids)
