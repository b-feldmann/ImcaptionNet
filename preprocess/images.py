import os
from PIL import Image


def preprocess(in_img_dir, out_img_dir, imgw, imgh):
  for file in os.listdir(in_img_dir):
    if file.endswith('.jpg'):
      image = Image.open(os.path.join(in_img_dir, file)).convert('RGB')
      image = image.resize((imgw, imgh), Image.ANTIALIAS)
      image.save(os.path.join(out_img_dir, file))


if __name__ == '__main__':
  import argparse
  p = argparse.ArgumentParser()
  p.add_argument('indir')
  p.add_argument('outdir')
  p.add_argument('-imgw', default=256, type=int)
  p.add_argument('-imgh', default=256, type=int)
  args = p.parse_args()
  preprocess(args.indir, args.outdir, args.imgw, args.imgh)
