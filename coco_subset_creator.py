import argparse
import json
from pycocotools.coco import COCO


def loadAnns(coco, img_ids):
  anns = []
  for imgid in img_ids:
    if imgid in coco.anns:
      anns.append(coco.anns[imgid])
  return anns


def create_subset(args):
  coco = COCO(args.ann_file)
  imgids = list(coco.imgs.keys())

  sub_imgids = imgids[:args.size]
  sub_anns = coco.loadAnns(coco.getAnnIds(imgIds=sub_imgids))
  sub_imgs = coco.loadImgs(sub_imgids)
  coco_json = json.load(open(args.ann_file))
  coco_json['annotations'] = sub_anns
  coco_json['images'] = sub_imgs
  with open(args.outfile, 'w+') as f:
    json.dump(coco_json, f)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description='COCO Subset Creator')
  p.add_argument('ann_file')
  p.add_argument('outfile')
  p.add_argument('-size', default=200, type=int)
  args = p.parse_args()
  create_subset(args)
