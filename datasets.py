import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from transform import HorizontalFlip, VerticalFlip

def default_loader(path):
    return Image.open(path)

class VOCDataSet(data.Dataset):
    def __init__(self, root, split="trainval", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()

        data_dir = osp.join(root, "VOC2012")
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = osp.join(data_dir, "ImageSets/Segmentation/%s.txt" % split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "JPEGImages/%s.jpg" % name)
                label_file = osp.join(data_dir, "SegmentationClass/%s.png" % name)
                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        # img = img.resize((256, 256), Image.NEAREST)
        # img = np.array(img, dtype=np.uint8)

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")
        label_size = label.size
        # label image has categorical value, not continuous, so we have to
        # use NEAREST not BILINEAR
        # label = label.resize((256, 256), Image.NEAREST)
        # label = np.array(label, dtype=np.uint8)
        # label[label == 255] = 21

        if self.img_transform is not None:
            img_o = self.img_transform(img)
            # img_h = self.img_transform(self.h_flip(img))
            # img_v = self.img_transform(self.v_flip(img))
            imgs = [img_o]
        else:
            imgs = img

        if self.label_transform is not None:
            label_o = self.label_transform(label)
            # label_h = self.label_transform(self.h_flip(label))
            # label_v = self.label_transform(self.v_flip(label))
            labels = [label_o]
        else:
            labels = label

        return imgs, labels


class VOCTestSet(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = collections.defaultdict(list)

        self.data_dir = osp.join(root, "VOC2012test")
        self.img_names = os.listdir(osp.join(self.data_dir, "JPEGImages"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]
        img = Image.open(osp.join(self.data_dir, "JPEGImages", name)).convert('RGB')
        size = img.size
        name = name.split(".")[0]

        if self.transform is not None:
            img = self.transform(img)

        return img, name, size

if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
