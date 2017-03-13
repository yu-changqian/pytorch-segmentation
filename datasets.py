import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data

def default_loader(path):
    return Image.open(path)

class VOCDataSet(data.Dataset):
    def __init__(self, root, split="trainval", is_transform=False, loader=default_loader):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.loader = loader

        data_dir = osp.join(root, "VOC2012")
        for split in ["train", "trainval", "val"]:
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
        img = Image.open(img_file)
        img = img.resize((224, 224), Image.NEAREST)
        img = np.array(img, dtype=np.uint8)

        label_file = datafiles["label"]
        label = Image.open(label_file)
        # label image has categorical value, not continuous, so we have to
        # use NEAREST not BILINEAR
        label = label.resize((224, 224), Image.NEAREST)
        label = np.array(img, dtype=np.int32)
        label[label == 255] = -1

        if self.is_transform:
            img, label = self.transform(img, label)

        return img, label

    def transform(self, img, label):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        return img, label

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
