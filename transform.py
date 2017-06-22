import numpy as np
import torch
from PIL import Image
import collections


class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


class ToParallel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        yield img
        for t in self.transforms:
            yield t(img)


class ToLabel(object):
    def __call__(self, inputs):
        tensors = []
        for i in inputs:
            tensors.append(torch.from_numpy(np.array(i)).long())
        return tensors


class ReLabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        # assert isinstance(input, torch.LongTensor), 'tensor needs to be LongTensor'
        for i in inputs:
            i[i == self.olabel] = self.nlabel
        return inputs


class ToSP(object):
    def __init__(self, size):
        self.scale2 = Scale(size/2, Image.NEAREST)
        self.scale4 = Scale(size/4, Image.NEAREST)
        self.scale8 = Scale(size/8, Image.NEAREST)
        self.scale16 = Scale(size/16, Image.NEAREST)
        self.scale32 = Scale(size/32, Image.NEAREST)

    def __call__(self, input):
        input2 = self.scale2(input)
        input4 = self.scale4(input)
        input8 = self.scale8(input)
        input16 = self.scale16(input)
        input32 = self.scale32(input)
        inputs = [input, input2, input4, input8, input16, input32]
        # inputs = [input]

        return inputs


class HorizontalFlip(object):
    """Horizontally flips the given PIL.Image with a probability of 0.5."""

    def __call__(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)


class VerticalFlip(object):
    def __call__(self, img):
        return img.transpose(Image.FLIP_TOP_BOTTOM)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7-j))*((i & (1 << (3*j))) >> (3*j))
            g = g + (1 << (7-j))*((i & (1 << (3*j+1))) >> (3*j+1))
            b = b + (1 << (7-j))*((i & (1 << (3*j+2))) >> (3*j+2))

        cmap[i, :] = np.array([r, g, b])

    return cmap


class Colorize(object):
    def __init__(self, n=22):
        self.cmap = labelcolormap(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
