import torch
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
from transform import Colorize
from torchvision.transforms import ToPILImage, Compose, ToTensor, CenterCrop
from transform import Scale
# from resnet import FCN
from upsample import FCN
# from gcn import FCN
from datasets import VOCTestSet
from PIL import Image
import numpy as np
from tqdm import tqdm


label_transform = Compose([Scale((256, 256), Image.BILINEAR), ToTensor()])
batch_size = 1
dst = VOCTestSet("./data", transform=label_transform)

testloader = data.DataLoader(dst, batch_size=batch_size,
                             num_workers=8)


model = torch.nn.DataParallel(FCN(22), device_ids=[0, 1, 2, 3])
# model = FCN(22)
model.cuda()
model.load_state_dict(torch.load("./pth/fcn-deconv-40.pth"))
model.eval()


# 10 13 48 86 101
img = Image.open("./data/VOC2012test/JPEGImages/2008_000101.jpg").convert("RGB")
original_size = img.size
img.save("original.png")
img = img.resize((256, 256), Image.BILINEAR)
img = ToTensor()(img)
img = Variable(img).unsqueeze(0)
outputs = model(img)
# 22 256 256
for i, output in enumerate(outputs):
    output = output[0].data.max(0)[1]
    output = Colorize()(output)
    output = np.transpose(output.numpy(), (1, 2, 0))
    img = Image.fromarray(output, "RGB")
    if i == 0:
        img = img.resize(original_size, Image.NEAREST)
    img.save("test-%d.png" % i)

'''

for index, (imgs, name, size) in tqdm(enumerate(testloader)):
    imgs = Variable(imgs.cuda())
    outputs = model(imgs)

    output = outputs[0][0].data.max(0)[1]
    output = Colorize()(output)
    print(output)
    output = np.transpose(output.numpy(), (1, 2, 0))
    img = Image.fromarray(output, "RGB")
    # img = Image.fromarray(output[0].cpu().numpy(), "P")
    img = img.resize((size[0].numpy(), size[1].numpy()), Image.NEAREST)
    img.save("./results/VOC2012/Segmentation/comp5_test_cls/%s.png" % name)
'''
