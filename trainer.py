import torch
from torch.autograd import Variable
from torch.utils import data
from resnet import ResNet50
from datasets import VOCDataSet
from loss import CrossEntropy2d, CrossEntropyLoss2d
from visualize import LinePlotter
import tqdm


trainloader = data.DataLoader(VOCDataSet("./data", is_transform=True), batch_size=4,
                                num_workers=8)

model = ResNet50()
# model = torch.nn.DataParallel(ResNet50(), device_ids=[0, 1])
if torch.cuda.is_available():
    model.cuda()

epoches = 100
lr = 0.1
weight_decay = 0.0001
momentum = 0.9
weight = torch.ones(22)
weight[21] = 0

criterion = CrossEntropyLoss2d(weight)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)
ploter = LinePlotter()

for epoch in range(epoches):
    running_loss = 0.0
    for i, (images, labels) in tqdm.tqdm(enumerate(trainloader)):
        print(i)
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(images.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss /= len(images)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        if (i+1) % 100 == 0:
            print("Epoch [%d/%d] Iter [%d/%d] Loss: %.4f" % (epoch+1, 80,
                  i+1, 2000, loss.data[0]))
            ploter.plot("loss", "train", (i/100, running_loss/100))
            running_loss = 0

    if (epoch+1) % 30 == 0:
        lr /= 10
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)


torch.save(model, "resnet50.pkl")
