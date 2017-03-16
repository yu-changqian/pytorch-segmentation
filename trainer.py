import torch
from torch.autograd import Variable
from torch.utils import data
from resnet import ResNet50
from datasets import VOCDataSet
from loss import CrossEntropy2d


trainloader = data.DataLoader(VOCDataSet("./data", is_transform=True), batch_size=4,
                                num_workers=8)

model = ResNet50()
if torch.cuda.is_available():
    model.cuda()

lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(80):
    for i, (images, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(images.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = CrossEntropy2d(outputs, labels, size_average=False)
        loss /= len(images)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print("Epoch [%d/%d] Iter [%d/%d] Loss: %.4f" % (epoch+1, 80,
                    i+1, 30, loss.data[0]))

    if (epoch+1) % 20 == 0:
        lr /= 3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

torch.save(model, "resnet50.pkl")
