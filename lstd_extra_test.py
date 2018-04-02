import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    #print('here')
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.cla()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

#imshow(torchvision.utils.make_grid(images))
#print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def add_lstd_extras(i):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    conv_add1 = nn.Conv2d(in_channels, 256,
                           kernel_size=3, stride=1, padding=1)

    in_channels = 256
    batchnorm_add1 = nn.BatchNorm2d(in_channels)
    conv_add2 = nn.Conv2d(in_channels, 256,
                           kernel_size=3, stride=2, padding=1)

    batchnorm_add2 = nn.BatchNorm2d(in_channels)
    #bbox_score_voc = nn.Linear(256, 21)

    layers += [conv_add1, batchnorm_add1, nn.ReLU(inplace=True), conv_add2, batchnorm_add2, nn.ReLU(inplace=True)]
    return layers

extras_lstd_ = add_lstd_extras(3)

class Net(nn.Module):
    def __init__(self, extras_lstd):
        super(Net, self).__init__()
        '''
        self.extras_lstd = nn.ModuleList(extras_lstd)
        self.classifier = nn.ModuleList([nn.Linear(65536, 10)])
        '''
        '''
        Linear(in_features=25088, out_features=4096, bias=True)
        ReLU(inplace)
        Dropout(p=0.5)
        Linear(in_features=4096, out_features=4096, bias=True)
        ReLU(inplace)
        Dropout(p=0.5)

        '''
        self.classifier = nn.ModuleList([nn.Linear(3 * 32 * 32, 4096),
            nn.ReLU(True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)])

        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        #print(x.size())
        '''
        for k, v in enumerate(self.extras_lstd):
            x = v(x)
            #print(x.size())
       
        x = x.view(x.size(0), -1)
        ''' 
        #print(x.size())
        x = x.view(x.size(0), -1)
        for k, v in enumerate(self.classifier):
           
            x = v(x)

        #cls_output = self.classifier(x)
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        '''
        return x


net = Net(extras_lstd_)
net.cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(6):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    predicted = predicted.cpu()
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu()
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
