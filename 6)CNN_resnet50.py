
"""
Cube test Resnet 50

"""


import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)


cubes_file = './data/test/cubes.npy'
silhouettes_file = './data/test/sils.npy'
parameters_file = './data/test/params.npy'

target_size = (512, 512)

cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)

# split set of data

ratio = 0.9  # 90%training 10%validation
split = int(len(cubes)*0.9)
test_length = 1000

train_im = cubes[:split]  # 90% training
train_sil = sils[:split]
train_param = params[:split]

val_im = cubes[split:]  # remaining ratio for validation
val_sil = sils[split:]
val_param = params[split:]

test_im = cubes[:test_length]
test_sil = sils[:test_length]
test_param = params[:test_length]

#  --------------------------------------------

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose

class CubeDataset(Dataset):
    # write your code
    def __init__(self, images, silhouettes, parameters, transform=None):
        self.images = images.astype(np.uint8)  # our image
        self.silhouettes = silhouettes.astype(np.uint8)  # our related parameter
        self.parameters = parameters.astype(np.float16)
        self.transform = transform

    def __getitem__(self, index):
        # Anything could go here, e.g. image loading from file or a different structure
        # must return image and center
        sel_images = self.images[index]
        sel_sils = self.silhouettes[index]
        sel_params = self.parameters[index]

        if self.transform is not None:
            sel_images = self.transform(sel_images)
            sel_sils = self.transform(sel_sils)

        return sel_images, sel_sils, torch.FloatTensor(sel_params)  # return all parameter in tensor form

    def __len__(self):
        return len(self.images)  # return the length of the dataset



batch_size = 32

transforms = Compose([ToTensor()])
train_dataset = CubeDataset(train_im, train_sil, train_param, transforms)
val_dataset = CubeDataset(val_im, val_sil, val_param, transforms)
test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)

#  Note:
#  DataLoader(Dataset,int,bool,int)
#  dataset (Dataset) – dataset from which to load the data.
#  batch_size (int, optional) – how many samples per batch to load (default: 1)
#  shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
#  num_workers = n - how many threads in background for efficient loading

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

# --------------------------------------------

#  try to iterate over the train dataset
import matplotlib.pyplot as plt

for image, silhouette, parameters in train_dataloader:
    print('number of cube images: {}, number of silhouettes: {}, number of parameters: {}'.format(image.size(), silhouette.size(), parameters.size()))

    image2show = image[2]  # indexing random  one image

    # tensor to numpy:
    image2shownp = image2show.numpy().reshape((512, 512, 3))  # reshape the torch format to numpy
    print('image has size: {}'.format(image2shownp.size))

    plt.imshow(image2shownp)

    break  # break here just to show 1 batch of data


# --------------------------------------------

# import torch.nn as nn
# import torch.nn.functional as F
#
# import math
# import torch.utils.model_zoo as model_zoo
#
# __all__ = ['resnet50']
#
# #model_urls = {
# #
# #    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
# #
# #}
#
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=2):
#         self.inplanes = 64
#         super(ResNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x
#
#
#
#
# def resnet50(pretrained=True, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
# #        train = torch.load('resnet50-19c8e357.pth')
#         model.load_state_dict(torch.load('resnet50.pth'), strict=False)
#         #'resnet-50-kinetics-ucf101_split1.pth'
#         #'resnet-50-kinetics.pth'
#         #'resnet50.pth'
#     return model
#
#
# import torch.optim as optim
#
# model = resnet50()
# model = model.to(device) # transfer the neural net onto the GPU
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # learning step and momentum accelerate gradients vectors in the right directions
#
# #--------------------------------------------
#
# import numpy as np
#
# def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_function, lr=0.001):
#     # monitor loss functions as the training progresses
#
#     optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
#     train_losses = []
#     train_accuracies = []
#     val_losses = []
#     val_accuracies = []
#     init = 5 #wait 5 iterations before dividing again lr
#     decr = 0
#     lr = 0.001 #learning rate start
#
#     for epoch in range(n_epochs):
#         ## Training phase
#
#         correct_train_predictions = 0 # We will measure accuracy
#         # Iterate mini batches over training dataset
#         losses = []
#         for images, labels in train_dataloader:
#             images = images.to(device) #we have to send the inputs and targets at every step to the GPU too
#             labels = labels.to(device)
#             output = model(images) # run prediction; output <- vector with probabilities of each class
#             # set gradients to zero
#             optimizer.zero_grad()
#             loss = loss_function(output, labels)
# #             print(loss.item())
#             loss.backward()  #computes dloss/dx for every parameter x
#             optimizer.step() #performs a parameter update based on the current gradient
#
#             # Metrics
#             losses.append(loss.item()) # gets the a scalar value held in the loss.
#             predicted_labels = output.argmax(dim=1)
# #            print(predicted_labels)
#             n_correct = (predicted_labels == labels).sum().item() #compare the computation with ground truth
#             correct_train_predictions += n_correct
#         train_losses.append(np.mean(np.array(losses))) #build a losses array on the way
#         train_accuracies.append(100.0*correct_train_predictions/len(train_dataloader.dataset)) #ratio of correct answer on ground truth
#
# #--------------------------------------------
#         # Evaluation phase
#
#         correct_val_predictions = 0 # We will measure accuracy
#         # Iterate mini batches over validation set
#         # We don't need gradients
#         losses = []
#         with torch.no_grad():
#             for images, labels in val_dataloader:
#                 images = images.to(device)
#                 labels = labels.to(device)
#                 output = model(images)
#                 loss = loss_function(output, labels)
#
#                 losses.append(loss.item())
#                 predicted_labels = output.argmax(dim=1)
#                 n_correct = (predicted_labels == labels).sum().item()
#                 correct_val_predictions += n_correct
#         val_losses.append(np.mean(np.array(losses)))
#         val_accuracies.append(100.0*correct_val_predictions/len(val_dataloader.dataset))
#
#         print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(epoch+1, n_epochs,
#                                                                                                       train_losses[-1],
#                                                                                                       train_accuracies[-1],
#                                                                                                       val_losses[-1],
#                                                                                                       val_accuracies[-1]))
#         if epoch > 5:
#             if np.abs(val_losses[epoch]-val_losses[epoch-1]) and np.abs(val_losses[epoch]-val_losses[epoch-2]) < 0.8:
#                 if decr == 0:
#                     decr = init
#                     if lr>0.00000001:
#                         lr=lr/10
#                         print('Learning rate is now: {:.8f}'.format(lr))
#                     else:
#                         lr=0.00000001
#
#                 else:
#                     decr = decr-1
# test_dir = './kaggle/test'
#     return train_losses, val_losses, train_accuracies, val_accuracies
#
#
# #--------------------------------------------
#
# n_epochs = 40
# train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_dataloader, val_dataloader, optimizer, n_epochs, criterion)
#
#