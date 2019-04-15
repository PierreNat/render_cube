
"""
script to train a resnet 50 network only with 1 epoch
training was made first with rendered image only with translation motion
the model parameters were saved and it was then tested with the script 7)Test_model with some test data

"""
import torch
import numpy as np

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.empty_cache()
# print(device)


cubes_file = './data/test/cubes.npy'
silhouettes_file = './data/test/sils.npy'
parameters_file = './data/test/params.npy'

target_size = (512, 512)

cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)

#  ------------------------------------------------------------------
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

#  ------------------------------------------------------------------

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
#  ------------------------------------------------------------------

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

#  ------------------------------------------------------------------

# #  try to iterate over the train dataset
# import matplotlib.pyplot as plt
#
# fig = plt.figure()
#
# for image, silhouette, parameters in train_dataloader:
#     print('number of cube images: {}, number of silhouettes: {}, number of parameters: {}'.format(image.size(),
#                                                                                                   silhouette.size(),
#                                                                                                   parameters.size()))
#
#     im = 2
#     image2show = image[im]  # indexing random  one image
#     sil2show = silhouette[im]
#     param = parameters[im]
#
#     #     print('image has size: {}'.format(image2show.size))
#
#     # tensor to numpy:
#     image2shownp = image2show.numpy()
#     sil2shownp = sil2show.numpy()
#     #     .reshape((512, 512,3))  # reshape the torch format to numpy
#     print('silhouette has size: {}'.format(np.shape(sil2shownp)))
#
#     image2shownp = np.transpose(image2shownp, (1, 2, 0))
#     sil2shownp = np.squeeze(
#         np.transpose(sil2shownp, (1, 2, 0)))  # squeeze allow to remove the third dimension [512, 512, 1] --> [512, 512]
#     #     print(image2shownp.shape)
#     fig.add_subplot(1, 2, 1)
#     plt.imshow(image2shownp)
#
#     fig.add_subplot(1, 2, 2)
#     plt.imshow(sil2shownp, cmap='gray')
#     print(param.numpy())
#
#     break  # break here just to show 1 batch of data

#  ------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}




def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride, groups)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=6, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('using pre-trained model')
        pretrained_state = model_zoo.load_url(model_urls['resnet50'])
        model_state = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            k in model_state and v.size() == model_state[k].size()}
        model_state.update(pretrained_state)
        model.load_state_dict(model_state)

        print('download finished')
    return model

#  ------------------------------------------------------------------

import torch.optim as optim

model = resnet50()
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.MSELoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#  ---------------------------------------------------------------
import numpy as np
import tqdm

def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_function, threshold):
    # monitor loss functions as the training progresses
    train_losses = []

    val_losses = []



    for epoch in range(n_epochs):
        ## Training phase


        parameters = []  # ground truth labels
        losses = []  # running loss
        loop = tqdm.tqdm(train_dataloader)
        count = 0

        for image, silhouette, parameter in loop:
            image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
            silhouette = silhouette.to(device)
            predicted_params = model(image)  # run prediction; output <- vector with probabilities of each class


            # zero the parameter gradients
            optimizer.zero_grad()

            # images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes') #create the silhouette with the renderer

            loss = loss_function(predicted_params, parameter) #MSE  value ?

            loss.backward()
            optimizer.step()

            parameters.extend(parameter.cpu().numpy())  # append ground truth label
            losses.append(loss.item())  # batch length is append every time
            count = count+1

            av_loss = np.mean(np.array(losses))
            train_losses.append(av_loss)  # global losses array on the way
            print('run: {} MSE train loss: {:.4f}'.format(count + 1, av_loss))

            # if loss < threshold:  #early stop to avoid over fitting
            #     break

        count2 = 0

        loop = tqdm.tqdm(val_dataloader)
        for image, silhouette, parameter in loop:
            model.eval()
            image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
            silhouette = silhouette.to(device)
            predicted_params = model(image)  # run prediction; output <- vector with probabilities of each class


            # zero the parameter gradients
            optimizer.zero_grad()

            # images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes') #create the silhouette with the renderer

            loss = loss_function(predicted_params, parameter) #MSE  value ?

            parameters.extend(parameter.cpu().numpy())  # append ground truth label
            losses.append(loss.item())  # running loss

            count2 = count2 + 1
            av_loss = np.mean(np.array(losses))
            val_losses.append(av_loss)  # global losses array on the way

            print('run: {} MSE val loss: {:.4f}'.format(count2 + 1, av_loss))

            if count2 == count:  # early stop to avoid over fitting
                break

    return train_losses, val_losses, count, count2


#  ------------------------------------------------------------------


n_epochs = 1
train_losses, val_losses, count, count2 = train(model, train_dataloader, val_dataloader, optimizer, n_epochs, criterion, threshold=0)

#  ------------------------------------------------------------------

torch.save(model.state_dict(), './model_train_1epoch_lr0_01.pth')
print('parameters saved')
#  ------------------------------------------------------------------

import matplotlib.pyplot as plt

def plot(count, train_losses):
    plt.figure()
    plt.plot(np.arange(count), train_losses) #display evenly scale with arange
    # plt.plot(np.arange(n_epochs), val_losses)
    plt.legend(['train_loss'])
    plt.xlabel('runs in dataloader')
    plt.ylabel('loss value')
    plt.title('Train loss')


plot(count, train_losses)

#  ------------------------------------------------------------------


def plot(count, val_losses):
    plt.figure()
    plt.plot(np.arange(count), val_losses) #display evenly scale with arange
    # plt.plot(np.arange(n_epochs), val_losses)
    plt.legend(['train_loss'])
    plt.xlabel('runs in dataloader')
    plt.ylabel('loss value')
    plt.title('Train loss')


plot(count2, val_losses)

#test set