
"""
script to train a resnet 50 network only with n epoch
Version 4
plot x y z alpha beta gamma error
plot render after each epoch
"""
import torch
import numpy as np

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

cubeSetName = 'cubes_rgb_test2'
silSetName = 'silsAlphaR_1000set'
paramSetName = 'params_rgb_test_param2.npy'

# cubeSetName = 'cubesRt'
# silSetName = 'silsRt'
# paramSetName = 'paramsRt'

date4File = '042619' #mmddyy

fileExtension = 'rgb_test'

batch_size = 6

n_epochs = 20

target_size = (512, 512)

cubes_file = './data/test/{}.npy'.format(cubeSetName)
silhouettes_file = './data/test/{}.npy'.format(silSetName)
parameters_file = './data/test/{}.npy'.format(paramSetName)



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
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda

class CubeDataset(Dataset):
    # write your code
    def __init__(self, images, silhouettes, parameters, transform=None):
        self.images = images.astype(np.uint8)  # our image
        self.silhouettes = silhouettes.astype(np.uint8)  # our related parameter
        self.parameters = parameters.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index):
        # Anything could go here, e.g. image loading from file or a different structure
        # must return image and center
        sel_images = self.images[index].astype(np.float32) / 255
        sel_sils = self.silhouettes[index]
        sel_params = self.parameters[index]

        if self.transform is not None:
            sel_images = self.transform(sel_images)
            sel_sils = self.transform(sel_sils)

        return sel_images, sel_images, torch.FloatTensor(sel_params)  # return all parameter in tensor form

    def __len__(self):
        return len(self.images)  # return the length of the dataset
#  ------------------------------------------------------------------


normalize = Normalize(mean=[0.5], std=[0.5])
gray_to_rgb = Lambda(lambda x: x.repeat(3, 1, 1) )
transforms = Compose([ ToTensor(),  normalize])
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


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('using pre-trained model')

        model.load_state_dict(torch.load('./model_train_nepoch.pth'))
        model.eval()
        # pretrained_state = model_zoo.load_url(model_urls['resnet50'])
        # model_state = model.state_dict()
        # pretrained_state = {k: v for k, v in pretrained_state.items() if
        #                     k in model_state and v.size() == model_state[k].size()}
        # model_state.update(pretrained_state)
        # model.load_state_dict(model_state)

        print('download finished')
    return model

#  ------------------------------------------------------------------

import torch.optim as optim

model = resnet50()
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.MSELoss()
# learning_rate = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#  ---------------------------------------------------------------
import numpy as np
import tqdm

def train(model, train_dataloader, val_dataloader, n_epochs, loss_function):
    # monitor loss functions as the training progresses
    learning_rate = 0.001
    train_losses = []
    train_epoch_losses = []
    val_losses = []
    val_epoch_losses = []
    firstrun = True
    all_translation_loss_per_epoch = []
    all_rotation_loss_per_epoch = []

    f = open("result/{}_{}_{}_batchs_{}_epochs_{}.txt".format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension), "w+")
    g = open("result/{}_{}_{}_batchs_{}_epochs_{}_Rtvalues.txt".format(date4File, cubeSetName, str(batch_size), str(n_epochs),
                                                              fileExtension), "w+")
    g.write('alpha alphaGT beta betaGT gamma gammaGT x xGT y yGT z zGT \r\n')
    for epoch in range(n_epochs):

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.004)
        f.write('Train, run epoch: {}/{} with Lr {} \r\n'.format(epoch, n_epochs, str(learning_rate)))
        g.write('Train, run epoch: {}/{} with Lr {} \r\n'.format(epoch, n_epochs, str(learning_rate)))
        print('run epoch: {} with Lr {}'.format(epoch, learning_rate))

        ## Training phase
        model.train()
        parameters = []  # ground truth labels
        predict_params = []  # predicted labels


        losses = []  # running loss
        loop = tqdm.tqdm(train_dataloader)
        count = 0
        firstrun = True

        for image, silhouette, parameter in loop:
            image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
            parameter = parameter.to(device)
            predicted_params = model(image)  # run prediction; output <- vector with probabilities of each class


            # zero the parameter gradients
            optimizer.zero_grad()

            # # initialisation rotation of the cube
            # if firstrun:
            #     predicted_params[:,0:2] = 1
            #     firstrun = False


            # images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes') #create the silhouette with the renderer

            loss = loss_function(predicted_params, parameter) #MSE  value ?
            alpha_loss = loss_function(predicted_params[:, 0], parameter[:, 0])
            beta_loss = loss_function(predicted_params[:, 1], parameter[:, 1])
            gamma_loss = loss_function(predicted_params[:, 2], parameter[:, 2])
            x_loss = loss_function(predicted_params[:, 3], parameter[:, 3])
            y_loss = loss_function(predicted_params[:, 4], parameter[:, 4])
            z_loss = loss_function(predicted_params[:, 5], parameter[:, 5])

            loss.backward()
            optimizer.step()

            parameters.extend(parameter.cpu().numpy())  # append ground truth label
            predict_params.extend(predicted_params.detach().cpu().numpy())  # append computed parameters
            losses.append(loss.item())  # batch length is append every time

            # store value GT(ground truth) and predicted param in a text file
            for i in range(0, batch_size):
                for j in range(0, 6):
                    estim = predicted_params[i][j]
                    gt = parameters[i][j]
                    g.write('{:.4f} {:.4f} '.format(estim, gt))
                g.write('\r\n')

            av_loss = np.mean(np.array(losses))
            train_losses.append(av_loss)  # global losses array on the way
            print('run: {}/{} MSE train loss: {:.4f}, angle loss: {:.4f} {:.4f} {:.4f}  translation loss: {:.4f} {:.4f} {:.4f}  '.format(count, len(loop), av_loss ,alpha_loss, beta_loss, gamma_loss, x_loss,y_loss, z_loss))
            f.write('run: {}/{} MSE train loss: {:.4f}, angle loss: {:.4f} {:.4f} {:.4f}  translation loss: {:.4f} {:.4f} {:.4f} \r\n'.format(count, len(loop), av_loss ,alpha_loss, beta_loss, gamma_loss, x_loss,y_loss, z_loss))

            count = count + 1

            # if loss < threshold:  #early stop to avoid over fitting
            #     break

        train_epoch_losses.append(np.mean(np.array(losses))) # global losses array on the way

        count2 = 0
        model.eval()
        f.write('Val, run epoch: {}/{} \r\n'.format(epoch, n_epochs))
        loop = tqdm.tqdm(val_dataloader)
        for image, silhouette, parameter in loop:

            image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
            parameter = parameter.to(device)
            predicted_params = model(image)  # run prediction; output <- vector with probabilities of each class

            # zero the parameter gradients
            optimizer.zero_grad()

            # images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes') #create the silhouette with the renderer

            loss = loss_function(predicted_params, parameter) #MSE  value ?

            parameters.extend(parameter.cpu().numpy())  # append ground truth label
            losses.append(loss.item())  # running loss


            av_loss = np.mean(np.array(losses))
            val_losses.append(av_loss)  # global losses array on the way

            print('run: {}/{} MSE val loss: {:.4f}\r\n'.format(count2, len(loop), av_loss))
            # f.write('run: {}/{}  MSE val loss: {:.4f}\r\n'.format(count2, len(loop), av_loss))

            count2 = count2 + 1
        val_epoch_losses.append(np.mean(np.array(losses))) #global losses array on the way
        torch.save(model.state_dict(),
                   'models/{}_TempModel_train_{}_{}_batchs_epochs_n{}_{}.pth'.format(date4File, cubeSetName, str(batch_size),
                                                                                str(epoch), fileExtension))
        print('parameters saved for epoch {}'.format(epoch))
        # learning_rate = learning_rate/10

    f.close()
    g.close()

    return train_epoch_losses, val_epoch_losses, count, count2


#  ------------------------------------------------------------------



train_losses, val_losses, count, count2 = train(model, train_dataloader, val_dataloader, n_epochs, criterion)

#  ------------------------------------------------------------------

torch.save(model.state_dict(), 'models/{}_FinalModel_train_{}_{}_batchs_{}_epochs_{}.pth'.format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension))
print('parameters saved')

#  ------------------------------------------------------------------
