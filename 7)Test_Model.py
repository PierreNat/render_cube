
"""
model that tests test images with the resnet 50 model with pretrained parameters loaded from models directory
plot ground truth images vs estimated parameter rendering

"""
import torch
import torch.nn as nn
import numpy as np
import tqdm
import utils
import matplotlib.pyplot as plt

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

modelName = '042619_TempModel_train_cubes_rgb_test2_6_batchs_epochs_n19_rgb_test'

# cubesAlphaR_1000set
cubeSetName = 'cubes_rgb_test2'
silSetName = 'silsAlphaR_1000set'
paramSetName = 'params_rgb_test_param2'

cubes_file = './data/test/{}.npy'.format(cubeSetName)
silhouettes_file = './data/test/{}.npy'.format(silSetName)
parameters_file = './data/test/{}.npy'.format(paramSetName)

target_size = (512, 512)



cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)


#  ------------------------------------------------------------------
test_length = 1000
batch_size = 6

test_im = cubes[:test_length]
test_sil = sils[:test_length]
test_param = params[:test_length]

plt.imshow(test_im[5])
plt.show()
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

transforms = Compose([ ToTensor(),  normalize])

test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


#  ------------------------------------------------------------------


for image, sil, param in test_dataloader:

    # print(image[2])
    print(image.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
    im =2
    print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])

    image2show = image[im]  # indexing random  one image
    print(image2show.size()) #torch.Size([3, 512, 512])
    plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
    plt.show()
    break  # break here just to show 1 batch of data

#  ------------------------------------------------------------------

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


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
        print('using own pre-trained model')
        model.load_state_dict(torch.load('models/{}.pth'.format(modelName)))
        model.eval()
        print('download finished')
    return model

#  ------------------------------------------------------------------


model = resnet50()
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.MSELoss()

#  ---------------------------------------------------------------


def test(model, test_dataloader, loss_function):
    # monitor loss functions as the training progresses
    test_losses = []

    # test phase
    parameters = []  # ground truth labels
    predicted_params = []
    losses = []  # running loss
    count2 = 0
    f = open("result/Test_result.txt", "w+")
    g = open("result/Test_result_save_param.txt", "w+")
    g.write('alpha alphaGT beta  betaGT gamma gammaGT x xGT yGT zGT \r\n')

    loop = tqdm.tqdm(test_dataloader)
    for image, silhouette, parameter in loop:

        image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
        parameter = parameter.to(device)
        predicted_param = model(image)  # run prediction; output <- vector with probabilities of each class
        # print(predicted_param)

        loss = loss_function(predicted_param, parameter) #MSE  value ?

        parameters.extend(parameter.detach().cpu().numpy())  # append ground truth parameters [array([...], dtype=float32), [...], dtype=float32),...)]
        predicted_params.extend(predicted_param.detach().cpu().numpy()) # append computed parameters
        losses.append(loss.item())  # running loss

        #store value GT(ground truth) and predicted param
        for i in range(0,batch_size):
            for j in range(0,6):
                estim = predicted_params[i][j]
                gt = parameters[i][j]
                g.write('{:.4f} {:.4f} '.format(estim, gt))
            g.write('\r\n')

        av_loss = np.mean(np.array(losses))
        test_losses.append(av_loss)  # global losses array on the way

        print('run: {}/{} MSE test loss: {:.4f}\r\n'.format(count2, len(loop), av_loss))
        f.write('run: {}/{}  MSE test loss: {:.4f}\r\n'.format(count2, len(loop),av_loss))

        count2 = count2 + 1

    f.close()
    g.close()

    return test_losses, count2, parameters, predicted_params


#  ------------------------------------------------------------------
# test the model
test_losses, count, parameters, predicted_params = test(model, test_dataloader, criterion)

#  ------------------------------------------------------------------
# display computed parameter against ground truth

from utils import tom_render_1_image

obj_name = 'rubik_color'

nb_im = 6
loop = tqdm.tqdm(range(0,nb_im))
for i in loop:

    randIm = i
    print('computed parameter_{}: '.format(i+1))
    print(predicted_params[randIm])
    print('ground truth parameter_{}: '.format(i+1))
    print(params[randIm])

    im = tom_render_1_image(obj_name, predicted_params[randIm])  # create the dataset
    # print(im)
    #
    plt.subplot(2, nb_im, i+1)
    plt.imshow(test_im[randIm])

    plt.subplot(2, nb_im, i+1+nb_im)
    plt.imshow(im)

plt.show()
print('finish')