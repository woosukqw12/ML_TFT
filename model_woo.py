import torch.nn as nn
import torch
import torch.nn.functional as F    
from torch import optim 
import torchvision.models
# from torchvision.models import resnet18
import torchvision
    
    
# from __future__ import print_function
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#custom
from dataloader import CustomDataset

import torch
import torch.nn as nn
# reference: https://pseudo-lab.github.io/pytorch-guide/docs/ch03-1.html
# reference: https://yhkim4504.tistory.com/3
# * qwe
# ! qwe
# ? qwe
# TODO qwe

# flow: conv1x1, 3x3 / BasicBlock / Bottleneck / ResNet(init, make_layer, forward) / resnets / notes

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
# !
# in_planes: in_channels(입력 필터개수)
# out_planes: out_channels(출력 필터개수)
# groups: input과 output의 connection을 제어, default=1
# dilation: 커널 원소간의 거리. 늘릴수록 같은 파라미터수로 더 넓은 범위를 파악할 수 있음
#           default=1.
# bias=False: BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정.
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
    


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    # !
    # inplanes: input channel size
    # planes: output channel size
    # groups, base_width: ResNeXt, Wide ResNet의 경우 사용.
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d #* default norm_layer --> BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # The structure of Basic Block
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes) # ! bn: Batch Normalization // >>변형된 분포가 나오지 않도록 하기 위해.
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x # * x

        out = self.conv1(x) # * from here
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) # * to here, F(x)

        if self.downsample is not None:
            # x와 F(x)의 tensor size가 다를 때,
            identity = self.downsample(x)
        # * identity mapping시, mapping후 ReLU를 적용한다.
        # * ReLU를 통과하면 양의 값만 남아 Residual의 의미가 제대로 유지되지 않기 때문.
        out += identity #* F(x)+x
        out = self.relu(out) #* activation(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4 # ! Block내의 마지막 conv1x1에서 차원 증사시키는 확장 계수

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # * ResNeXt, WideResNet의 경우 사용
        width = int(planes * (base_width / 64.)) * groups
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # * The structure of Bottleneck Block
        self.conv1 = conv1x1(inplanes, width) # 논문 실험들에선 여기서 차원 축소 -> 3x3에서 연산부담 줄임
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation) # ! conv2에서 downsample, stride=2.
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion) # 다시 차원 증가
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x) # conv
        out = self.bn1(out) # batch norm
        out = self.relu(out)# ReLU
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out) # 여기까지 F(x)
        # skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity 
        out = self.relu(out) # * H(x) = ReLU(F(x) + x)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # * default values
        self.inplanes = 64 # input feature map
        self.dilation = 1
        # * stride를 dilation으로 대체할지 선택.
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        # * 7x7 conv, 3x3 max pooling
        # * 3: input이 RGB이미지여서 conv layer의 input channel수는 3
        self.conv1 = nn.Conv2d(146, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ### * residual blocks
        # * filter의 개수는 각 block들을 거치면서 증가. 
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # * layer --end--
        
        # * 모든 block을 거친 후엔 AvgPolling으로 (n, 512, 1, 1)의 텐서로 변환
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # * fully connected layer에 연결
        # 이전 레이어의 출력을 평탄화하여 다음 stage의 입력이 될 수 있는 단일 벡터로 변환한다.
        # falt -> activation -> Softmax
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # !
        # * 논문의 conv2_x, conv3_x, ... 5_x를 구현하며,
        # * 각 층에 해당하는 block을 개수에 맞게 생성 및 연결시켜주는 역할을 한다.
        # * convolution layer 생성 함수
        # block: block 종류
        # planes: feature map size (input shape)
        # blocks: layers[0], [1], ... 과 같이, 해당 블록이 몇개 생성돼야 하는지, 블록의 개수 (layer 반복해서 쌓는 개수)
        # stride, dilate is fixed.
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
            
        # stride가 1이 아님-->크기가 줄어듦, 차원의 크기가 맞지 않음 --> downsampling
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 블록 내 시작 layer, downsampling해야함.
        # 왜 처음 한번은 따로 쌓음? -> 첫 block을 쌓고 self.inplanes을 plane에 맞춰주기 위함.
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion # * update inplanes
        # 동일 블록 반복.
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    # !
    # arch: ResNet model 이름
    # block: 어떤 block형태를 사용할지 (Basic or Bottleneck)
    # layers: 해당 block이 몇번 사용되는지를 list형태로 넘겨주는 부분
    # pretrained: pretrain된 model weights를 불러오기
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model

# !
# 공통 Arguments
## pretrained (bool): If True, returns a model pre-trained on ImageNet
## progress (bool): If True, displays a progress bar of the download to stderr
def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=146, out_channels=256, 
                            kernel_size=3, stride=1, padding='same') 
        # in_channels, out_channels, kernel_size,stride=1, padding=0, 
        self.conv2 = nn.Conv2d(256, 256, 3, 1, padding='same')
        self.conv3 = nn.Conv2d(256, 512, 3, 1, padding='same')
        self.conv4 = nn.Conv2d(512, 512, 3, 1, padding='same')
        # self.conv5 = nn.Conv2d(512, 512, 3, 1, padding='same')
        # self.conv6 = nn.Conv2d(512, 1024, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        
        # self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        # print("shape:", x.shape) # shape: torch.Size([64, 146, 4, 4])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # x = self.conv5(x)
        # x = F.relu(x)
        # x = self.conv6(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        
        x = self.dropout(x)
        
        x = torch.flatten(x,1)
        
        # x = self.fc1(x)
        # x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # print("shape:", x.shape) # shape: torch.Size([64, 8])
        
        # output = F.log_softmax(x, dim=1)
        
        return x
    
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=146, out_channels=256, 
                            kernel_size=3, stride=1, padding='same') 
        # in_channels, out_channels, kernel_size,stride=1, padding=0, 
        self.conv2 = nn.Conv2d(256, 256, 3, 1, padding='same')
        self.conv3 = nn.Conv2d(256, 512, 3, 1, padding='same')
        self.conv4 = nn.Conv2d(512, 512, 3, 1, padding='same')
        self.conv5 = nn.Conv2d(512, 512, 3, 1, padding='same')
        self.conv6 = nn.Conv2d(512, 1024, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        # print("shape1:", x.shape) # shape: torch.Size([64, 146, 4, 4])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # print("shape2:", x.shape)
        x = torch.flatten(x,1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # print("shape3:", x.shape) # shape: torch.Size([64, 8])
        # print("n", x)
        # x = F.log_softmax(x, dim=1)
        # print("s",x)
        return x
    
class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=146, out_channels=256, 
                            kernel_size=3, stride=1, padding='same') 
        # in_channels, out_channels, kernel_size,stride=1, padding=0, 
        self.conv2 = nn.Conv2d(256, 256, 3, 1, padding='same')
        self.conv3 = nn.Conv2d(256, 512, 3, 1, padding='same')
        self.conv4 = nn.Conv2d(512, 512, 3, 1, padding='same')
        self.conv5 = nn.Conv2d(512, 512, 3, 1, padding='same')
        self.conv6 = nn.Conv2d(512, 1024, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 8)
        
    def forward(self, x):
        # print("shape1:", x.shape) # shape: torch.Size([64, 146, 4, 4])
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # print("shape2:", x.shape)
        x = torch.flatten(x,1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        # print("shape3:", x.shape) # shape: torch.Size([64, 8])
        # print("n", x)
        # x = F.log_softmax(x, dim=1)
        # print("s",x)
        return x


def train_(model, cost, optim, train_dl, device):
    N = len(train_dl.dataset)
    n_batch = int(N / train_dl.batch_size)
    
    model.train()
    losses = []
    for batch, (x,y) in enumerate(train_dl):
        # print(f"x: {x} type: {type(x)}\ny: {y} \n")
        x = x.to(device).float()
        y = y.to(device).float()
        # print("test() x.shape", x.shape)
        pred = model(x)
        # print("a")
        # print("pred:",pred.shape)
        # print(y.shape)
        cost_ = cost(pred, y)
        losses.append(cost_.item())
        # Backpropagation.
        optim.zero_grad()
        cost_.backward()        
        optim.step()
        if batch%1000==0:
            print(f"\rTrain: {batch+1}/{n_batch}\tloss:{cost_}")
            # sys.stdout.write(f"\rTrain: {batch+1}/{n_batch} loss:{cost_}")
            # sys.stdout.flush()
        # break
    avg_loss = np.mean(losses)
    return avg_loss

def test_(model, loss_fn, test_dl, device):
    model = model.to(device)
    N = len(test_dl.dataset)
    n_batch = int(N / test_dl.batch_size)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch, (x,y) in enumerate(test_dl):
            x,y = x.to(device).float(), y.to(device).float()
            
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            if batch%500==0:
                print(f"\rTest: {batch+1}/{n_batch}\tloss_test:{loss}")
                # sys.stdout.write(f"\rTest: {batch+1}/{n_batch} loss_test:{loss}")
                # sys.stdout.flush()
            # break
    avg_loss = np.mean(losses)
    return avg_loss

def predict_(model, loss_fn, test_dl, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch, (x,y) in enumerate(test_dl):
            x,y = x.to(device).float(), y.to(device).float()
            
            pred = model(x)
            L = loss_fn(pred, y)
            y = y.cpu().numpy().squeeze()
            pred = pred.cpu().numpy().squeeze()
            y += 1
            pred += 1 
            # print(f"y: {y[:10]}")
            # print(f"h: {pred[:10]}")
            # print(sum(abs(y[:10]-pred[:10])))
            # print(sum(abs(y-pred)),'\n\n')
            
        print(f"L1 loss: {sum(abs(y-pred))/len(y)}")
        print(f"L2 loss: {L}")
            
        
if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset()
    train_ds, test_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    model = Net2().to(device)
    # model = resnet18(num_classes=1)
    # model = resnet18(num_classes=1).to(device)
    # model = resnet50(num_classes=1).to(device)
    
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    
    train_dl = DataLoader(train_ds, 
                            batch_size=64, 
                            shuffle=True,
                            drop_last=True
                        )
    test_dl  = DataLoader(test_ds, 
                            batch_size=64, 
                            shuffle=True,
                            drop_last=True
                        )
    try:
        model.load_state_dict(torch.load("model_param2_no_dropout.pt", map_location=device))
        predict_(model, criterion, test_dl, device)
        print(1)
        # quit()
    except:
        print(2)
        quit()
        # print(torchvision.models.list_models())
        # transform = transforms.Compose([
        #         # transforms.ToTensor(),
        #         transforms.Resize(224),
        #         # transforms.RandomHorizontalFlip(),
        #     ])
        
        # print(f"len: {len(train_ds)}\nlen: {len(test_ds)}") #len: 378796, len: 94698
        
        # model = nn.Sequential(nn.Linear(146, 128),
        #                       nn.ReLU(),
        #                       nn.Linear(128, 64),
        #                       nn.ReLU(),
        #                       nn.Linear(64, 8)
        #                        )
        # model = model.to(device)
        train_loss = []
        test_loss = []
        for epoch in range(epochs):
            print(f"[ epoch: {epoch+1}/{epochs} ]")
            train_loss.append(train_(model, criterion, optimizer, train_dl, device))
            test_loss.append(test_(model, criterion, test_dl, device))
        torch.save(model.state_dict(), "model_param4_resnet50.pt")
        predict_(model, criterion, test_dl, device)
        
        