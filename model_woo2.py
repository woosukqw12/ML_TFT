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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#custom
from dataloader import CustomDataset


class _Loss(nn.Module): 
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = nn._Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
            
            
class CustomLoss(_Loss): # 등수(label)가 3~6등일때는 비중을 줄인다. 왜? 극의 등수보단, 중간에 위치한 등수는 변동되기 쉬움을 감안하여 비중을 줄여보는 시도를 했다.
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(CustomLoss, self).__init__(size_average, reduce, reduction)
    # def custom_loss(input, target, reduction):
    #     return 
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        res = abs(input-target)
        for i in res:
            if res[0] in [3,4,5,6]:
                res[0] *= 0.8
        res = res*res
        # print(res.shape)
        return res.mean() #sum(res)/len(target)
        return custom_loss(input, target, reduction=self.reduction)
    
class Net_(nn.Module): # NN, ReLU, 146 -> 64 -> 16 -> 4 -> 1
    def __init__(self) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(146, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(4, 1)
        
    
    def forward(self, x):
        # print("q,", x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
    
class Net_2(nn.Module): # NN, ReLU, 146 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    def __init__(self) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(146, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 4)
        self.fc7 = nn.Linear(4, 2)
        self.fc8 = nn.Linear(2, 1)
        
    
    def forward(self, x):
        # print("q,", x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        return x
    # def __init__(self) -> None:
    #     super().__init__()
    #     dim = 128
    #     self.fcLayers = [nn.Linear(146, dim)]
    #     div = 2
    #     out_ = dim
    #     while out_ > 1:
    #         in_ = out_
    #         out_ = in_//div
    #         self.fcLayers += [nn.Linear(in_, out_)]
    
    # def forward(self, x):
    #     # print("q,", x.shape)
    #     for fc in self.fcLayers[:-1]:
    #         x  = fc(x)
    #         x = F.relu(x)
    #     x = self.fcLayers[-1](x)
    #     return x
    
class Net_3(nn.Module): # NN, ReLU, 146 -> 128 -> 112 -> 96 -> 80 -> 64 -> 48 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    def __init__(self) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(146, 128)
        self.fc2 = nn.Linear(128, 112)
        self.fc3 = nn.Linear(112, 96)
        self.fc4 = nn.Linear(96, 80)
        self.fc5 = nn.Linear(80, 64)
        self.fc6 = nn.Linear(64, 48)
        self.fc7 = nn.Linear(48, 32)
        self.fc8 = nn.Linear(32, 16)
        self.fc9 = nn.Linear(16, 8)
        self.fc10 = nn.Linear(8, 4)
        self.fc11 = nn.Linear(4, 2)
        self.fc12 = nn.Linear(2, 1)
        
    
    def forward(self, x):
        # print("q,", x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)
        x = F.relu(x)
        x = self.fc11(x)
        x = F.relu(x)
        x = self.fc12(x)
        return x
    
class Net_4(nn.Module): # NN, ReLU, 146 -> 128 -> 96 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    def __init__(self) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(146, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 4)
        self.fc8 = nn.Linear(4, 2)
        self.fc9 = nn.Linear(2, 1)
        
    
    def forward(self, x):
        # print("q,", x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        x = F.relu(x)
        x = self.fc9(x)
        
        return x

class Net_5(nn.Module): # NN, LeakyReLU, 146 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    def __init__(self) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(146, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 4)
        self.fc7 = nn.Linear(4, 2)
        self.fc8 = nn.Linear(2, 1)
        self.leaky = nn.LeakyReLU(0.1)
        
    
    def forward(self, x):
        # print("q,", x.shape)
        x = self.fc1(x)
        x = self.leaky(x)
        x = self.fc2(x)
        x = self.leaky(x)
        x = self.fc3(x)
        x = self.leaky(x)
        x = self.fc4(x)
        x = self.leaky(x)
        x = self.fc5(x)
        x = self.leaky(x)
        x = self.fc6(x)
        x = self.leaky(x)
        x = self.fc7(x)
        x = self.leaky(x)
        x = self.fc8(x)
        return x

class Net2(nn.Module): # ConvNet, Conv를 하려면 dataloader에서 추가로 수정해야한다. dataloader의 __getitem__함수에 주석되어있는 expand를 통해 강제로 크기를 늘려서 convolution을 진행한다.
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
    
class Net5(nn.Module): # ConvNet, Conv를 하려면 dataloader에서 추가로 수정해야한다. dataloader의 __getitem__함수에 주석되어있는 expand를 통해 강제로 크기를 늘려서 convolution을 진행한다.
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
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1, padding='same')
        self.conv8 = nn.Conv2d(1024, 2048, 3, 1, padding='same')
        self.dropout = nn.Dropout2d(0.25)
        
        self.fc0 = nn.Linear(2048, 1024)
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
        
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # print("shape2:", x.shape)
        x = torch.flatten(x,1)
        
        x = self.fc0(x)
        x = F.relu(x)
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

def train_(model, cost, optim, train_dl, device): # train_dataset을 학습시키는 함수
    N = len(train_dl.dataset)
    n_batch = int(N / train_dl.batch_size)
    
    model.train()
    losses = []
    for batch, (x,y) in enumerate(train_dl):
        # print(f"x: {x} type: {type(x)}\ny: {y} \n") # 주석처리한 print함수들은 디버깅, 테스트용 코드입니다.
        x = x.to(device).float()
        y = y.to(device).float()
        # print(x.shape)
        # print("test() x.shape", x.shape)
        pred = model(x) # 현재 상태의 model에 input을 넣어 prediction값을 구한다.
        # print("a")
        # print("pred:",pred.shape)
        # print(y.shape) #torch.Size([64, 1])
        # print(1,y[0]) #1 tensor([0.], device='cuda:0')
        # print(2,y[0][0]) #2 tensor(0., device='cuda:0')
        # print(3,y[0][0]==0) #3 tensor(True, device='cuda:0')
        # print(3,y[0][0] in [0,1,2])
        # print(y[0][0]==torch.Tensor([0.]))
        
        cost_ = cost(pred, y) # cost function을 통해 loss값을 구한다. 
        losses.append(cost_.item()) # 
        # Backpropagation.
        optim.zero_grad()
        cost_.backward() # 역전파 계산  
        optim.step() # 최적화 
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
    with torch.no_grad(): # test_ 에선 gradient계산이 필요하지 않으므로 불필요한 연산을 하지 않기위해 no_grad()처리를 해준다.
        for batch, (x,y) in enumerate(test_dl):
            x,y = x.to(device).float(), y.to(device).float()
            
            pred = model(x) 
            loss = loss_fn(pred, y) # cost function을 통해 loss값을 구한다. 
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
    with torch.no_grad(): # predict_ 에선 gradient계산이 필요하지 않으므로 불필요한 연산을 하지 않기위해 no_grad()처리를 해준다.
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
            # quit()
        # L1 loss와 L2 loss 둘 다 판단하는게 필요한 task였기 때문에 predict_함수에서 두 loss를 계산했다.
        # L1과 L2 loss 둘 다 비교하는 이유는 두 loss가 이번 task에서 가지는 특성이 있기 때문이다.
        # L1은 예측 등수가 크게 차이나도 linear한 증가치로 loss를 측정하고,
        # L2는 등수 예측 차이가 1 이하로 나면 (L1에 비해) 상대적으로 적은 loss를, 예측 차이가 1 이상으로(크게)나면 상대적으로 더 큰 loss를 준다.
        # 이는 label이 '등수'라는 상황에선 중요하게 고려해야할 요소라고 판단했다.
        print(f"L1 loss: {sum(abs(y-pred))/len(y)}") 
        print(f"L2 loss: {L}")
            
        
if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset()
    train_ds, test_ds = torch.utils.data.random_split(dataset, [0.8, 0.2]) # 데이터셋을 8 : 2비율로 나눠준다. 
    
    # model = Net2().to(device)
    # model = resnet18(num_classes=1)
    # model = resnet18(num_classes=1).to(device)
    # model = resnet50(num_classes=1).to(device)
    # model = Net_4().to(device)
    model = Net_2().to(device) 
    # 최종적으로 채택한 모델은 Net_2()이다. convolution layer를 추가한 모델도 이와 유사한 성능을 보였으나, 큰 차이가 나지 않았고 computation관점에서 생각을 했을 때 Net_2()가 더 낫다고 생각했다.
    
    criterion = nn.MSELoss().to(device) # loss = Mean Square Error를 사용했다.
    # criterion = CustomLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    epochs = 30
    load_name = "model_param14_justDNN25.pt" # 이미 train시킨 모델의 파라미터를 불러오려면 이 변수 이름을 변경하여 사용한다.
    # model_param12_justDNN23 net2, lr=0.0005, ep=30 bset,,
    # model_param13_justDNN24 net2, lr=0.0003, ep=40
    # model_param14_justDNN25 net2, lr=0.0007, ep=30
    train_dl = DataLoader(train_ds, 
                            batch_size=64, 
                            shuffle=True,
                            drop_last=True
                        ) # train dataset을 dataloader에 담아준다. 
    test_dl  = DataLoader(test_ds, 
                            batch_size=64, 
                            shuffle=True,
                            drop_last=True
                        ) # test datasert을 dataloader에 담아준다. 
    try: # load_name에 맞는 파라미터 저장 정보가 있으면 load해서 성능 측정(predict_)하고, 저장 정보가 없으면 except구문을 실행한다(== train시작).
        # model.load_state_dict(torch.load("model_param6_custom_loss.pt", map_location=device))
        model.load_state_dict(torch.load(load_name, map_location=device))
        predict_(model, criterion, test_dl, device)
        print(1)
    except:
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
        min_ = 999
        for epoch in range(epochs):
            print(f"[ epoch: {epoch+1}/{epochs} ]")
            train_loss.append(train_(model, criterion, optimizer, train_dl, device)) # training & train loss를 list에 저장한다. -> plot을 하기 위한 용도
            temp = test_(model, criterion, test_dl, device) # testing
            test_loss.append(temp) # test loss를 list에 저장한다. -> plot을 하기 위한 용도
            if min_ > temp:  # 가장 loss가 적을때의 파라미터 정보를 저장한다.
                min_ = temp
                torch.save(model.state_dict(), load_name) # 파라미터 정보를 저장하는 코드이다. 
        # torch.save(model.state_dict(), load_name)
        print(train_loss)
        print(test_loss)
        print(min_)
        plt.plot(train_loss, 'b-', label='train_loss') # train_loss plotting
        plt.plot(test_loss, 'r-', label='test_loss')   # test_loss plotting
        plt.legend()
        plt.savefig(f"{load_name}.png") # 사진 저장
        predict_(model, criterion, test_dl, device)
        
        