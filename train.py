import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import json
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from models import *
from utils import *
import math
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'




transform_train = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
])


transform_test = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
])




#path_train=args.inputPathtrain
path_train="trainset/"
path_test="testset/"
training_data=KI67Dataset(path_train,transform=transform_train)
valid_data=KI67Dataset(path_train,transform=transform_test)
test_data=KI67Dataset(path_test,transform=transform_test,mytransform=None)
split=300
num_train=len(training_data)
indices = list(range(num_train))
np.random.seed(123)
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
training_data = torch.utils.data.Subset(training_data,train_idx)
valid_data = torch.utils.data.Subset(valid_data,valid_idx)
batch_size=32
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# definition of hyperparameters
lr = 0.01
ne = 30
nsc = 10
gamma = 0.1
wd=1e-9
net=Unet()
net = net.to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=wd)
lr_sc = lr_scheduler.StepLR(optimizer, step_size=nsc, gamma=gamma)

def train(epoch,trainloader,nb_train=1):
    net.train()
    train_loss = 0
    correctinf16 = 0
    correctsup16inf30=0
    correctsup30 = 0
    total = 0
    mse_pred_ratios=0
    for i in range(nb_train):
        loop = tqdm(enumerate(trainloader), total=len(trainloader))
        for batch_idx, (inputs, targets, counts, ratios) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pred_counts=torch.sum(F.relu(outputs),(2,3)).to('cpu')
            pred_ratios=pred_counts[:,0]/(pred_counts[:,0]+pred_counts[:,1])
            ratios=ratios.squeeze(1)
            mse_pred_ratios+=torch.sum(torch.square(pred_ratios-ratios)).item()
            inf16, sup16inf30, sup30=sum_cut_of(pred_ratios, ratios)
            correctinf16+=inf16
            correctsup16inf30+=sup16inf30
            correctsup30+=sup30
            train_loss += loss.item()*targets.size(0)
            total += targets.size(0)
            loop.set_description(f"Epoch [{epoch}]")
            loop.set_postfix(mse=train_loss/total, rmse_ratio=math.sqrt(mse_pred_ratios/total), cutoff=(correctinf16+correctsup30+correctsup16inf30)/total)
        


def test(validloader,nb_val=1):
    global best_acc
    net.eval()
    test_loss = 0
    correctinf16 = 0
    correctsup16inf30=0
    correctsup30 = 0    
    total = 0
    mse_pred_ratios=0
    with torch.no_grad():
        for i in range(nb_val):
            loop = tqdm(enumerate(validloader), total=len(validloader))
            for batch_idx, (inputs, targets, counts, ratios) in loop:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                pred_counts=torch.sum(F.relu(outputs),(2,3)).to('cpu')
                pred_ratios=pred_counts[:,0]/(pred_counts[:,0]+pred_counts[:,1])
                ratios=ratios.squeeze(1)
                mse_pred_ratios+=torch.sum(torch.square(pred_ratios-ratios)).item()
                inf16, sup16inf30, sup30=sum_cut_of(pred_ratios, ratios)
                correctinf16+=inf16
                correctsup16inf30+=sup16inf30
                correctsup30+=sup30            
                loss = criterion(outputs, targets)
                test_loss += loss.item()*targets.size(0)
                total += targets.size(0)
                loop.set_postfix(mse=test_loss/total, rmse_ratio=math.sqrt(mse_pred_ratios/total), cutoff=(correctinf16+correctsup30+correctsup16inf30)/total)
    # Save checkpoint.
    acc = test_loss
    if acc < best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

def testfinal(testloader):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7',map_location='cuda:0')
    net.load_state_dict(checkpoint['net'],strict=False)
    net.eval()
    test_loss = 0
    correctinf16 = 0
    correctsup16inf30=0
    correctsup30 = 0
    total = 0
    mse_pred_ratios=0
    with torch.no_grad():
        loop = tqdm(enumerate(testloader), total=len(testloader))
        for batch_idx, (inputs, targets, counts, ratios) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()*targets.size(0)
            pred_counts=torch.sum(F.relu(outputs),(2,3)).to('cpu')
            pred_ratios=pred_counts[:,0]/(pred_counts[:,0]+pred_counts[:,1])
            ratios=ratios.squeeze(1)
            mse_pred_ratios+=torch.sum(torch.square(pred_ratios-ratios)).item()
            inf16, sup16inf30, sup30=sum_cut_of(pred_ratios, ratios)
            correctinf16+=inf16
            correctsup16inf30+=sup16inf30
            correctsup30+=sup30            
            total += targets.size(0)
            loop.set_postfix(mse=test_loss/total, rmse_ratio=math.sqrt(mse_pred_ratios/total), cutoff=(correctinf16+correctsup30+correctsup16inf30)/total)
        print("\nMSE finale : ", test_loss/total)
        print("\nRMSE_ratio finale : ", math.sqrt(mse_pred_ratios/total))
        print("\ncutoff final : ", (correctinf16+correctsup30+correctsup16inf30)/total)






best_acc = 1e36        
for epoch in range(0, ne):
    train(epoch,train_dataloader,nb_train=6)
    test(valid_dataloader,nb_val=6)
    lr_sc.step()


testfinal(test_dataloader)








