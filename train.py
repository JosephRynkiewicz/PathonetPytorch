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
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as ws
import cv2
from models import *
import math
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_label(fichier):
    """print image labels"""
    with open(fichier) as fichier_label:
        label = json.load(fichier_label)
    return label

def count_cells(labels):
    positives=0
    negatives=0
    TIL=0
    ratio=0
    for cell in labels:
        if cell['label_id']==1:
            positives +=1
        elif cell['label_id']==2:
            negatives +=1
        else:
            TIL +=1
    total = positives+negatives
    if (total != 0):
        ratio = positives/total
    else:
        ratio=0
    #return positives, negatives, TIL, ratio
    return torch.Tensor([positives,negatives]), torch.Tensor([ratio])



def sum_cut_of(pred_ratios, ratios):
    predinf16=pred_ratios<0.16
    ratiosinf16=ratios<0.16
    predsup30=pred_ratios>0.3
    ratiossup30=ratios>0.3
    predsup16inf30=~predinf16 & ~predsup30
    ratiossup16inf30=~ratiosinf16 & ~ratiossup30
    correctinf16=sum(predinf16 & ratiosinf16).item()
    correctsup30=sum(predsup30 & ratiossup30).item()
    correctsup16inf30=sum(predsup16inf30 & ratiossup16inf30).item()
    return correctinf16,  correctsup16inf30, correctsup30


transform_train = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
])


transform_test = transforms.Compose([
    transforms.ConvertImageDtype(torch.float),
])



class KI67Dataset(Dataset):
    def __init__(self, img_dir="", transform=None, counts_transform=count_cells):
        self.img_dir = img_dir
        self.transform = transform
        self.counts_transform = counts_transform
        self.data_img = [img_dir+f for f in os.listdir(img_dir) if '.jpg' in f]
        self.data_labels = [f[:-3]+"npy" for f in self.data_img]
        self.data_counts = [f[:-3]+"json" for f in self.data_img]
    def __len__(self):
        return len(self.data_img)
    def __getitem__(self, idx):
        image = read_image(self.data_img[idx])
        #On est obligé de traiter les labels à ceux niveau
        #Bug de Pytorch?
        label = np.load(self.data_labels[idx])
        label = label.astype(np.float32)
        label = np.transpose(label,(2,0,1))
        label = torch.from_numpy(label)
        count = get_label(self.data_counts[idx])
        if self.transform:
            image = self.transform(image)
        if self.counts_transform:
            count, ratio = self.counts_transform(count)
        return image, label, count, ratio

#Il faut des liens symboliques sur les bons répertoires
path_train="trainset/"
path_test="testset/"
training_data=KI67Dataset(path_train,transform=transform_train)
valid_data=KI67Dataset(path_train,transform=transform_test)
test_data=KI67Dataset(path_test,transform=transform_test)
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
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)





def tensor_to_image(tensor,Normalisation=255):
    tensor = torch.ceil(tensor*Normalisation)
    tensor = np.array(tensor, dtype=np.uint8)
    return Image.fromarray(tensor,mode='RGB')


figure = plt.figure()
num_of_images = 6

for index in range(num_of_images):
    image, label, _, _ = training_data[index]
    image = image.permute(1,2,0)
    label = label.permute(1,2,0)
    aff_image=tensor_to_image(image)
    aff_label = tensor_to_image(label,Normalisation=1)
    plt.subplot(3, 4, 2*index+1)
    plt.axis('off')
    plt.imshow(aff_image)
    plt.subplot(3, 4, 2*(index+1))
    plt.axis('off')
    plt.imshow(aff_label)


plt.show()




# definition of hyperparameters
lr = 0.01
ne = 30
nsc = 10
gamma = 0.1
net=Unet()
net = net.to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=lr)
lr_sc = lr_scheduler.StepLR(optimizer, step_size=nsc, gamma=gamma)

def train(epoch,trainloader):
    net.train()
    train_loss = 0
    correctinf16 = 0
    correctsup16inf30=0
    correctsup30 = 0
    total = 0
    mse_pred_ratios=0
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
        


def test(epoch,validloader):
    global best_acc
    net.eval()
    test_loss = 0
    correctinf16 = 0
    correctsup16inf30=0
    correctsup30 = 0    
    total = 0
    mse_pred_ratios=0
    with torch.no_grad():
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
    checkpoint = torch.load('./checkpoint/ckpt.t7')
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
            #loop.set_postfix(mse=(test_loss/total)*batch_size)
        print("\nMSE finale : ", test_loss/total)
        print("\nRMSE_ratio finale : ", math.sqrt(mse_pred_ratios/total))
        print("\ncutoff final : ", (correctinf16+correctsup30+correctsup16inf30)/total)


best_acc = 1e36        
for epoch in range(0, ne):
    train(epoch,train_dataloader)
    test(epoch,valid_dataloader)
    lr_sc.step()


testfinal(test_dataloader)



