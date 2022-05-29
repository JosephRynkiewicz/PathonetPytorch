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
from models import *
import math
import torch.nn.functional as F

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



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



path_test="testset/"
test_data=KI67Dataset(path_test,transform=transform_test)
test_dataloader = DataLoader(test_data, batch_size=10, shuffle=False)



#net=Unet()
net=PathoNet()
net = net.to(device)
criterion = nn.MSELoss(reduction='mean')


def evalfinal(testloader):
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
        print("MSE finale : ", test_loss/total)
        print("RMSE_ratio finale : ", math.sqrt(mse_pred_ratios/total))
        print("cutoff final : ", (correctinf16+correctsup30+correctsup16inf30)/total)
        print("\n")




def eval_pts(img_dir=""):
    img_dir = img_dir
    data_img = [img_dir+f for f in os.listdir(img_dir) if '.jpg' in f]
    #Group data by patients
    grouped_data= {}
    for im in data_img:
        key = im.partition('_')[0]
        grouped_data.setdefault(key, []).append(im)
    grouped_data = list(grouped_data.values())
    pred_ki67_pt=torch.zeros(23)
    gt_ki67_pt=torch.zeros(23)
    with torch.no_grad():
        for pt_idx, pt_path in enumerate(grouped_data):
            print("Patient ",pt_idx)
            lb=[]
            pred_count=torch.zeros(3)
            gt_count=torch.zeros(2)
            for i,d in enumerate(grouped_data[pt_idx]):
                img = read_image(d)
                img = transform_test(img)
                img = img.unsqueeze(0)
                img = img.to(device)
                output=net(img)
                pred_count += torch.sum(F.relu(output),(2,3)).to('cpu').squeeze()
                data_count = get_label(d[:-3]+"json")
                count, _ = count_cells(data_count)
                gt_count += count
            gt_ki67_pt[pt_idx]=gt_count[0]/(gt_count[0]+gt_count[1])
            pred_ki67_pt[pt_idx]=pred_count[0]/(pred_count[0]+pred_count[1])
    mse_pred_ratios=math.sqrt(torch.mean(torch.square(gt_ki67_pt-pred_ki67_pt)))
    correctinf16, correctsup16inf30, correctsup30 = sum_cut_of(pred_ki67_pt, gt_ki67_pt)
    print("rmse patients : ", mse_pred_ratios)
    print("cutoff patients : ", (correctinf16+correctsup30+correctsup16inf30)/23)
    return gt_ki67_pt, pred_ki67_pt
        


checkpoint = torch.load('./checkpoint/ckpt.t7',map_location='cuda:0')
net.load_state_dict(checkpoint['net'],strict=False)
net.eval()

evalfinal(test_dataloader)
gt_ki67_pt, pred_ki67_pt = eval_pts(path_test)
gt_ki67_pt=gt_ki67_pt.numpy()
pred_ki67_pt=pred_ki67_pt.numpy()
plt.scatter(gt_ki67_pt,pred_ki67_pt)
plt.xlabel("Taux ki67")
plt.ylabel("Pred ki67")
plt.show()
