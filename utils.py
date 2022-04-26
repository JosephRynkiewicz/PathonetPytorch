import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from PIL import Image
import random


def tensor_to_image(tensor,Normalisation=255):
    tensor = torch.ceil(tensor*Normalisation)
    tensor = np.array(tensor, dtype=np.uint8)
    return Image.fromarray(tensor,mode='RGB')




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




def MyTransform(image, tarjet, angles=[0,90,180,270]):
        angle = random.choice(angles)
        image=TF.rotate(image, angle)
        tarjet=TF.rotate(tarjet,angle)
        if random.random() > 0.5:
            image=TF.hflip(image)
            tarjet=TF.hflip(tarjet)
        if random.random() > 0.5:
            image=TF.vflip(image)
            tarjet=TF.vflip(tarjet)            
        return image, tarjet



class KI67Dataset(Dataset):
    def __init__(self, img_dir="", transform=None, mytransform=MyTransform, counts_transform=count_cells):
        self.img_dir = img_dir
        self.transform = transform
        self.mytransform = mytransform
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
        if self.mytransform:
            image, label = self.mytransform(image,label)
        if self.counts_transform:
            count, ratio = self.counts_transform(count)
        return image, label, count, ratio

