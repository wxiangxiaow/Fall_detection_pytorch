# -*- coding: utf-8 -*-
import os
import cv2
from PIL import Image
import torch

root = os.path.abspath(os.path.join(os.getcwd(), '..'))


def make_one_hot(label, C=3):
    return torch.eye(C)[label, :]

class Pneumonia(object):
    '''
    txt: transferred from stage_2_train_labels.csv
    pick out the 1st row(img_name) and last row(labels)
    read by each line to get images and labels
    '''
    def __init__(self, txt, mode, class_to_idx, transforms=None):        
        fh = open(txt, 'r')
        imgs = []
        if mode == 'train':
            data_path = os.path.join(root, 'dataset_train')
       
        for line in fh:
            line = line.rstrip()
            words = line.split(',')
            fn = words[0]               #img_name
            imgs.append((os.path.join(data_path, fn), int(words[1])))
        self.imgs = imgs 
        self.transform = transforms
        self.mode = mode
        self.class_to_idx = class_to_idx
   
    def __getitem__(self, index):
        img, label = self.imgs[index]
        Img = cv2.imread(img)
        img = Image.fromarray(Img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
