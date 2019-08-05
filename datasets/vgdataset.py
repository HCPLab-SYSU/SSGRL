import torch
import sys
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random
import os
#category_info = '../data/VG/vg_category_1000.json'

class VGDataset(data.Dataset):
    def __init__(self, img_dir, img_list, input_transform, label_path):
        with open(img_list, 'r') as f:
            self.img_names = f.readlines()
        with open(label_path, 'r') as f:
            self.labels = json.load(f) 
        
        self.input_transform = input_transform
        self.img_dir = img_dir
        self.num_classes= 500
    
    def __getitem__(self, index):
        name = self.img_names[index][:-1]
        input = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        #b, g, r = input.split()
        #input = Image.merge("RGB", (r, g, b))
        if self.input_transform:
           input = self.input_transform(input)
        label = np.zeros(self.num_classes).astype(np.float32)
        label[self.labels[name]] = 1.0
        return input, label

    def __len__(self):
        return len(self.img_names)

