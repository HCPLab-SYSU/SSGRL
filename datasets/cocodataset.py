import torch
import sys
sys.path.append('/data1/multi-label/MS-COCO_2014/cocoapi/PythonAPI')
#sys.path.append('/home/chentianshui/xmx/multi-label/cocoapi/PythonAPI')
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
import json
import random


class CoCoDataset(data.Dataset):
    def __init__(self, image_dir, anno_path, input_transform=None, labels_path=None):
        self.coco = dset.CocoDetection(root=image_dir, annFile=anno_path)
        with open('./data/coco/category.json','r') as load_category:
            self.category_map = json.load(load_category)
	self.input_transform = input_transform
	self.labels_path = labels_path
	
        self.labels = []
        if self.labels_path:
	    self.labels = np.load(self.labels_path).astype(np.float64)
            self.labels = (self.labels > 0).astype(np.float64)
 	else:
            l = len(self.coco)
            for i in range(l):
                item = self.coco[i]
                print(i)
                categories = self.getCategoryList(item[1])
                label = self.getLabelVector(categories)
                self.labels.append(label)


    def __getitem__(self, index):
        input = self.coco[index][0]
        if self.input_transform:
            input = self.input_transform(input)
        return input, self.labels[index]


    def getCategoryList(self, item):
        categories = set()
        for t in item:
            categories.add(t['category_id'])
        return list(categories)

    def getLabelVector(self, categories):
        label = np.zeros(80)
        label_num = len(categories)
        for c in categories:
            index = self.category_map[str(c)]-1
            label[index] = 1.0 / label_num
        return label


    def __len__(self):
        return len(self.coco)
