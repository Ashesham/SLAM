import torch

import os
import pandas as pd
import glob
from pathlib import Path

from torchvision.io import read_image
import torchvision.transforms.functional as transform
from torch.utils.data import Dataset

from utils.np_utils.helper import txt_to_q

class TartanData(Dataset):
    def __init__(self, data_dir, batch_size, transform=None, target_transform=None):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform

        self.Env_paths = glob.glob(str(self.data_dir/'*'))
        self.data = []
        self.st_idx = []

        count = 0
        for env in self.Env_paths:
            # env = i.split('/')
            # self.data.append([])
            if len(self.st_idx)==0:
                self.st_idx.append(count)
            for fname in sorted(glob.glob(env+'/P*.txt')):
                im_dir = fname.split('.')[0]
                gts = [[i] for i in list(txt_to_q(fname))]
                [gts[i].append(im_dir+'/'+"%05d" % (i)+'.png') for i in range(len(gts))]
                self.data.append(gts)
                count +=len(gts) 
                self.st_idx.append(count)

    def get_path_idx(self,idx):
        id = 0
        offset = self.batch_size
        for i in range(len(self.st_idx)-1):
            t = idx + offset
            if self.st_idx[i]<t and t<self.st_idx[i+1]:
                idx = t - self.st_idx[i] - self.batch_size 
                return id, idx
            id+=1
            offset += self.batch_size
        raise "Unknown error occured!!!"

    def __len__(self):
        total = 0
        for path in self.data:
            total += len(path)-self.batch_size
        return total

    def __getitem__(self, idx):        
        id, idx = self.get_path_idx(idx)
        print(id,idx)
        for i in range(idx,idx+self.batch_size):
            image = [id,idx,i]#read_image(self.data[id][idx][1])
            label = self.data[id][idx][0]

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            im_tensor = torch.tensor(image).unsqueeze(0) #transform.to_tensor(image).unsqueeze(0)
            l_tensor = torch.tensor(label).unsqueeze(0)#transform.to_tensor(label).unsqueeze(0)
            if i == idx:
                im_batch_tensor = im_tensor
                l_batch_tensor = l_tensor
            else:
                im_batch_tensor =  torch.cat((im_batch_tensor,im_tensor))
                l_batch_tensor =  torch.cat((l_batch_tensor,l_tensor))

        return im_batch_tensor, l_batch_tensor

