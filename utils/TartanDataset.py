import torch

import os
import pandas as pd
import glob
from pathlib import Path

from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image 
import numpy as np

from utils.np_utils.helper import txt_to_q, pose_vec_q_to_mat, inv

class TartanData(Dataset):
    def __init__(self, data_dir, batch_size, transform=transforms.Compose([transforms.ToTensor()]), target_transform=torch.tensor, d_divider=1):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform

        self.Env_paths = glob.glob(str(self.data_dir/'*'))
        self.data = []
        self.st_idx = []

        self.divider = d_divider

        count = 0
        for env in self.Env_paths:
            if len(self.st_idx)==0:
                self.st_idx.append(count)
            for fname in sorted(glob.glob(env+'/Hard/P*/pose_left.txt')):
                im_dir = '/'.join(fname.split('/')[:-1])+'/image_left'
                gts = [[pose_vec_q_to_mat(i)] for i in list(txt_to_q(fname))]
                [gts[i].append(im_dir+'/'+"%06d" % (i)+'_left.png') for i in range(len(gts))]
                import os
                for i in range(len(gts)):
                    assert os.path.exists(gts[i][-1]), f"Image path {gts[i][-1]} doesn't exist" 
                self.data.append(gts)
                count +=len(gts)
                self.st_idx.append(count)

        self.all_prev_idx = []

    def get_path_idx(self,idx):
        while idx in self.all_prev_idx:
            if idx+self.__len__()>=self.__len__()*self.divider:
                self.all_prev_idx = []
            idx += self.__len__()
        self.all_prev_idx.append(idx)
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
        return int(total/self.divider)

    def __getitem__(self, idx):        
        id, idx = self.get_path_idx(idx)
        first = True
        init_pose = None
        for i in range(idx,idx+self.batch_size):
            image = Image.open(self.data[id][i][1])
            if first:
                init_pose = inv(self.data[id][i][0])
                label = np.eye(4)
                first = False
            else:
                label = init_pose @ self.data[id][i][0]

            if self.transform:
                im_tensor = self.transform(image).unsqueeze(0)
            if self.target_transform:
                l_tensor = self.target_transform(label).unsqueeze(0)

            # im_tensor = torch.tensor(image).unsqueeze(0) #transform.to_tensor(image).unsqueeze(0)
            # l_tensor = torch.tensor(label).unsqueeze(0)#transform.to_tensor(label).unsqueeze(0)
            if i == idx:
                im_batch_tensor = im_tensor
                l_batch_tensor = l_tensor
            else:
                im_batch_tensor =  torch.cat((im_batch_tensor,im_tensor))
                l_batch_tensor =  torch.cat((l_batch_tensor,l_tensor))

        return im_batch_tensor, l_batch_tensor

