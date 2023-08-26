from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import torchvision.transforms as transforms
import torch

# Custom dataset class for loading images and their corresponding labels
class CustomDataset(Dataset):
    def __init__(self, data_folder, params):
        transform = transforms.Compose([
              transforms.Resize((params.im_size, params.im_size)),
              transforms.ToTensor()
          ])
        self.data = ImageFolder(data_folder, transform=transform)
        self.paired = True
        self.indx_prev = -1

    def __getitem__(self, index):
        if self.paired:
          self.indx_prev = index
          self.paired = False
        else:
          self.paired = True
        
        image, _ = self.data[index]
        if index+1 < self.__len__():
          image1,_ = self.data[index+1]
        else:
          image1,_ = self.data[index-1]
        image = torch.stack((image,image1))
        return image

    def __len__(self):
        return len(self.data)