import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from time import time

import cv2
import os
import numpy as np

import sys
sys.path.append('/home/ashesham/Projects/basics/SLAM/')
# import ipdb;ipdb.set_trace()

from models.models import depthwise_separable_conv
from utils.d3_1.loader import CustomDataset
from utils.d3_1.loss import calculate_orb_features, calculate_orb_features_v2
from utils.d3_1.parameters import ModelParameters
from utils.d3_1.helper import train
from utils.TartanDataset import TartanData
import random

random_seed = 10

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Set the path to the image folder and other parameters
params = ModelParameters(batch_size=8, n_epoches=100, lr=0.0116, layers_per_kernel=4, im_folder='/home/datasets/SLAM/TartainAir', imsize=512, dropout=0.02, n_workers=4)

writer = SummaryWriter(f'runs/experiment_{params.batch_size}_{params.learning_rate}_{params.dropout}_{params.layer_per_kernel}'+'_'+str(int(time())))

# import ipdb;ipdb.set_trace()
# Initialize the models
model3_1 = depthwise_separable_conv(3,1,params.layer_per_kernel ,params.dropout)
model1_3 = depthwise_separable_conv(1,3,params.layer_per_kernel ,params.dropout)

# Create the dataset and dataloader
# dataset = CustomDataset(params.image_folder, params=params)
dataset = TartanData(data_dir=params.image_folder,batch_size=params.batch_size, d_divider=params.num_epochs) #Implemented dataset divider
dataloader = DataLoader(dataset, shuffle=True, num_workers=params.n_workers)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(list(model3_1.parameters())+list(model1_3.parameters()), lr=params.learning_rate)

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model and loss function to the device
model3_1,model1_3 = model3_1.to(device),model1_3.to(device)
criterion = criterion.to(device)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Training loop
for epoch in range(params.num_epochs):
    train_loss = train([model3_1,model1_3], dataloader, optimizer, criterion, device, epoch=epoch,writer=writer)
    print(f"Epoch {epoch + 1}/{params.num_epochs}, Loss: {train_loss}")
    scheduler.step(train_loss)
    
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('loss_epoch', train_loss, epoch)
    

    if epoch%5 == 0:
        # Save the trained model
        torch.save(model3_1.state_dict(), f"./weights/demo/trained_model3_1_0_{epoch}.pth")
        torch.save(model1_3.state_dict(), f"./weights/demo/trained_model1_3_0_{epoch}.pth")
torch.save(model3_1.state_dict(), f"./weights/demo/trained_model3_1_0_{epoch}.pth")
torch.save(model1_3.state_dict(), f"./weights/demo/trained_model1_3_0_{epoch}.pth")

# Save the trained model
# torch.save(model3_1.state_dict(), "trained_model3_1_0.pth")
# torch.save(model1_3.state_dict(), "trained_model1_3_0.pth")