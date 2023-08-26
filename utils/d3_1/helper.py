import cv2
import numpy as np
import torch
from utils.d3_1.loss import calculate_orb_features_v2, sift_matches, find_poses
from utils.loss import ape

def train(models, dataloader, optimizer, criterion, device, alpha=0.1, epoch=0, writer=None):
    scale = 1
    if type(writer) == type(None):
       record = False
    else:
       record = True

    model1 = models[0].train()
    model3 = models[1].train()
    total_loss = 0
    counter =1
    # b_s = -1
    # import ipdb;ipdb.set_trace()
    for image_batches in dataloader:
        # if b_s == -1:
        #   b_s = len(images)
        images, labels = image_batches[0][0].to(device), image_batches[1][0].to(device)

        # Forward pass
        outputs3_1 = model1(images)
        outputs1_3 = model3(outputs3_1)

        # Calculate the number of ORB features detected in the output images
        P_ = find_poses(outputs3_1)
        # loss = ape(labels,P_,True,True)

        # Calculate the loss based on the number of ORB features
        loss1 =  ape(labels,P_,True,True)
        loss2 =  criterion((outputs1_3/torch.max(outputs1_3)),images)*255
        # loss2 = alpha*n_features
        # if loss1.item()<loss2.item():
        #   alpha = alpha*0.9
        #   scale = scale*1/0.9
        #   loss2 = alpha*n_features
        loss = loss1 + loss2
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        counter +=1
        if counter % 100 == 0:
          if type(device) == torch.device:
            P_ref = find_poses(images)
            loss_ref = ape(labels,P_ref,True,True)
            if device.type == 'cuda':
              print("Loss = ",loss.item(),loss1.item(),'/',loss_ref.item(),loss2.item())

          if device==0 or device=='cuda':
            print("Loss = ",loss.item(),loss1.item(),'/',loss_ref.item(),loss2.item())
            
        #     writer.add_image('Images1',images[4].cpu().detach().numpy(), epoch * len(dataloader) + counter)
        #     writer.add_image('Images2',images[5].cpu().detach().numpy(), epoch * len(dataloader) + counter)
        if record:
          # writer.add_scalar('Matching failures', fails, epoch * len(dataloader) + counter)
          # writer.add_scalar('Reconstruction loss', loss1.item()*scale, epoch * len(dataloader) + counter)
          writer.add_scalar('Pose APE loss', loss1.item(), epoch * len(dataloader) + counter)
          writer.add_scalar('Reconstruction loss', loss2.item(), epoch * len(dataloader) + counter)
          writer.add_scalar('Total loss', loss.item(), epoch * len(dataloader) + counter)
          # writer.add_scalar('Total loss', loss.item(), epoch * len(dataloader) + counter)

    return total_loss / len(dataloader)