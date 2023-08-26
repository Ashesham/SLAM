import os
import torch
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist

from utils.d3_1.loader import CustomDataset



class Dist_Helper:
    def __init__(self, params, rank, world_size) -> None:
        self.params = params
        self.rank = rank
        self.world_size = world_size
        
        torch.manual_seed(params.random_seed)
        np.random.seed(params.random_seed)
        random.seed(params.random_seed)
        
        self.dataloader = self.prepare_dataloader()

        

    def prepare_dataloader(self, pin_memory=False, num_workers=0):
        batch_size = self.params.batch_size
        dataset = CustomDataset(self.params.image_folder, params=self.params)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False)
        
        dataloader = DataLoader(dataset, batch_size=int(batch_size/2), pin_memory=pin_memory, 
                                num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)

        return dataloader
    
    def setup(self):    
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'    
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)