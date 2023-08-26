import torch

class ModelParameters:
    def __init__(self, batch_size=2**4, n_epoches=10, lr=2e-3, layers_per_kernel=3, im_folder="data", imsize=512,dropout=0., n_workers=0):
        self.image_folder = im_folder
        self.batch_size = batch_size
        self.num_epochs = n_epoches
        self.learning_rate = lr
        self.dropout = dropout
        self.layer_per_kernel = layers_per_kernel
        self.im_size = imsize
        self.alpha = 100/(self.batch_size*self.im_size)
        self.random_seed = 10
        self.n_workers = n_workers