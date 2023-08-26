from torch import nn
import torch.nn.functional as F
class depthwise_separable_conv(nn.Module):
 def __init__(self, nin=3, nout=1,kernels_per_layer=1,p=0.0):
   super(depthwise_separable_conv, self).__init__()
   self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
   self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
   self.dropout = nn.Dropout2d(p=p)

 def forward(self, x):
   out = self.depthwise(x)
   out = F.relu(self.pointwise(out))
   return self.dropout(out)