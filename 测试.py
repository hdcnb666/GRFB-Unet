import torch
from src import GRFBUNet

model = GRFBUNet(in_channels=3, num_classes=2, base_c=32).cuda()
x = torch.randn(1, 3, 480, 480).cuda()
y = model(x)
print(y.shape)
