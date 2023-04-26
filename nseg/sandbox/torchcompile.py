import torch
from elektronn3.models.unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m = UNet().to(device)
c = torch.compile(m)
x = torch.randn(8, 1, 80, 80, 80, device=device)



import IPython ; IPython.embed(); raise SystemExit