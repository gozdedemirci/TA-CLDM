import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"  # specify which GPU(s) to be used

import numpy as np 
import torch
import torch.nn as nn
from .models.net_factory import net_factory

seed = 1337
np.random.seed(seed)
torch.cuda.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

unet = net_factory(net_type='unetorg', in_chns=3, class_num=2, drop=True)
unet= nn.DataParallel(unet)

## Loading the model state dict
ch_path = 'best_model.pth'
# checkpoint = torch.load(ch_path, weights_only=False)
checkpoint = torch.load(ch_path)

# Loading the trained model weight
unet.load_state_dict(checkpoint['state_dict'])
unet = unet.module
unet.eval()

with torch.no_grad():
    for param in unet.parameters():
        param.requires_grad = False
    
print("Model loaded successfully")

from torchvision import transforms as T
vessel_transformer = T.Compose([ T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

if __name__ == "__main__":
    import cv2
    sample_img = cv2.imread('sample.png')
    sample_img = cv2.resize(sample_img, (256, 256))
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    sample_img = sample_img / 255.0
    sample_img = torch.tensor(sample_img).permute(2, 0, 1).float().unsqueeze(0)
    input_img = vessel_transformer(sample_img)

    vessel_mask = torch.softmax(unet(input_img.to(device)), dim=1)[:,1,:,:].cpu().detach().numpy()
    cv2.imwrite('vessel_mask.png', vessel_mask[0]*255)
