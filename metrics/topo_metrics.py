import numpy as np
import torch
import sys
import cv2
import cc3d

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import variation_of_information
from sklearn.metrics import adjusted_rand_score
import diplib as dip

sys.path.append("/home/gdemi/multi_view/")
from segmentation.pipeline import unet, vessel_transformer
from segmentation.pipeline_fn import get_unet

device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
unet, vessel_transformer = get_unet('rop')
unet = unet.to(device)
unet.eval()

def get_ssim_rgb(img1, img2):
    img1 = img1.squeeze().cpu().permute(1,2,0).numpy()*255
    img2 = img2.squeeze().cpu().permute(1,2,0).numpy()*255

    ssim_res = 0
    for i in range(3):
        ssim_res += ssim(img1[:,:,i], img2[:,:,i], data_range=255, multichannel=False)
    return ssim_res/3

def conn_comp(arr):
    labels_out, numcomp = cc3d.connected_components(arr, connectivity=26, return_N=True) # 26-connected
    return numcomp

def post_process(prediction):
    kernel = np.ones((3,3), np.uint8) 
    d_im = cv2.dilate(prediction.permute(1,2,0).cpu().numpy(), kernel, iterations=1)
    e_im = cv2.erode(d_im, kernel, iterations=1) 

    out = dip.AreaOpening(e_im, filterSize=150, connectivity=2)
    out = torch.from_numpy(np.array(out)).unsqueeze(0).to(device, dtype=torch.float32)
    return out

def get_betti_error(arr1, arr2, patchsize=[64,64], stepsize=[64,64]):
    arrsize = arr1.shape
    all_betti = []
    
    for x in range(0,arrsize[0],stepsize[0]):
        for y in range(0,arrsize[1],stepsize[1]):
            newidx = [x+patchsize[0],y+patchsize[1]]
            if(check_bounds([x,y],arrsize) and check_bounds(newidx,arrsize)):
                minivol1 = arr1[x:newidx[0],y:newidx[1]]
                minians1 = conn_comp(minivol1)

                minivol2 = arr2[x:newidx[0],y:newidx[1]]
                minians2 = conn_comp(minivol2)

                all_betti.append(np.abs(minians1-minians2))

    avg_betti = np.asarray(all_betti).mean()
    return avg_betti

def check_bounds(idx, volsize):
    if idx[0] < 0 or idx[0] > volsize[0]:
        return False
    if idx[1] < 0 or idx[1] > volsize[1]:
        return False
    return True

def get_topo_measures(imgs_real_batch, imgs_fake_batch, domain='rop'):
    # segmentor, vessel_transformer
    with torch.no_grad():

        if domain == 'rop':
            # rgb
            temp_real = cv2.cvtColor(imgs_real_batch[0].cpu().permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)
            img_pre_real = temp_real/255.
            xx_real = vessel_transformer(torch.from_numpy(img_pre_real).permute(2,0,1).unsqueeze(0).float().to(device))
        elif domain == 'xray':
            temp_real = cv2.cvtColor(imgs_real_batch[0].cpu().permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)
            img_pre_real = temp_real/255.
            xx_real = vessel_transformer(torch.from_numpy(img_pre_real).unsqueeze(0).unsqueeze(0).float().to(device))
        mask_real = torch.softmax(unet(xx_real), dim=1)
        mask_real = post_process(mask_real[:,1,:,:])
        mask_real[mask_real > 0.3] = 1
        mask_real[mask_real < 1] = 0
        # masked_real = torch.argmax(mask_real, dim=1)
        masked_real = mask_real.cpu().squeeze(0).numpy().astype(int)

        if domain == 'rop':
            # rgb
            temp_fake = cv2.cvtColor(imgs_fake_batch[0].cpu().permute(1,2,0).numpy(), cv2.COLOR_BGR2RGB)
            img_pre_fake = temp_fake/255.
            xx_fake = vessel_transformer(torch.from_numpy(img_pre_fake).permute(2,0,1).unsqueeze(0).float().to(device))
        elif domain == 'xray':
            temp_fake = cv2.cvtColor(imgs_fake_batch[0].cpu().permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)
            img_pre_fake = temp_fake/255.
            xx_fake = vessel_transformer(torch.from_numpy(img_pre_fake).unsqueeze(0).unsqueeze(0).float().to(device))

        # temp_fake = cv2.cvtColor(imgs_fake_batch[0].cpu().permute(1,2,0).numpy(), cv2.COLOR_BGR2GRAY)
        # img_pre_fake = temp_fake/255. # imgs_fake_batch/255.
        # xx_fake = vessel_transformer(torch.from_numpy(img_pre_fake).unsqueeze(0).unsqueeze(0).float().to(device))
        mask_fake = torch.softmax(unet(xx_fake), dim=1)
        mask_fake = post_process(mask_fake[:,1,:,:])
        # # rop generation - thresholding
        mask_fake[mask_fake > 0.3] = 1
        mask_fake[mask_fake < 1] = 0
        # masked_fake = torch.argmax(mask_fake, dim=1)
        masked_fake = mask_fake.cpu().squeeze(0).numpy().astype(int)

        voi = sum(variation_of_information(masked_real, masked_fake))
        ari = adjusted_rand_score(masked_real.ravel(), masked_fake.ravel())
        betti = get_betti_error(masked_real, masked_fake, patchsize=[64,64], stepsize=[64,64])

        return betti, voi, ari
    
def get_vessel_mask(imgs_real_batch, imgs_fake_batch):
    with torch.no_grad():
        img_pre_real = imgs_real_batch/255.
        xx_real = vessel_transformer(img_pre_real)
        mask_real = torch.softmax(unet(xx_real), dim=1)
        mask_real = post_process(mask_real[:,1,:,:])
        mask_real[mask_real > 0.3] = 1
        mask_real[mask_real < 0] = 0
        # masked_real = torch.argmax(mask_real, dim=1)
        masked_real = mask_real.cpu().squeeze(0).numpy().astype(int)

        img_pre_fake = imgs_fake_batch/255.
        xx_fake = vessel_transformer(img_pre_fake)
        mask_fake = torch.softmax(unet(xx_fake), dim=1)
        mask_fake = post_process(mask_fake[:,1,:,:])
        mask_fake[mask_fake > 0.3] = 1
        mask_fake[mask_fake < 0] = 0
        # masked_fake = torch.argmax(mask_fake, dim=1)
        masked_fake = mask_fake.cpu().squeeze(0).numpy().astype(int)

        return masked_real, masked_fake