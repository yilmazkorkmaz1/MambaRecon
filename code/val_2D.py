import numpy as np
import torch
from utils import psnr


def calculate_psnr_ssim(pred, gt):
    psnr_, ssim_ = psnr.compute_psnr(pred, gt), psnr.compute_ssim(pred, gt)
    if np.isnan(psnr_) or np.isnan(ssim_):
        return 0,0
    else:
        return psnr_, ssim_


def test_single_slice(image, target, us_mask, coil_map, net):
    target = target.cpu().detach().numpy()

    net.eval()
    with torch.no_grad():
        out = net(image, us_mask, coil_map)
        out = torch.abs(out[:,0,:,:] + 1j * out[:,1,:,:])
    
    out = out.cpu().detach().numpy()
    metric_list = []
    metric_list.append(calculate_psnr_ssim(out[0,:,:], target[0,0,:,:]))
    return metric_list