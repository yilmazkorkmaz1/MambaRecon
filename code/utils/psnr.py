
from skimage import metrics as measure
import numpy as np 


def compute_psnr(generated_image, original_image):
    data1 = np.float32(generated_image.copy())  
    max1 = np.max(data1)
    data1 /= max1

    data2 = np.float32(original_image.copy())
    max2 = np.max(data2)
    data2 /= max2

    psnr = measure.peak_signal_noise_ratio(data2, data1, data_range=1.0)
    return psnr

def compute_ssim(generated_image, original_image):
    data1 = np.float32( generated_image.copy())  
    max1 = np.max(data1)
    data1 /= max1

    data2 = np.float32(original_image.copy())
    max2 = np.max(data2)
    data2 /= max2
 
    ssim = measure.structural_similarity(data2, data1, sigma=1.5, gaussian_weights=True, use_sample_covariance=False, data_range=1.0)   
    return ssim                        