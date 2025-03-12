import numpy as np
import h5py 


def fft2c_np(im):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im, axes=[-1,-2]), norm="ortho"), axes=[-1,-2]) 

def ifft2c_np(d):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(d, axes=[-1,-2]), norm="ortho"), axes=[-1,-2])

def get_ixi_dataset(phase="train"):
    fs_array = []
    us_array = []
    mask_array = []


    target_file="datasets/ixi/ixi_" + phase + ".h5"
    f = h5py.File(target_file,'r')

    data_fs=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=1)
    data_fs = data_fs.astype(np.float32) 
    fs_array.extend(data_fs)

    data_us=np.transpose(np.array(f['data_us']),(1,0,3,2))
    phase_=(data_us[:,1,:,:]*2*np.pi)-np.pi
    data_us=data_us[:,0,:,:]*np.exp(1j*phase_)
    us_array.extend(data_us)

    data_masks = np.expand_dims(np.transpose(np.array(f['us_masks']),(0,2,1)),axis=1)
    data_masks = data_masks.astype(np.float32) 
    mask_array.extend(data_masks)

    us_array = np.asarray(us_array)
    us_array = np.stack([np.real(us_array), np.imag(us_array)], axis=1).astype(np.float32) 

    print("total number of images in ixi:", len(us_array))

    return us_array, fs_array, mask_array