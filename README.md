# MambaRecon
Official Implementation for MambaRecon: MRI Reconstruction with Structured State Space Models. Recently featured at WACV 2025. [Paper Link](https://openaccess.thecvf.com/content/WACV2025/html/Korkmaz_MambaRecon_MRI_Reconstruction_with_Structured_State_Space_Models_WACV_2025_paper.html)

## Installation
Clone the repository:
```
git clone git@github.com:yilmazkorkmaz1/MambaRecon.git
```
Create the environment from the environment.yml:
```
conda env create -f environment.yml
```
Activate the environment:
```
conda activate mamba_recon_env
```
Install causal convolution and mamba packages:
```
cd casual-conv1d

python setup.py install
```

```
cd mamba

python setup.py install
```


## Dataset

Download datasets and place them in datasets folder inside code:

https://drive.google.com/drive/folders/1XReBWt_oirOSdfc8rQf5OXIwdkxqX0xF?usp=share_link

## Pretrained Checkpoint

Download pretrained checkpoints:

https://drive.google.com/drive/folders/1aPCqYbREsk5Q-vO8aXwDLFF51pe8XPCq?usp=share_link


## Run Commands
```
python train.py --exp mamba_unrolled --dataset ixi --model mamba_unrolled --patch_size 2 --batch_size 4 --gpu_id 0
```


## Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@InProceedings{Korkmaz_2025_WACV,
    author    = {Korkmaz, Yilmaz and Patel, Vishal M.},
    title     = {MambaRecon: MRI Reconstruction with Structured State Space Models},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {4142-4152}
}
```

## Contact

ykorkma1[at]jhu.edu


## Acknowledgements
We gratefully acknowledge the authors of the following repositories, from which we utilized code in our work:  

- [Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet/tree/main)  
- [VMamba](https://github.com/MzeroMiko/VMamba)  
