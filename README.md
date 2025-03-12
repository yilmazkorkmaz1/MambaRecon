# MambaRecon
Official Implementation for MambaRecon: MRI Reconstruction with Structured State Space Models.

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
@article{korkmaz2024mambarecon,
  title={MambaRecon: MRI Reconstruction with Structured State Space Models},
  author={Korkmaz, Yilmaz and Patel, Vishal M},
  journal={arXiv preprint arXiv:2409.12401},
  year={2024}
}
```

## Contact

ykorkma1[at]jhu.edu


## Acknowledgements
We gratefully acknowledge the authors of the following repositories, from which we utilized code in our work:  

- [Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet/tree/main)  
- [VMamba](https://github.com/MzeroMiko/VMamba)  
