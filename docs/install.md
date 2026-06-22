# Step-by-step installation instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation.

Detailed package versions can be found in [requirements.txt](../requirements.txt).



**a. Create a conda virtual environment and activate it.**
```shell
conda create -n sar python=3.8 -y
conda activate sar
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# Recommended torch>=1.9
```

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-5 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.5.2
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
```

**e. Install timm.**
```shell
pip install timm
```

**f. Install mmdet3d.**
```shell
conda activate sar
pip install mmdet3d==1.0.0rc4
```

**g. Install nuscenes-devkit.**
```shell
pip install nuscenes-devkit==1.1.9
```

**h. Clone SAR.**
```shell
git clone https://github.com/PW812/SAR.git
```

