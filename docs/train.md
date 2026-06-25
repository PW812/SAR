## NuScenes
Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data as follows.


**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```
**Prepare nuScenes data**
```
python tools/data_converter/vad_nuscenes_converter.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag vad_nuscenes --version v1.0 --canbus ./data/nuscenes
```
## Prepare
- [Installation](docs/install.md)
- [Dataset](docs/prepare_dataset.md)

## Train and Test
### Train SAR with 4 GPUs 
```shell
cd /path/to/SAR
conda activate sar
python -m torch.distributed.run --nproc_per_node=4 --master_port=2333 tools/train.py projects/configs/SAR/SAR_pts_motion_e2e_v2.py --launcher pytorch --deterministic --work-dir path/to/save/outputs
```
OR
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --master_port=2333 tools/train.py projects/configs/SAR_motion_e2e_v2.py --launcher pytorch --deterministic --work-dir work_dir/SAR
```

### Eval SAR with 1 GPU
```shell
cd /path/to/SAR
conda activate sar
CUDA_VISIBLE_DEVICES=0 python tools/test.py projects/configs/SAR/SAR_pts_motion_e2e_v2.py ckpts/sar.pth --launcher none --eval bbox --tmpdir tmp

```


