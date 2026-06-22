# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
import sys
sys.path.append('.')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.datasets import build_dataset
# from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
#from tools.misc.fuse_conv_bn import fuse_module
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='MMDet benchmark a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--savedir', default=None, help='save path')
    parser.add_argument('--samples', default=2000, help='samples to benchmark')
    parser.add_argument(
        '--log-interval', default=50, help='interval of logging')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    print(cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    #if args.fuse_conv_bn:
    #    model = fuse_module(model)

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    n = 10
    import os
    save_dir = args.savedir
    os.makedirs(save_dir, exist_ok=True)
    # benchmark with several samples and take the average
    # for i, data in enumerate(data_loader):
    #     torch.cuda.synchronize()
    #     start_time = time.perf_counter()
    #     with torch.no_grad():
    #         outputs = model(return_loss=False, rescale=True, **data)
    #     scene_query = outputs[0]["pts_bbox"]['scene_query']  # [16, B, C]
    #     scene_query_pos = outputs[0]["pts_bbox"]['scene_query_pos']  # [16, B, C]
    #     H, W = 100, 100
    #     # scene_query_pos_np = scene_query_pos.squeeze(1).cpu().numpy()  # [16, 256]
    #     # pca = PCA(n_components=2)
    #     # scene_query_2d = pca.fit_transform(scene_query_pos_np)  # [16, 2]
    #     # scene_query_2d = (scene_query_2d - scene_query_2d.min()) / (scene_query_2d.max() - scene_query_2d.min())
    #     # scene_query_coords = (scene_query_2d * [H, W]).astype(int)
        
    #     # 取第一个 batch 的 scene_query
    #     scene_query_np = scene_query.cpu().numpy()  # [16, C]

    #     # # PCA 降维到 2D
    #     pca = PCA(n_components=2)
    #     scene_query_2d = pca.fit_transform(scene_query_np)  # [16, 2]
    #     bev_feat = outputs[0]["pts_bbox"]['bev_embed'].detach().cpu().numpy()[0]  # [H*W, C]
    #     bev_feat = bev_feat.reshape(H, W, -1).mean(axis=-1)  # [H, W]，取均值降维
    #     # # 归一化到 BEV 坐标范围
    #     scene_query_2d = (scene_query_2d - scene_query_2d.min()) / (scene_query_2d.max() - scene_query_2d.min())
    #     scene_query_coords = (scene_query_2d * [H, W]).astype(int)
        
    #     # 可视化
    #     plt.figure(figsize=(5,5))
    #     plt.imshow(bev_feat, cmap='viridis', interpolation='nearest')
    #     plt.scatter(scene_query_coords[:, 0], scene_query_coords[:, 1], c='red', s=30)
    #     plt.title("Scene Query Positions in BEV")
    #     plt.axis('off')  # 隐藏坐标轴
    #     file_name = f"BEV_vis_{i}.png"  # `i` 是当前样本索引
    #     file_path = os.path.join(root_path, file_name)

    #     # 保存图像
    #     plt.savefig(file_path, dpi=300, bbox_inches='tight')  # dpi 控制分辨率，bbox_inches 控制边距
    #     plt.close()  # 关闭当前图像，避免内存占用
    #     if i >= n :
    #         break
    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        
        with torch.no_grad():
            outputs = model(return_loss=False, rescale=True, **data)

        # scene_query = outputs[0]["pts_bbox"]['scene_query']        # [16, B, C]
        # scene_query_pos = outputs[0]["pts_bbox"]['scene_query_pos']  # [16, B, C]

        # # 只处理第一个 batch 的数据
        # scene_query = scene_query#.cpu().numpy()        # [16, C]
        # scene_query_pos = scene_query_pos#.cpu().numpy()  # [16, C]
        # # plan_cmd = np.argmax([0]["pts_bbox"]["ego_fut_cmd"][0,0,0])
        # # cmd_list = ['Turn Right', 'Turn Left', 'Go Straight']
        # # plan_cmd_str = cmd_list[plan_cmd]
        # # Step 1: 使用 PCA 将位置编码映射为 2D 坐标
        # # Step 1: 将位置编码降维为 2D 坐标
        # pca_pos = PCA(n_components=2)
        # coords_2d = pca_pos.fit_transform(scene_query_pos)  # [16, 2]

        # # 归一化坐标至图像像素坐标 (0 ~ 99)
        # coords_min = coords_2d.min(axis=0)
        # coords_max = coords_2d.max(axis=0)
        # coords_norm = (coords_2d - coords_min) / (coords_max - coords_min + 1e-6)
        # coords_img = (coords_norm * 99).astype(int)  # [16, 2]

        # # Step 2: 将特征降维为 3D RGB 值
        # pca_feat = PCA(n_components=3)
        # features_rgb = pca_feat.fit_transform(scene_query)  # [16, 3]
        # features_rgb_norm = (features_rgb - features_rgb.min()) / (features_rgb.max() - features_rgb.min() + 1e-6)

        # # Step 3: 创建空白 RGB 图像
        # canvas_rgb = np.zeros((100, 100, 3))

        # # 将颜色值填入图像坐标位置
        # for j, (x, y) in enumerate(coords_img):
        #     canvas_rgb[y, x] = features_rgb_norm[j]

        # # Step 4: 可视化并保存
        # plt.figure(figsize=(5, 5))
        # plt.imshow(canvas_rgb)
        # plt.title(f'RGB Feature Map (Frame {i})')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, f'ego_agent_scene_queryrgb_{i:04d}.png'))
        # plt.close()
        # H, W, C = 100, 100, 256
        # grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
        # grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).cuda()  # [10000, 2]

        # # 映射为 256 维位置编码（可以替换成正弦编码）
        # pos_encoder = torch.nn.Linear(2, 256).cuda()
        # spatial_pos_embed = pos_encoder(grid)
        # query = (scene_query[0] + scene_query_pos[0]).cuda()
        # # 计算 query 与每个位置的相似度（点积或 cosine）
        # sim = torch.nn.functional.cosine_similarity(spatial_pos_embed, query.unsqueeze(0), dim=1)  # [10000]
        # sim = sim.view(H, W).detach().cpu().numpy()

        # # 可视化
        # plt.figure(figsize=(6, 6))
        # plt.imshow(sim, cmap='hot')
        # plt.colorbar(label='Activation')
        # plt.title('Single Query Activation in 2D Space')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('single_query_heatmap.png')
        # plt.close()
        if i >= n:
            break
if __name__ == '__main__':
    main()
