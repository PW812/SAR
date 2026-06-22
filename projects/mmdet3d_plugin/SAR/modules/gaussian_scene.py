import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GaussianScene(nn.Module):
    def __init__(self, 
                 embed_dims=256,
                 num_gaussians=1000,
                 gaussian_dim=3,
                 temperature=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_gaussians = num_gaussians
        self.gaussian_dim = gaussian_dim
        self.temperature = temperature
        
        # 高斯参数预测网络
        self.gaussian_net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, num_gaussians * (gaussian_dim * 2 + 1))  # 位置(3) + 协方差(3) + 权重(1)
        )
        
        # 场景特征提取网络
        self.scene_net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims)
        )
        
    def forward(self, memory, inter_states):
        """
        Args:
            memory: [(h*w + v*h*w), bs, c] 场景特征
            inter_states: [num_layers, num_query, bs, dim] agent特征
        Returns:
            scene_features: [bs, num_gaussians, embed_dims] 场景特征
            gaussian_params: [bs, num_gaussians, gaussian_dim*2+1] 高斯参数
        """
        bs = memory.size(1)
        
        # 1. 提取场景特征
        scene_features = self.scene_net(memory.mean(0))  # [bs, embed_dims]
        
        # 2. 预测高斯参数
        gaussian_params = self.gaussian_net(scene_features)  # [bs, num_gaussians * (gaussian_dim*2+1)]
        gaussian_params = gaussian_params.view(bs, self.num_gaussians, -1)
        
        # 3. 分离位置、协方差和权重
        positions = gaussian_params[..., :self.gaussian_dim]
        covariances = F.softplus(gaussian_params[..., self.gaussian_dim:-1])
        weights = F.softmax(gaussian_params[..., -1] / self.temperature, dim=-1)
        
        # 4. 将agent特征与场景特征融合
        agent_features = inter_states[-1].mean(0)  # [num_query, bs, dim]
        agent_features = agent_features.permute(1, 0, 2)  # [bs, num_query, dim]
        
        # 5. 计算场景表示
        scene_representation = torch.cat([
            positions,
            covariances,
            weights.unsqueeze(-1),
            scene_features.unsqueeze(1).expand(-1, self.num_gaussians, -1)
        ], dim=-1)
        
        return scene_representation, gaussian_params 