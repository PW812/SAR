import torch
import torch.nn as nn
import torch.nn.functional as F

class MLN(nn.Module):
    ''' 
    from "https://github.com/exiawsh/StreamPETR"
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x
class ConvFuser(nn.Module):
    def __init__(self, in_channels, out_channels, verbose=False):
        super(ConvFuser, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.mlp(x.permute(0,2,1))
        return x
       
class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}', MLP(in_channels, hidden_unit))
            in_channels = hidden_unit*2

    def forward(self, pts_lane_feats):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        '''
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
   
        return x_max


class EgoQueryExtractor_V2(nn.Module):
    def __init__(self, embed_dim, bev_h, bev_w):
        
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        
        
        self.ego_pos_encoder = nn.Sequential(
            nn.Linear(2, 64),  
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        
        self.bev_roi_extractor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        
        self.risk_aware = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        
        
        # self.fusion_gate = nn.Sequential(
        #     nn.Linear(3 * embed_dim, 3),
        #     nn.Softmax(dim=-1)
        # )
        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 3)
        )

    def forward(self, bev_embed, navi_embed, agent_pos=None, ego_pos=None, ego_pos_emb=None):
        
        B, N, D = bev_embed.shape
        
       
        bev_spatial = bev_embed.view(B, self.bev_h, self.bev_w, D).permute(0, 3, 1, 2).contiguous()
        
        
        center_mask = torch.zeros(B, 1, self.bev_h, self.bev_w, device=bev_embed.device)
        center_y, center_x = self.bev_h // 2, self.bev_w // 2
        roi_size = min(self.bev_h, self.bev_w) // 4  
        
        center_mask[:, :, 
                   center_y-roi_size//2:center_y+roi_size//2, 
                   center_x-roi_size//2:center_x+roi_size//2] = 1.0
        
       
        bev_roi = bev_spatial * center_mask
        bev_roi = self.bev_roi_extractor(bev_roi)
        bev_feat = F.adaptive_avg_pool2d(bev_roi, 1).view(B, D)  # [B, D]
        
        
        if agent_pos is not None:
            
            distances = torch.norm(agent_pos - ego_pos, dim=-1)  # [B, num_agents]
            min_distances, _ = torch.min(distances, dim=1)  # [B]
            
            
            risk_feat = torch.zeros(B, D, device=bev_embed.device)
            
            risk_mask = min_distances < 0.3  #
            risk_value = 1.0 / (min_distances + 1e-6)  
            risk_feat[risk_mask] = risk_value[risk_mask].unsqueeze(1) * 0.1  
            
            
            bev_feat = self.risk_aware(torch.cat([bev_feat, risk_feat], dim=1))
        
       
        navi_feat = navi_embed.squeeze(1)  # [B, D]
        features = torch.stack([bev_feat, ego_pos_emb.squeeze(1), navi_feat], dim=1)  # [B, 3, D]
        
        
        combined = features.flatten(start_dim=1)  # [B, 3*D]
        gate_logits  = self.fusion_gate(combined)  # [B, 3]
        
        
        bev_bias = 0.4  
        adjusted_logits = gate_logits.clone()
        adjusted_logits[:,0] = gate_logits[:,0] + bev_bias
        gate_weights = F.softmax(adjusted_logits, dim=-1)  
        
        ego_query = (features * gate_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        
        return ego_query.unsqueeze(1)  # [B, 1, D]


class EgoQueryExtractor_V3(nn.Module):
    """
    Latent Ego Action (implicit):
    - Local context from ego-centric ROI in BEV
    - Navigation intent embedding
    - Implicit interaction readout from agent motion tokens (preferred) or agent positions (fallback)
    - Intent-aware gated fusion to produce z_ego (ego action token)
    """

    def __init__(self, embed_dim, bev_h, bev_w, roi_ratio=0.25):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dim = embed_dim

        # 1) Local ego ROI encoder (lightweight)
        self.bev_roi_extractor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Precompute a fixed ego-centric ROI mask (ego is centered in BEV frame)
        roi_size = int(min(bev_h, bev_w) * roi_ratio)
        roi_size = max(2, roi_size)
        mask = torch.zeros(1, 1, bev_h, bev_w)
        cy, cx = bev_h // 2, bev_w // 2
        y0, y1 = cy - roi_size // 2, cy + roi_size // 2
        x0, x1 = cx - roi_size // 2, cx + roi_size // 2
        mask[:, :, y0:y1, x0:x1] = 1.0
        self.register_buffer("ego_roi_mask", mask, persistent=False)

        # 2) Condition query for interaction readout
        self.cond_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # 3) Interaction readout attention (single-head, lightweight)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Optional fallback: embed agent positions to tokens (if motion tokens not provided)
        self.agent_pos_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # 4) Intent-aware gating fusion (logits -> softmax once)
        self.gate = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 3),
        )

        # 5) Output stabilization
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        bev_embed,          # [B, H*W, D] or [B, N, D] where N=H*W
        navi_embed,         # [B, 1, D] or [B, D]
        ego_pos_emb,        # [B, 1, D] or [B, D]  (your pipeline already has this)
        motion_tokens=None, # [B, K, D] preferred: high-confidence agent motion tokens
        agent_pos=None,     # [B, Na, 2] fallback if motion_tokens is None
    ):
        B, N, D = bev_embed.shape
        assert D == self.embed_dim, f"embed_dim mismatch: {D} vs {self.embed_dim}"
        assert N == self.bev_h * self.bev_w, f"bev tokens N should be H*W, got {N}"

        # ----- Local context f_ctx -----
        bev_spatial = bev_embed.view(B, self.bev_h, self.bev_w, D).permute(0, 3, 1, 2).contiguous()  # [B,D,H,W]
        bev_roi = bev_spatial * self.ego_roi_mask  # [B,D,H,W]
        bev_roi = self.bev_roi_extractor(bev_roi)  # [B,D,H,W]
        f_ctx = F.adaptive_avg_pool2d(bev_roi, 1).flatten(1)  # [B,D]

        # ----- Navigation intent f_navi -----
        if navi_embed.dim() == 3:
            f_navi = navi_embed.squeeze(1)  # [B,D]
        else:
            f_navi = navi_embed  # [B,D]

        # ----- Ego state embedding (no explicit physics) -----
        if ego_pos_emb.dim() == 3:
            f_ego = ego_pos_emb.squeeze(1)  # [B,D]
        else:
            f_ego = ego_pos_emb  # [B,D]

        # ----- Implicit interaction readout f_int -----
        # Condition query uses local context + navigation intent (you can also include f_ego if you want)
        q_cond = self.cond_proj(torch.cat([f_ctx, f_navi], dim=-1))  # [B,D]

        if motion_tokens is None:
            assert agent_pos is not None, "Provide motion_tokens [B,K,D] or agent_pos [B,Na,2] as fallback."
            # Fallback: treat embedded agent positions as tokens
            # This is still implicit (learned embedding), no distance thresholding
            M = self.agent_pos_embed(agent_pos)  # [B,Na,D]
        else:
            M = motion_tokens  # [B,K,D]

        # Scaled dot-product attention: A = softmax(QK^T / sqrt(D)), f_int = A V
        Q = self.W_q(q_cond).unsqueeze(1)        # [B,1,D]
        K = self.W_k(M)                         # [B,K,D]
        V = self.W_v(M)                         # [B,K,D]
        attn = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(D)  # [B,1,K]
        A = torch.softmax(attn, dim=-1)                               # [B,1,K]
        f_int = torch.matmul(A, V).squeeze(1)                         # [B,D]

        # ----- Intent-aware gated fusion -> z_ego -----
        gate_logits = self.gate(torch.cat([f_ctx, f_int, f_navi], dim=-1))  # [B,3]
        gate_w = torch.softmax(gate_logits, dim=-1)                         # [B,3]

        # Optionally include f_ego in the mixture; here we keep your original 3-way fusion
        z_ego = (
            gate_w[:, 0:1] * f_ctx +
            gate_w[:, 1:2] * f_int +
            gate_w[:, 2:3] * f_navi
        )
        z_ego = self.out_norm(z_ego + f_ego)  # residual ego-state bias, still no explicit physics

        return z_ego.unsqueeze(1)  # [B,1,D]
        
class EgoQueryExtractor_EAD_NoVel(nn.Module):
    def __init__(self, embed_dim, bev_h, bev_w, num_heads=8, sigma=0.8):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.sigma = sigma
        D = embed_dim

        self.bev_roi_extractor = nn.Sequential(
            nn.Conv2d(D, D, 3, padding=1), nn.ReLU(),
            nn.Conv2d(D, D, 3, padding=1), nn.ReLU()
        )

        # ego->agent cross-attention
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.risk_attn = nn.MultiheadAttention(D, num_heads=num_heads, batch_first=True)

        # 用距离/方位/置信度 生成“注意力先验门控”
        # [log(d), inv(d), gauss(d), cos, sin, score] -> scalar gate
        self.prior_mlp = nn.Sequential(
            nn.Linear(6, D), nn.ReLU(),
            nn.Linear(D, 1)
        )

        self.risk_post = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(), nn.LayerNorm(D)
        )

        # gate: logits only (修复你原先 softmax->softmax 的问题)
        self.fusion_logits = nn.Sequential(
            nn.Linear(3 * D, D), nn.ReLU(),
            nn.Linear(D, 3)
        )

    def forward(self, bev_embed, navi_embed, ego_pos_emb,
                agent_pos=None, agent_query=None, agent_mask=None, agent_score=None):
        B, N, D = bev_embed.shape

        # ---- BEV ROI ----
        bev_spatial = bev_embed.view(B, self.bev_h, self.bev_w, D).permute(0, 3, 1, 2).contiguous()
        mask = torch.zeros(B, 1, self.bev_h, self.bev_w, device=bev_embed.device, dtype=bev_embed.dtype)
        cy, cx = self.bev_h // 2, self.bev_w // 2
        r = min(self.bev_h, self.bev_w) // 4
        mask[:, :, cy-r//2:cy+r//2, cx-r//2:cx+r//2] = 1.0
        bev_roi = self.bev_roi_extractor(bev_spatial * mask)
        bev_feat = F.adaptive_avg_pool2d(bev_roi, 1).view(B, D)  # [B,D]

        # ---- Risk (no velocity) ----
        risk_feat = torch.zeros(B, D, device=bev_embed.device, dtype=bev_embed.dtype)

        if agent_pos is not None and agent_query is not None and agent_query.numel() > 0:
            # agent_mask: True=padding (ignore) ——与你的select_and_pad_query完全对齐
            if agent_mask is None:
                agent_mask = torch.zeros(agent_pos.shape[:2], device=bev_embed.device, dtype=torch.bool)

            # 置信度（可选）
            if agent_score is None:
                score = torch.ones(agent_pos.shape[:2], device=bev_embed.device, dtype=bev_embed.dtype)
            else:
                score = agent_score.to(bev_embed.dtype)  # [B,A']

            # ego 在你的代码里是(0,0)，但这里仍按通用写法
            ego_xy = torch.zeros(B, 1, 2, device=bev_embed.device, dtype=bev_embed.dtype)
            rel = agent_pos - ego_xy                        # [B,A',2]
            d = torch.norm(rel, dim=-1)                     # [B,A']
            eps = 1e-3

            # 连续、稳定的距离特征
            logd = torch.log(d + eps)
            invd = 1.0 / (d + 0.1)                          # 防爆
            gauss = torch.exp(-(d**2) / (2*(self.sigma**2) + eps))

            # 方位特征（相对方向）
            dir_vec = rel / (d.unsqueeze(-1) + eps)         # [B,A',2]
            cos_t, sin_t = dir_vec[..., 0], dir_vec[..., 1]

            prior_in = torch.stack([logd, invd, gauss, cos_t, sin_t, score], dim=-1)  # [B,A',6]
            prior = torch.sigmoid(self.prior_mlp(prior_in)).squeeze(-1)               # [B,A']

            # 对 padding 位置把 prior 置 0
            prior = prior.masked_fill(agent_mask, 0.0)      # True=pad -> 0

            # 用 prior 对 K/V 做门控（简洁、稳定、效果通常很好）
            gate = (1.0 + prior).unsqueeze(-1)              # [B,A',1]
            k = self.k_proj(agent_query) * gate
            v = self.v_proj(agent_query) * gate
            q = self.q_proj(ego_pos_emb)                    # [B,1,D]

            risk_ctx, _ = self.risk_attn(query=q, key=k, value=v, key_padding_mask=agent_mask)
            risk_feat = self.risk_post(risk_ctx.squeeze(1))  # [B,D]

        # ---- Fusion (BEV, Risk, Navi) ----
        navi_feat = navi_embed.squeeze(1)      # [B,D]
        features = torch.stack([bev_feat, risk_feat, navi_feat], dim=1)  # [B,3,D]
        logits = self.fusion_logits(torch.cat([bev_feat, risk_feat, navi_feat], dim=-1))  # [B,3]
        w = F.softmax(logits, dim=-1)          # [B,3]
        ego_query = (features * w.unsqueeze(-1)).sum(dim=1)  # [B,D]
        return ego_query.unsqueeze(1)

    
