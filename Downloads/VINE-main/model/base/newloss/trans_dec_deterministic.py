
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import kornia
# from model.MultiViewGAT_pseudo_nospace import MultiViewGraph
# from model.MultiViewGAT import MultiViewGraph
# from model.MultiViewGAT_pseudo import MultiViewGraph
# from model.MultiViewGAT_pseudo import MultiViewGraph
from model.MultiViewGAT_imagelevel import MultiViewGraph
from model.base.MultiHeadAttention import MaskMultiHeadAttention
# --- 加入这两行 ---
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import math, copy
import numpy as np


def get_gauss(mu, sigma):
    gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gauss


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        # 按行累加的，从上到下累积。
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 按列累加的，从左到右累积。
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t 的长度为 num_pos_feats（即每个位置编码的维度）
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 通过对 pos_x 和 pos_y 的奇数和偶数维度分别应用正弦和余弦函数，生成二维位置编码。
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        # 将 pos_y 和 pos_x 沿着通道维度拼接，形成最终的位置编码 pos。
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, attn = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, attn_drop_out=0.2,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop_out)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # 根据指定的 activation 参数来选择激活函数（如 ReLU 等）。
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        # 	重新初始化模型中的参数。
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 将位置编码（pos）与输入的特征（tensor）相加。位置编码通常用于保留输入序列的位置信息
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_key, memory_value,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     value_pos: Optional[Tensor] = None):
        # 注意力机制的前向传播。
        # q :tgt 是目标张量----output
        # k : memory_key----src_x_s_sam
        # v : memory_value---- src_x_s_sam
        #  memory_mask=self.processing_for_attn_mask(support_mask, self.args.spt_num_query),
        #  memory_key_padding_mask=None,
        #  pos=pos_x_s,          # → support feature 的位置编码
        #  query_pos=None        # → query feature 的位置编码（这里没有）
        tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                         key=self.with_pos_embed(memory_key, pos),
                                         value=self.with_pos_embed(memory_value, value_pos), attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn

    def forward_pre(self, tgt, memory_key, memory_value,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    value_pos: Optional[Tensor] = None):
        # 先对输入进行归一化处理
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=self.with_pos_embed(memory_value, value_pos), attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory_key, memory_value,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                value_pos: Optional[Tensor] = None):
        # 根据 normalize_before 的值选择调用 forward_pre 还是 forward_post 方法
        if self.normalize_before:
            return self.forward_pre(tgt, memory_key, memory_value, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, value_pos)
        return self.forward_post(tgt, memory_key, memory_value, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, value_pos)


class CrossAggregationLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # MaskMultiHeadAttention 层来进行跨注意力计算
        self.my_attn = MaskMultiHeadAttention(4, 256, dropout=0.5)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory_key, memory_value,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     value_pos: Optional[Tensor] = None,
                     add_attn=False):

        tgt2, attn = self.my_attn(self.with_pos_embed(tgt, query_pos).permute(1, 0, 2),
                                  self.with_pos_embed(memory_key, pos).permute(1, 0, 2),
                                  self.with_pos_embed(memory_value, value_pos).permute(1, 0, 2), mask=memory_mask,
                                  add_attn=add_attn)

        tgt2 = tgt2.permute(1, 0, 2)

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn

    def forward_pre(self, tgt, memory_key, memory_value,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    value_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory_key, pos),
                                   value=self.with_pos_embed(memory_value, value_pos), attn_mask=memory_mask,
                                   # 加入mask
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory_key, memory_value,
                add_attn=False,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                value_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory_key, memory_value, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, value_pos)
        return self.forward_post(tgt, memory_key, memory_value, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, value_pos, add_attn=add_attn)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def batch_distance_transform(mask: torch.Tensor, normalize=True):
    """
    计算二值掩码到前景的欧氏距离图。

    参数:
        mask: [B, 1, H, W]，前景为1，背景为0
        normalize: 是否将结果归一化到 [0, 1]
    返回:
        dist_map: [B, 1, H, W]，每个像素点到最近前景像素的距离
    """
    assert mask.dim() == 4 and mask.size(1) == 1, "Input must be [B, 1, H, W]"

    B, _, H, W = mask.shape
    device = mask.device

    # 建立网格坐标
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid = torch.stack((grid_y, grid_x), dim=0).float()  # [2, H, W]

    dist_maps = torch.zeros((B, 1, H, W), device=device)

    for b in range(B):
        # 取出当前 mask 的前景点坐标
        foreground = mask[b, 0] > 0.5
        if foreground.sum() == 0:
            # 没有前景，直接返回全 1
            dist_maps[b, 0] = torch.ones((H, W), device=device)
            continue

        yx_fg = torch.nonzero(foreground, as_tuple=False).float()  # [N, 2]

        # [2, H, W] → [2, H*W]
        grid_flat = grid.view(2, -1)  # [2, H*W]
        # 计算所有点到前景点的距离，找最小值
        dists = torch.cdist(grid_flat.T.unsqueeze(0), yx_fg.unsqueeze(0)).squeeze(0)  # [H*W, N]
        min_dist, _ = dists.min(dim=1)  # [H*W]
        dist_maps[b, 0] = min_dist.view(H, W)

    if normalize:
        # 归一化到 [0, 1]
        max_val = dist_maps.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        dist_maps = dist_maps / (max_val + 1e-6)

    return dist_maps

class SupportFeaturePurifier(torch.nn.Module):
    def __init__(self, in_channels, dist_embed_channels=1):
        super().__init__()
        self.gate_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.fusion_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels + dist_embed_channels, in_channels, kernel_size=1),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, f_s, support_mask):
        """
        f_s: 支持特征图 [B, C, H, W]
        support_mask: 支持掩码 [B, 1, h, w]（原始分辨率）
        返回：净化后的支持特征 f_s_final [B, C, H, W]
        """
        # 1a: 前景-背景对比增强
        m_s = F.interpolate(support_mask.float(), size=f_s.shape[-2:], mode='nearest')  # resize 掩码

        kernel = torch.ones(3, 3, device=m_s.device)
        m_s_dilated = kornia.morphology.dilation(m_s, kernel)
        m_bg_nearby = (m_s_dilated - m_s).clamp(min=0)

        epsilon = 1e-8
        p_fg = (f_s * m_s).sum(dim=(2, 3)) / (m_s.sum(dim=(2, 3)) + epsilon)
        p_bg = (f_s * m_bg_nearby).sum(dim=(2, 3)) / (m_bg_nearby.sum(dim=(2, 3)) + epsilon)
        p_contrast = (p_fg - p_bg).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        gate = torch.sigmoid(self.gate_conv(f_s))
        f_s_purified = f_s + gate * p_contrast  # 注入对比增强特征

        # 1b: 距离变换编码
        dist_map = self._compute_distance_transform(m_s)  # [B, 1, H, W]
        #自己实现的
        #dist_map = batch_distance_transform(m_s, normalize=True)  # [B, 1, H, W]
        # 高斯近似
        # dist_map = 1 - gaussian_blur(mask.float(), kernel_size=11)
        f_s_with_dist = torch.cat([f_s_purified, dist_map], dim=1)
        f_s_final = self.fusion_conv(f_s_with_dist)

        return f_s_final

    def _compute_distance_transform(self, binary_mask):
        """
        binary_mask: [B, 1, H, W] 取值为 0 或 1
        返回：距离图 [B, 1, H, W]，归一化到 [0, 1] 范围
        """
        import cv2
        dist_maps = []
        for b in range(binary_mask.shape[0]):
            mask_np = binary_mask[b, 0].cpu().numpy().astype('uint8')
            dist = cv2.distanceTransform(1 - mask_np, distanceType=cv2.DIST_L2, maskSize=5)
            dist = dist / (dist.max() + 1e-6)  # 归一化
            dist_maps.append(torch.from_numpy(dist).to(binary_mask.device).unsqueeze(0))
        return torch.stack(dist_maps, dim=0)  # [B, 1, H, W]

class LightweightSupportFeatureRefiner(nn.Module):
    def __init__(self, in_channels, use_distance=False):
        super().__init__()
        self.use_distance = use_distance

        # 自适应注意力引导
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)
        )

        fusion_channels = in_channels * 2 + (1 if use_distance else 0)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_s, supp_feat_bin, dist_map=None):
        """
        Args:
            f_s: [B, C, H, W] - 原始支持特征
            supp_feat_bin: [B, C, 1, 1] - 掩码池化后的类原型特征
            dist_map: [B, 1, H, W] (optional) - 前景距离图
        Returns:
            f_final: [B, C, H, W] - 融合后的支持特征
        """
        B, C, H, W = f_s.shape
        supp_feat_bin = supp_feat_bin.expand(-1, -1, H, W)  # [B, C, H, W]

        # 构造注意力图 A
        attn_input = torch.cat([f_s, supp_feat_bin], dim=1)  # [B, 2C, H, W]
        attn_score = torch.sigmoid(self.attn_conv(attn_input))  # [B, 1, H, W]

        f_attended = f_s * attn_score  # [B, C, H, W]

        # 特征融合
        fusion_inputs = [f_s, f_attended]
        if self.use_distance and dist_map is not None:
            fusion_inputs.append(dist_map)
        fusion_input = torch.cat(fusion_inputs, dim=1)  # [B, 2C(+1), H, W]

        f_fused = self.fusion_conv(fusion_input)
        f_final = f_fused + f_s  # 残差稳定性增强

        return f_final

class HybridSupportFeatureRefiner(nn.Module):
    def __init__(self, in_channels, use_distance=True):
        super().__init__()
        self.gate_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.use_distance = use_distance
        fusion_in_channels = in_channels * 3 + (1 if use_distance else 0)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_s, supp_feat_bin,support_mask):
        """
        f_s: [B, C, H, W] 支持特征
        support_mask: [B, 1, h, w] 原始掩码（与原图等大）
        """
        B, C, H, W = f_s.shape

        # 1. Resize mask
        m_s = F.interpolate(support_mask.float(), size=(H, W), mode='nearest')  # [B, 1, H, W]

        # 2. supp_feat_bin (原型特征)
        eps = 1e-8
        # supp_feat_bin = (f_s * m_s).sum(dim=(2, 3), keepdim=True) / (m_s.sum(dim=(2, 3), keepdim=True) + eps)  # [B, C, 1, 1]
        # supp_feat_bin = supp_feat_bin.expand(-1, -1, H, W)  # [B, C, H, W]
        # Step 3: Semantic attention map
        attn_map = torch.sigmoid(torch.sum(f_s * supp_feat_bin, dim=1, keepdim=True))  # [B, 1, H, W]
        f_s_weighted = f_s * attn_map  # [B, C, H, W]
        # Step 4: Distance map (optional)
        if self.use_distance:
            dist_map = batch_distance_transform(m_s)  # [B, 1, H, W]
            fusion_input = torch.cat([f_s, supp_feat_bin, f_s_weighted, dist_map], dim=1)  # [B, 3C+1, H, W]
        else:
            fusion_input = torch.cat([f_s, supp_feat_bin, f_s_weighted], dim=1)  # [B, 3C, H, W]

        f_s_weighted = f_s * attn_map  # [B, C, H, W]
        # Step 5: Fusion
        f_s_fused = self.fusion_conv(fusion_input)  # [B, C, H, W]
        # # 3. 对比向量
        # kernel = torch.ones(3, 3, device=m_s.device)
        # m_s_dilated = kornia.morphology.dilation(m_s, kernel)
        # m_bg_nearby = (m_s_dilated - m_s).clamp(min=0)
        #
        # p_fg = (f_s * m_s).sum(dim=(2, 3)) / (m_s.sum(dim=(2, 3)) + eps)
        # p_bg = (f_s * m_bg_nearby).sum(dim=(2, 3)) / (m_bg_nearby.sum(dim=(2, 3)) + eps)
        # p_contrast = (p_fg - p_bg).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, C, H, W]
        #
        # # 4. Gate control
        # gate = torch.sigmoid(self.gate_conv(f_s))  # [B, C, H, W]
        # f_s_purified = f_s + gate * p_contrast
        #
        # # 5. 距离图
        # dist_map = batch_distance_transform(m_s)  # [B, 1, H, W]
        #
        # # 6. 融合特征: f_s_purified + supp_feat_bin + dist_map
        # fusion_input = torch.cat([f_s_purified, supp_feat_bin, dist_map], dim=1)  # [B, 2C+1, H, W]
        # f_s_final = self.fusion_conv(fusion_input)  # [B, C, H, W]

        return f_s_fused
def visualize_pca(feat_q, feat_s, img_size=(512, 512)):
    """
    将 Query 和 Support 的特征联合进行 PCA 降维，映射到 RGB 空间
    feat_q, feat_s: [B, 256, 64, 64]
    """
    # 1. 准备数据：取 Batch 0
    fq = feat_q[0].detach().cpu().numpy().transpose(1, 2, 0) # [64, 64, 256]
    fs = feat_s[0].detach().cpu().numpy().transpose(1, 2, 0) # [64, 64, 256]
    H, W, C = fq.shape
    
    # 2. 拉平并拼接 (为了保证颜色一致性，必须一起做 PCA)
    flat_q = fq.reshape(-1, C) # [4096, 256]
    flat_s = fs.reshape(-1, C) # [4096, 256]
    combined = np.concatenate([flat_q, flat_s], axis=0)
    
    # 3. PCA 降维到 3 (RGB)
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(combined)
    
    # 4. 归一化到 0-1 之间以便显示
    # 使用 Min-Max Scaling
    m_min = transformed.min(axis=0)
    m_max = transformed.max(axis=0)
    transformed = (transformed - m_min) / (m_max - m_min + 1e-8)
    
    # 5. 还原形状
    pca_q = transformed[:H*W, :].reshape(H, W, 3)
    pca_s = transformed[H*W:, :].reshape(H, W, 3)
    
    # 6. Resize 到原图大小
    pca_q_img = cv2.resize(pca_q, img_size)
    pca_s_img = cv2.resize(pca_s, img_size)
    
    return pca_q_img, pca_s_img
class transformer_decoder(nn.Module):
    """ Transformer decoder to get point query"""

    def __init__(self, args, num_queries, fea_dim, hidden_dim, dim_feedforward, nheads=4, num_layers=3, pre_norm=False):
        super().__init__()
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.args = args
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = num_layers

        self.transformer_cross_attention_layers_spt = nn.ModuleList()
        self.transformer_cross_attention_layers_qry = nn.ModuleList()
        self.transformer_cross_attention_layers_1 = nn.ModuleList()
        self.transformer_self_attention_layers_1 = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_cross_attention_layers_spt.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    attn_drop_out=args.attn_drop_out,
                    normalize_before=pre_norm,
                )
            )
            self.transformer_cross_attention_layers_qry.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    attn_drop_out=args.attn_drop_out,
                    normalize_before=pre_norm,
                )
            )

        self.transformer_cross_attention_layers_1.append(
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        )

        self.transformer_self_attention_layers_1.append(
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        )

        self.merge_sam_and_mask = nn.Sequential(
            nn.Conv2d(hidden_dim * 2 + 1, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        self.merge = nn.Sequential(nn.Conv2d(hidden_dim * 2 + 1, hidden_dim, kernel_size=1, padding=0, bias=False),
                                   nn.ReLU(inplace=True)
                                   )
        # 新融合特征
        # self.refiner_sam = HybridSupportFeatureRefiner(in_channels=256)
        # self.refiner_res = HybridSupportFeatureRefiner(in_channels=256)
        # self.refiner_sam =  LightweightSupportFeatureRefiner(in_channels=256)
        # self.refiner_res =  LightweightSupportFeatureRefiner(in_channels=256)
        # self.refiner_sam_v2 = SupportFeaturePurifier(in_channels=256)
        # self.refiner_res_v2 = SupportFeaturePurifier(in_channels=256)
        # 加入多视角统一原型
        self.MultiviewGraph = MultiViewGraph(256, hidden_dim, hidden_dim, num_heads_space=1, num_heads_view=1, k=4)
        self.downsample_for_transformer = nn.Conv2d(fea_dim, hidden_dim, kernel_size=1, bias=False)

        self.num_queries = num_queries
        self.spt_protos = nn.Embedding(self.args.spt_num_query, hidden_dim)
        self.qry_protos = nn.Embedding(num_queries, hidden_dim)

    def forward(self, x_q, x_s, x_q_sam, x_s_sam,
                support_mask, spt_prototype, qry_res_feat_4, pseudo_mask,
                supp_feat_bin, supp_feat_bin_sam,
                supp_feat_all,
                batch_id_all,
                batch_id,deter_dist=None):
        '''
          x_q:resnet50 layer2和layer3进行拼接后downsample
          x_s:resnet50 layer2和layer3进行拼接后downsample
          x_q_sam:sam_encoder
          x_s_sam:sam_encoder
          support_mask: support gt
          spt_prototype :resnet_layer4经过MF得到的聚合结果，
          qry_res_feat_4:resnet_layer4
          pseudo_mask：用query_layer4和support gt生成的query掩码
          supp_feat_bin：resnet50 support和gt经过MF得到的原型[8,256,64,64]
          supp_feat_bin_sam:sam_encoder support 和gt经过MF得到的原型[8,256,64,64]
        '''
        bs, C, H, W = x_q.shape
        # x_q [8,256,64,64]
        # x_s [8,256,64,64]
        # x_s_sam[8,256,64,64]
        # support_mask[8,1,64,64]
        # ,spt_prototype[8,2048,1,1],
        # supp_feat_bin[8,256,64,64]
        spt_prototype = spt_prototype.squeeze(-1).squeeze(-1).unsqueeze(1)  # [8,1,2048]
        src_x_q = None
        pos_x_s = self.pe_layer(x_s, None).flatten(2).to(x_s.device).permute(2, 0, 1)   # [4096,4,256]
        src_x_s = None
        src_x_q_sam = x_q_sam.flatten(2).permute(2, 0, 1)
        # [8，256，64，64]---[8,256,64,64]
        # multi_view_x_s = self.MultiviewGraph(x_s, batch_id)
        multi_view_x_s = self.MultiviewGraph(supp_feat_all,batch_id_all, batch_id)
        # 炼丹
        # multi_view_x_s = multi_view_x_s + 0.3 * x_s
        # pos_x_s = multi_view_x_s.flatten(2).to(x_s.device).permute(2, 0, 1)
        src_x_q = None
        # pos_x_s = multi_view_x_s.flatten(2).to(x_s.device).permute(2, 0, 1)
        support_sam_c_attn = []
        # [spt_num_query, hidden_dim]---- [spt_num_query, bs, hidden_dim]
        # 支持集的cross attention 过程 ---[50,8,256]
        output = self.spt_protos.weight.unsqueeze(1).repeat(1, bs, 1)
        # deterministic mask
        P_disc_res,P_fg_res,P_bg_res = deter_dist["res"]["disc"],deter_dist["res"]["fg"],deter_dist["res"]["bg"]
        P_disc_res = P_disc_res.unsqueeze(1)
        P_disc_sam,P_fg_sam,P_bg_sam = deter_dist["sam"]["disc"],deter_dist["sam"]["fg"],deter_dist["sam"]["bg"]
        P_disc_sam = P_disc_sam.unsqueeze(1)
        for i in range(self.num_layers):
            # x_s_sam[8,256,64,64]，supp_feat_bin_sam[8,256,64,64],support_mask[8,1,64,64]
            """sam feature updates"""
            # x_s_sam_merged = self.refiner_sam(x_s_sam, supp_feat_bin_sam.detach(), (support_mask * 10).detach())
            x_s_sam_merged = self.merge_sam_and_mask(
                torch.cat(
                    [x_s_sam,# b,256,64,64
                     supp_feat_bin_sam,
                     support_mask * 10],
                dim=1).float().contiguous()
            )
            # [8，256，64，64]---[4096,8,256]
            src_x_s_sam = x_s_sam_merged.flatten(2).permute(2, 0, 1)
            # [8,256,64,64]
            if i != self.num_layers - 1:
                # mask cross attention
                output, s_c_attn_map = self.transformer_cross_attention_layers_spt[i](
                    output, src_x_s_sam, src_x_s_sam,
                    # [bs*num_heads, num_queries, h*w] 的布尔张量
                    memory_mask=self.processing_for_attn_mask(support_mask, self.args.spt_num_query),
                    memory_key_padding_mask=None,
                    pos=pos_x_s, query_pos=None
                )
                support_sam_c_attn.append(s_c_attn_map)

            else:
                # soft_mask = support_mask.detach()  # 或者加入 learnable alpha
                # x_s= soft_mask * multi_view_x_s + (1 - soft_mask) * supp_feat_bin
                x_s = self.merge(torch.cat([multi_view_x_s,supp_feat_bin.detach(),(support_mask*10).detach()], dim=1))
                # x_s = self.refiner_res(x_s, supp_feat_bin, support_mask * 10)
                # x_s = self.refiner_res(multi_view_x_s, supp_feat_bin.detach(),(support_mask * 10).detach())
                src_x_s = x_s.flatten(2).permute(2, 0, 1)  # [4096,8,256]
                output, s_c_attn_map = self.transformer_cross_attention_layers_spt[i](
                    output, src_x_s_sam, src_x_s,
                    memory_mask=self.processing_for_attn_mask(support_mask, self.args.spt_num_query),
                    memory_key_padding_mask=None,
                    pos=pos_x_s, query_pos=None
                )
                support_sam_c_attn.append(s_c_attn_map)

        # 计算query video
        spt_protos = output  # [8,1,2048]

        multi_view_x_q = self.MultiviewGraph(x_q,batch_id, batch_id)
        # multi_view_x_q = multi_view_x_q + 0.3 * x_q
        pos_x_q = self.pe_layer(x_q, None).flatten(2).to(x_q.device).permute(2, 0, 1)  # [4096,8,256]
        # pos_x_q = multi_view_x_q.flatten(2).to(x_q.device).permute(2, 0, 1)  # [4096,8,256]
        # 你可以根据 batch idx 或者类别名来命名文件
        # --- B. 设置保存路径 ---
        pseudo_mask_loss = []
        query_sam_c_attn = []
        pseudo_mask_vis = []
        multiview_proto_loss = []
        pseudo_mask_vis.append(pseudo_mask.float())
        pseudo_mask_vis.append((pseudo_mask > 0.5).float())
        # 查询集的cross attention过程
        output = self.qry_protos.weight.unsqueeze(1).repeat(1, bs, 1)
        for i in range(self.num_layers):

            pseudo_mask_naive = pseudo_mask
            """sam feature updates"""
            if self.args.concat_th:
                pseudo_mask_for_concat = (pseudo_mask > 0.5).float()
            else:
                pseudo_mask_for_concat = pseudo_mask
            # 在query
            soft_mask = pseudo_mask_naive.detach()  # 或者加入 learnable alpha
            # x_q_sam_merged = soft_mask * x_q_sam + (1 - soft_mask) * supp_feat_bin_sam
            # print(multi_view_x_s.shape)
            # print(supp_feat_bin.shape)
            # print(( P_disc_sam * 2).shape)
            # print((pseudo_mask_naive * 10+ P_disc_sam*2).shape)
            x_q_sam_merged = self.merge_sam_and_mask(torch.cat([x_q_sam, supp_feat_bin_sam, (pseudo_mask_naive * 10+ P_disc_sam*2)], dim=1))
            # x_q_sam_merged = self.refiner_sam(x_q_sam, supp_feat_bin.detach(), (pseudo_mask_naive * 10).detach())
            src_x_q_sam = x_q_sam_merged.flatten(2).permute(2, 0, 1)
            # 在每一层（除了最后一层），进行跨注意力操作。
            if i != self.num_layers - 1:
                # 生成新的注意力权重和更新后的 output。
                output, q_c_attn_map = self.transformer_cross_attention_layers_qry[i](
                    output, src_x_q_sam, src_x_q_sam,
                    # 用于处理支持图像掩码，以确保注意力机制只关注有效区域。
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=pos_x_q, query_pos=None, value_pos=None
                )
                # 在每层结束后，通过注意力图（q_c_attn_map）生成伪标签（pseudo mask）
                query_sam_c_attn.append(q_c_attn_map)

                """pseudo mask"""
                tau = 0.5  # 或根据实验调整到 0.7/1.0
                q_c_attn_map = F.gumbel_softmax(q_c_attn_map / tau, dim=1, hard=False)  # [B, C, N]
                pseudo_mask = q_c_attn_map.sum(dim=1, keepdim=True).reshape(bs, 1, 64, 64)  # [B, 1, 64, 64]
                # min-max

                # q_c_attn_map = (q_c_attn_map - q_c_attn_map.min(-1, keepdim=True)[0]) / (
                #         q_c_attn_map.max(-1, keepdim=True)[0] - q_c_attn_map.min(-1, keepdim=True)[0] + 1e-9)
                #
                # # mask merge
                # cur_pseudo_mask = q_c_attn_map.max(1, keepdim=True)[0]
                # pseudo_mask = cur_pseudo_mask.reshape(bs, 1, 64, 64)

                pseudo_mask_for_loss = pseudo_mask
                # 并计算损失。
                pseudo_mask_loss.append(pseudo_mask_for_loss)
                pseudo_mask_vis.append(pseudo_mask)
                pseudo_mask_vis.append((pseudo_mask > 0.5).float())

            else:
                # 用于处理支持图像掩码，以确保注意力机制只关注有效区域。
                """sam feature updates"""
                if self.args.concat_th:
                    pseudo_mask_for_concat = (pseudo_mask > 0.5).float()
                else:
                    pseudo_mask_for_concat = pseudo_mask
                # 换成多视角的值
                # 模态不对齐
                # soft_mask = pseudo_mask_for_concat.detach()  # 或者加入 learnable alpha
                # x_q = soft_mask * multi_view_x_q + (1 - soft_mask) * supp_feat_bin
                x_q = self.merge(torch.cat([multi_view_x_q, supp_feat_bin, pseudo_mask_for_concat * 10+P_disc_res*2], dim=1))
                # x_q = self.refiner_res(x_q, supp_feat_bin.detach(),(pseudo_mask_for_concat * 10).detach())
                src_x_q = x_q.flatten(2).permute(2, 0, 1)
                # 合并与跨注意力计算
                output, q_c_attn_map = self.transformer_cross_attention_layers_qry[i](
                    output, src_x_q_sam, src_x_q,
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=pos_x_q, query_pos=None, value_pos=None
                )
                query_sam_c_attn.append(q_c_attn_map)
        proto_s = F.adaptive_avg_pool2d(multi_view_x_s, 1).squeeze(-1).squeeze(-1)  # [B, C]
        proto_q = F.adaptive_avg_pool2d(multi_view_x_q, 1).squeeze(-1).squeeze(-1)  # [B, C]
        vis_q = F.adaptive_avg_pool2d(x_q, 1).squeeze(-1).squeeze(-1)  # [B, C]
        # 用于查询特征
        # 这里支持和查询的cross attention 全部结束
        qry_protos = output
        P_disc_res_list = []
        P_disc_res_list.append(P_disc_res.float())
        P_disc_res_list.append((P_disc_res > 0.5).float())
        for i in range(1):
            # 进行支持集和查询集原型的cross attention，计算出promps
            output, atten_layer_1 = self.transformer_cross_attention_layers_1[i](
                spt_protos, qry_protos, qry_protos,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=None, value_pos=None
            )
            # prompts自己进行self-attention
            output, _ = self.transformer_self_attention_layers_1[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )
            # 更新后的 output，并根据需要返回注意力图（s_c_attn_map, q_c_attn_map）和伪标签（pseudo_mask_vis）以及计算的损失
        return output.permute(1, 0, 2), [s_c_attn_map, q_c_attn_map, pseudo_mask_vis, (q_c_attn_map).float(),P_disc_res_list], \
            {'pseudo_mask_loss': pseudo_mask_loss, \
             'query_sam_c_attn': query_sam_c_attn, 'support_sam_c_attn': support_sam_c_attn,
             'proto_q': proto_q,
             'proto_s': proto_s,
             'cross_atten': atten_layer_1
             },vis_q

    def processing_for_attn_mask(self, mask, num, empty_check=False):
        # 将空间维度转换为序列维度
        mask = mask.flatten(2)
        # check empty pseudo mask
        if empty_check:
            # 如果某个位置为空，则通过将 empty_mask 添加到 mask 中来处理空的掩码。
            empty_mask = (mask.sum(-1, keepdim=True) == 0.).float()
            mask = mask + empty_mask * 1.0

        # arrange
        # [bs, 1, 1, 1, h*w]---- [bs*num_heads, num, h*w]
        mask = mask.unsqueeze(2).unsqueeze(2).repeat(1, 1, self.num_heads, num, 1).flatten(start_dim=0, end_dim=2)
        # [bs*num_heads, num, h*w] 的布尔张量
        mask = mask == 0.
        return mask