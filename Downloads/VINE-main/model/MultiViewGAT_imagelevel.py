import torch
import torch.nn as nn
# You will need PyTorch Geometric for GATConv and knn_graph
from torch_geometric.nn import GATConv, knn_graph
from collections import defaultdict
import torch.nn.functional as F
def group_by_class(batch_id):
    """
    将 batch 内的样本按类别进行分组。
    :param batch_id: [batch_size]，每个样本的类别 ID
    :return: class_groups {类别ID: [图像索引列表]}，num_views_per_class {类别ID: 该类别的视角数量}
    """
    class_groups = defaultdict(list)
    for i, cls in enumerate(batch_id.tolist()):
        class_groups[cls].append(i)
    num_views_per_class = {cls: len(img_list) for cls, img_list in class_groups.items()}
    return class_groups, num_views_per_class

def augment_single_feature(x: torch.Tensor, num_aug=2, angle_range=15, scale_range=(0.9, 1.1)):
    """
    对单个特征图 [C, H, W] 做仿射增强（Affine Transform），返回 [num_aug, C, H, W]
    num_aug 构造几个伪多视角（默认2个）。
    angle_range: 最大旋转角度 ±15°。
    scale_range: 缩放因子范围（如 [0.9, 1.1] 表示最多缩小10%，或放大10%）。
    """
    C, H, W = x.shape
    device = x.device
    aug_list = []
    for _ in range(num_aug):
        angle = (torch.rand(1).item() * 2 - 1) * angle_range  # [-angle_range, +angle_range]
        scale = torch.empty(1).uniform_(*scale_range).item()  # 在 [0.9, 1.1] 之间随机采样缩放因子。
        theta = torch.tensor([
            [scale * torch.cos(torch.deg2rad(torch.tensor(angle))),
             -scale * torch.sin(torch.deg2rad(torch.tensor(angle))),
             0.0],
            [scale * torch.sin(torch.deg2rad(torch.tensor(angle))),
             scale * torch.cos(torch.deg2rad(torch.tensor(angle))),
             0.0]
        ], dtype=torch.float, device=device).unsqueeze(0)  # [1, 2, 3]
        # 坐标网格 grid: [1, H, W, 2]，表示新图像中每个像素对应原图中的浮点坐标（x,y）；
        grid = F.affine_grid(theta, size=(1, C, H, W), align_corners=False) #[1,64,64,2]
        # grid_sample 会用 grid 对原图做双线性插值采样，生成旋转缩放后的图；
        x_aug = F.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)  # [C, H, W]
        # 把每次增强的 [C, H, W] 特征存起来。
        aug_list.append(x_aug)

    return torch.stack(aug_list)  # [num_aug, C, H, W]

def augment_single_class_inplace(x, batch_id, class_groups, cls, num_aug=2):
    """
    如果该类别只有一个视角，对该样本进行仿射增强，并插入 x, batch_id, class_groups
    :param x: Tensor, shape [B, C, H, W]
    :param batch_id: Tensor, shape [B]
    :param class_groups: dict[int, List[int]]
    :param cls: 当前类别 ID
    :return: x, batch_id, class_groups
    """
    if len(class_groups[cls]) != 1:
        return x, batch_id, class_groups  # 非单视角，跳过

    idx = class_groups[cls][0]
    x_single = x[idx]  # [C, H, W]
    x_aug = augment_single_feature(x_single, num_aug=num_aug)  # [num_aug, C, H, W]

    # 拼接到特征与标签
    x_new = torch.cat([x, x_aug], dim=0)  # [B + num_aug, C, H, W]
    new_ids = batch_id.new_full((num_aug,), batch_id[idx].item())  # [num_aug]
    batch_id_new = torch.cat([batch_id, new_ids], dim=0)

    # 更新 class_groups
    start_idx = x.shape[0]  # 注意：还没拼上时的 B
    new_indices = list(range(start_idx, start_idx + num_aug))
    class_groups[cls].extend(new_indices)

    return x_new, batch_id_new, class_groups

class MultiViewGraph(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads_space=2, num_heads_view=1, k=9):
        super(MultiViewGraph, self).__init__()
        self.k = k

        # 空间注意力 GAT
        self.gat_space = GATConv(in_channels=in_dim, out_channels=hidden_dim, heads=num_heads_space, concat=True)
        # 视角注意力 GAT
        self.gat_view = GATConv(in_channels=hidden_dim * num_heads_space, out_channels=out_dim, heads=num_heads_view,
                                concat=True)
    def forward(self, x, batch_id_all, batch_id):
        #supp_feat_all,batch_id_all, batch_id
        """
        :param x: [batch_size, feature_dim, height, width]，输入图像特征XQW
        :param batch_id: [batch_size]，记录 batch 内每个样本的类别 ID
        :return: [num_groups, num_patches, max_views, out_dim]，多视角 GAT 计算后的特征
        """
        original_batch_size =batch_id.shape[0]
        batch_size, feat_dim, H, W = x.shape  # 形状：[8,256,64,64]
        num_patches = H * W  # 计算每张图像的 patch 数量 = 4096
        # 空间 GAT
        node_features = x[:original_batch_size].permute(0, 2, 3, 1).reshape(-1, feat_dim)
        # 构造 batch_index，记录每个 patch 来自哪个样本
        batch_index = torch.arange(original_batch_size, device=x.device).repeat_interleave(
            int(num_patches) if not torch.is_tensor(num_patches) else num_patches.to(x.device)
        )

        coords = torch.stack(torch.meshgrid(torch.arange(H, device=x.device),
                                            torch.arange(W, device=x.device), indexing='ij'), dim=-1)
        coords_all = coords.view(-1, 2).float().repeat(original_batch_size, 1)
        coords_norm = coords_all / coords_all.max()
        
        edge_index_space = knn_graph(coords_norm, k=self.k, batch=batch_index, loop=False)  # 形状：[2, E_space]
        x_space = self.gat_space(node_features, edge_index_space)  # 形状：[32768, hidden_dim * num_heads_space]
        #--------------------------------------------------------------------
        # Image-level 表征（仅原图）
        image_embeddings = torch.stack([
            x_space[i * num_patches:(i + 1) * num_patches].mean(dim=0)
            for i in range(original_batch_size)
        ])#[batch_size,256]
        # 为增强图分配其原图表征
        image_embeddings_all = []
        global_augment_map = {}
        for i in range(batch_size):
            if i < original_batch_size:
                image_embeddings_all.append(image_embeddings[i])
            else:
                src = global_augment_map.get(i, i % original_batch_size)
                image_embeddings_all.append(image_embeddings[src])
        image_embeddings = torch.stack(image_embeddings_all)  # [B_all, C]

        def is_augmented(idx):
            return idx >= original_batch_size
        # 构建视角图
        class_groups, _ = group_by_class(batch_id_all)
        # 遍历 batch 内所有类别
        edge_index_view_list = []
        for cls, images_with_cls in class_groups.items():
            for i in range(len(images_with_cls)):
                for j in range(i + 1, len(images_with_cls)):  # 确保 i != j
                    img_i = images_with_cls[i]
                    img_j = images_with_cls[j]
                    # 保留主图之间的边，以及主图 ↔ 增强图的边
                    if is_augmented(img_i) and is_augmented(img_j):
                        continue  # 不要增强图之间互连
                    # 添加双向边
                    # 对每个类别内部图像进行完全连接，即每对视角样本之间建边；
                    edge_index_view_list.append(torch.tensor([img_i, img_j], device=x.device))
                    edge_index_view_list.append(torch.tensor([img_j, img_i], device=x.device))
        edge_index_view = (torch.cat([e.unsqueeze(1) for e in edge_index_view_list], dim=1)
                           if edge_index_view_list else
                           torch.empty((2, 0), dtype=torch.long, device=x.device))
         # [8,256] 类别原型
        # 执行图像级的 GAT
        image_features_view = self.gat_view(image_embeddings, edge_index_view)  # [batch_size, out_dim]
        alpha = 0.5  # 权重超参数，控制融合程度
        # 7️⃣ 还原 `image_features_view` 到 `patch` 级别
        B, C = original_batch_size, image_features_view.size(1)
        x_patch = x_space.view(B, num_patches, C)
        image_feat = image_features_view[:B].unsqueeze(1).expand(B, num_patches, C)
        x_view = (alpha * x_patch + (1 - alpha) * image_feat).reshape(-1, C)
        return  x_view.view(B, C, H, W)  # 直接返回
