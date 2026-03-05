r""" Visual Prompt Encoder of VRP-SAM """
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends.cudnn import deterministic
from sklearn.decomposition import PCA
import model.base.resnet as models
import model.base.vgg as vgg_models
from torch.nn import BatchNorm2d as BatchNorm
from common.utils import get_stroke_preset, get_random_points_from_mask, get_mask_by_input_strokes
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
import numpy as np
import torchvision.transforms.functional as TF
from model.base.newloss.trans_dec_deterministic import transformer_decoder

# copy from SEEM
def get_bounding_boxes(mask):
    """
    Returns:
        Boxes: tight bounding boxes around bitmasks.
        If a mask is empty, it's bounding box will be all zero.
    """
    boxes = torch.zeros(mask.shape[0], 4, dtype=torch.float32).to(mask.device)
    box_mask = torch.zeros_like(mask).to(mask.device)
    x_any = torch.any(mask, dim=1)
    y_any = torch.any(mask, dim=2)
    for idx in range(mask.shape[0]):
        x = torch.where(x_any[idx, :])[0].int()
        y = torch.where(y_any[idx, :])[0].int()
        if len(x) > 0 and len(y) > 0:
            boxes[idx, :] = torch.as_tensor(
                [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
            )
            x1, y1, x2, y2 = x[0], y[0], x[-1] + 1, y[-1] + 1

            box_mask[idx, y1:y2, x1:x2] = 1
    return boxes, box_mask

def get_point_mask(mask, training, max_points=20):
    """
    Returns:
        Point_mask: random 20 point for train and test.
        If a mask is empty, it's Point_mask will be all zero.
    """
    max_points = min(max_points, mask.sum().item())  # 每个样本在掩码中会标记随机选取的点。如果掩码为空（没有目标），输出的掩码会是全零的。
    if training:
        num_points = random.Random().randint(1, max_points)  # 随机选择一个点的数量，范围从 1 到 max_points 之间。
    else:
        num_points = max_points
    b, h, w = mask.shape
    point_masks = []

    for idx in range(b):
        view_mask = mask[idx].view(-1)
        non_zero_idx = view_mask.nonzero()[:, 0]  # get non-zero index of mask
        selected_idx = torch.randperm(len(non_zero_idx))[:num_points]  # select id
        non_zero_idx = non_zero_idx[selected_idx]  # select non-zero index
        rand_mask = torch.zeros(view_mask.shape).to(mask.device)  # init rand mask
        rand_mask[non_zero_idx] = 1  # get one place to zero
        point_masks.append(rand_mask.reshape(h, w).unsqueeze(0))
    return torch.cat(point_masks, 0)

def get_scribble_mask(mask, training, stroke_preset=['rand_curve', 'rand_curve_small'], stroke_prob=[0.5, 0.5]):
    """
    Returns:
        Scribble_mask: random 20 point for train and test.
        If a mask is empty, it's Scribble_mask will be all zero.
    """
    if training:
        stroke_preset_name = random.Random().choices(stroke_preset, weights=stroke_prob, k=1)[0]
        nStroke = random.Random().randint(1, min(20, mask.sum().item()))
    else:
        stroke_preset_name = random.Random(321).choices(stroke_preset, weights=stroke_prob, k=1)[0]
        nStroke = random.Random(321).randint(1, min(20, mask.sum().item()))
    preset = get_stroke_preset(stroke_preset_name)

    b, h, w = mask.shape

    scribble_masks = []
    for idx in range(b):
        points = get_random_points_from_mask(mask[idx].bool(), n=nStroke)
        rand_mask = get_mask_by_input_strokes(init_points=points, imageWidth=w, imageHeight=h,
                                              nStroke=min(nStroke, len(points)), **preset)
        rand_mask = (~torch.from_numpy(rand_mask)) * mask[idx].bool().cpu()
        scribble_masks.append(rand_mask.float().unsqueeze(0))
    return torch.cat(scribble_masks, 0).to(mask.device)

def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs, targets = inputs.flatten(1), targets.flatten(1)
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1 + 1e-9)
    return loss.sum() / num_masks

def collect_augmented_views_N(support_img, batch_id, support_mask, N=3):
    from collections import Counter
    from kornia.geometry.transform import get_perspective_transform, warp_perspective

    class_counter = Counter(batch_id.tolist())
    single_view_classes = {cls for cls, count in class_counter.items() if count == 1}

    aug_img_list = []
    aug_mask_list = []
    aug_class_id_list = []

    for idx in range(support_img.size(0)):
        cls = batch_id[idx].item()

        if cls not in single_view_classes:
            continue

        img = support_img[idx].unsqueeze(0)    # [1, C, H, W]
        mask = support_mask[idx].unsqueeze(0)  # [1, H, W]

        H_img = img.size(2)
        W_img = img.size(3)

        # 对每个单视角 sample 生成 N 个增强视角
        for _ in range(N):

            # 原图四点
            src_pts = torch.tensor(
                [[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]],
                device=img.device
            )

            # 控制扰动幅度：0.05~0.15 之间随机
            perturb = (0.05 + 0.10 * torch.rand_like(src_pts)) * (torch.randn_like(src_pts).sign())
            dst_pts = src_pts + perturb

            H_mat = get_perspective_transform(src_pts, dst_pts)

            warped_img = warp_perspective(img, H_mat, dsize=(H_img, W_img)).squeeze(0)
            warped_mask = warp_perspective(mask.float().unsqueeze(1), H_mat, dsize=(H_img, W_img)).squeeze(0)

            # mask二值化
            warped_mask = warped_mask > 0.5

            aug_img_list.append(warped_img)
            aug_mask_list.append(warped_mask)
            aug_class_id_list.append(cls)

    if len(aug_img_list) == 0:
        return None, None, None

    aug_imgs = torch.stack(aug_img_list, dim=0)
    aug_masks = torch.stack(aug_mask_list, dim=0)
    aug_ids = torch.tensor(aug_class_id_list, device=batch_id.device)

    return aug_imgs, aug_masks, aug_ids

def collect_augmented_views(support_img, batch_id, support_mask):
    from collections import Counter
    from kornia.geometry.transform import get_perspective_transform, warp_perspective

    class_counter = Counter(batch_id.tolist())
    single_view_classes = {cls for cls, count in class_counter.items() if count == 1}
    aug_img_list = []
    aug_mask_list = []
    aug_class_id_list = []
    for idx in range(support_img.size(0)):
        cls = batch_id[idx].item()
        if cls in single_view_classes:
            img = support_img[idx].unsqueeze(0)
            mask = support_mask[idx].unsqueeze(0)
            src_pts = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]], device=img.device)
            max_offset = 0.001
            offset = (torch.rand_like(src_pts) - 0.5) * 2 * max_offset
            dst_pts = torch.clamp(src_pts + offset, 0.0, 1.0)
            H = get_perspective_transform(src_pts, dst_pts)
            warped_img = warp_perspective(img, H, dsize=(img.size(2), img.size(3))).squeeze(0)
            warped_mask = warp_perspective(mask.float().unsqueeze(1), H, dsize=(mask.size(1), mask.size(2))).squeeze(0)
            def denormalize(tensor, mean, std):
                mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
                std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
                return tensor * std + mean
            aug_img_list.append(warped_img)
            aug_mask_list.append(warped_mask > 0.5)
            aug_class_id_list.append(cls)
    if len(aug_img_list) == 0:
        return None, None, None
    aug_imgs = torch.stack(aug_img_list, dim=0)
    aug_masks = torch.stack(aug_mask_list, dim=0)
    aug_ids = torch.tensor(aug_class_id_list, device=batch_id.device)

    return aug_imgs, aug_masks, aug_ids

class SpatialAwareMaskEnhancer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 输入为 support_mask + image_feature → 逐像素预测 alpha_map
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, features, support_mask, se_mask=None):
        """
        Args:
            features: [B, C, H, W] support特征
            support_mask: [B, 1, H, W] GT掩码，0为背景，1为前景
            se_mask: 保留接口兼容性（可选结构增强mask）

        Returns:
            mask: [B, 1, H, W] 融合后的 soft mask
        """
        # 背景掩码
        background_mask = 1 - support_mask  # [B, 1, H, W]

        # 拼接support_mask + 特征作为输入预测alpha_map
        x = torch.cat([support_mask, features], dim=1)  # [B, C+1, H, W]
        alpha_map = self.alpha_conv(x)  # [B, 1, H, W] ∈ (0,1)

        # 融合前景与动态背景权重
        enhanced_mask = support_mask + background_mask * alpha_map  # [B, 1, H, W]
        return enhanced_mask

# 计算带有判别信息的原型
def generate_disc_prompt(feat_s, mask_s, feat_q, eps=1e-6):
    """
        Generate discriminative prior mask P_Q^Disc.

        Args:
            feat_s: Support features, [B, C, H, W]
            mask_s: Support foreground mask, [B, 1, H, W]
            feat_q: Query features, [B, C, H, W]
        Returns:
            P_Q_disc: Discriminative prior mask, [B, H, W]
        """
    B, C, H, W = feat_s.shape

    # 1. Generate FG and BG masks
    fg_mask = (mask_s == 1).float()  # [B, 1, H, W]
    bg_mask = (mask_s == 0).float()  # [B, 1, H, W]

    # 2. Global Average Pooling to get prototypes
    def masked_gap(feat, mask):
        masked_feat = feat * mask  # broadcasting [B, C, H, W] × [B, 1, H, W]
        gap = masked_feat.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + eps)  # [B, C]
        return gap  # [B, C]
    # 3. Normalize prototypes and query features
    proto_fg = masked_gap(feat_s, fg_mask)
    proto_bg = masked_gap(feat_s, bg_mask)

    feat_q_flat = feat_q.view(B, C, -1)  # [B, C, H*W]
    feat_q_norm = F.normalize(feat_q_flat, dim=1)  # cosine-normalized

    proto_fg_norm = F.normalize(proto_fg.unsqueeze(2), dim=1)  # [B, C, 1]
    proto_bg_norm = F.normalize(proto_bg.unsqueeze(2), dim=1)
    # 4. Cosine similarity
    P_q_fg = torch.sum(feat_q_norm * proto_fg_norm, dim=1)  # [B, H*W]
    P_q_bg = torch.sum(feat_q_norm * proto_bg_norm, dim=1)  # [B, H*W]

    # 5. Reshape and min-max normalize
    def minmax_norm(x):
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + eps)

    # P_q_fg = minmax_norm(P_q_fg).view(B, H, W)
    # P_q_bg = minmax_norm(P_q_bg).view(B, H, W)
    P_q_fg = P_q_fg.view(B, H, W)
    P_q_bg = P_q_bg.view(B, H, W)
    # 6. Discriminative prior mask
    P_q_disc = F.relu(P_q_fg - P_q_bg)  # remove negatives
    P_q_disc = minmax_norm(P_q_disc.view(B, -1)).view(B, H, W)
    return P_q_disc,P_q_fg, P_q_bg # [B, H, W]


def visualize_features_pca(feats, img_size=(473, 473)):
    """
    feats: [B, C, H, W] (例如 [1, 768, 32, 32])
    """
    # 1. 展平空间维度
    B, C, H, W = feats.shape
    feats = feats.permute(0, 2, 3, 1).reshape(-1, C) # [B*H*W, C]
    
    # 2. PCA 降维到 3 (对应 RGB)
    feats = feats.cpu().detach().numpy()
    pca = PCA(n_components=3)
    pca_feats = pca.fit_transform(feats) # [B*H*W, 3]
    
    # 3. 归一化到 [0, 1] 以便绘图
    m_min = pca_feats.min(axis=0)
    m_max = pca_feats.max(axis=0)
    pca_feats = (pca_feats - m_min) / (m_max - m_min)
    
    # 4. 恢复形状并上采样到原图大小
    pca_img = pca_feats.reshape(B, H, W, 3)
    pca_tensor = torch.from_numpy(pca_img).permute(0, 3, 1, 2).float() # [B, 3, H, W]
    
    # 上采样以便看清楚细节
    pca_upsampled = F.interpolate(pca_tensor, size=img_size, mode='nearest') # 用nearest看马赛克，用bilinear看平滑度
    
    return pca_upsampled.permute(0, 2, 3, 1).numpy()[0] # [H_orig, W_orig, 3]
class VRP_encoder(nn.Module):
    def __init__(self, args, backbone, use_original_imgsize):
        super(VRP_encoder, self).__init__()
        self.args = args
        self.global_tsne_counter = 0

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=True)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.layer0.eval(), self.layer1.eval(), self.layer2.eval(), self.layer3.eval(), self.layer4.eval()
        if backbone == 'vgg16':
            fea_dim = 512 + 256
        else:
            fea_dim = 512 + 1024
        hidden_dim = 256
        self.downsample_query = nn.Sequential(
            nn.Conv2d(fea_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))

        self.downsample_sam_query = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.num_query = args.num_query

        self.transformer_decoder = transformer_decoder(args, args.num_query,fea_dim, hidden_dim, hidden_dim * 2,
                                                       num_layers=args.num_layers)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        # self.aug_N = args.aug_N
        # self.mask_enhancer = SpatialAwareMaskEnhancer(in_channels=hidden_dim)


    # condition :'mask' query_sam_feat:(8,256,64,64)   supp_Sam_feat:(8,256,64,64)
    # query_img:(8,3,512,512) support_mask:(8,512,512)
    # support_img： [8,3,512,512]
    def forward(self, condition, query_sam_feat, supp_sam_feat,
                query_img, query_mask, support_img, support_mask,training,
                # query_se, support_se,
                batch_id
                ):

        if condition == 'scribble':
            support_mask_ori = get_scribble_mask(support_mask, training)  # scribble_mask
        elif condition == 'point':
            support_mask_ori = get_point_mask(support_mask, training)  # point_mask
        elif condition == 'box':
            boxes, support_mask_ori = get_bounding_boxes(support_mask)  # box_mask
        elif condition == 'mask':
            support_mask_ori = support_mask  # [8,512,512]
        with torch.no_grad():
            # 通过逐层处理 support_img（支持图像）来提取特征
            query_feat_0 = self.layer0(query_img)  # [8,128,128,128]
            query_feat_1 = self.layer1(query_feat_0)  # [8,256,128,128]
            query_feat_2 = self.layer2(query_feat_1)  # [8,512,128,128]
            query_feat_3 = self.layer3(query_feat_2)  # [8,1024,128,128]
            query_feat_4 = self.layer4(query_feat_3)  # [8,2048,128,128]

            if self.backbone_type == 'vgg16':
                query_feat_2 = F.interpolate(query_feat_2, size=(64, 64), mode='bilinear', align_corners=True)  # [4,256,64,64]
                query_feat_3 = F.interpolate(query_feat_3, size=(64, 64), mode='bilinear', align_corners=True)  # [4,512,64,64]
                query_feat_4 = F.interpolate(query_feat_4, size=(64, 64), mode='bilinear', align_corners=True)  # [4,512,64,64]
            # 将 query_feat_3 和 query_feat_2 按通道方向（维度 1）进行拼接，
            # 生成 query_feat，形状为 [8, 1536, 128, 128]。---vgg[4,768,64,674]
            query_feat = torch.cat([query_feat_3, query_feat_2], 1)

            # 单视角的变换增强--图像级
            aug_imgs, aug_masks, aug_ids = collect_augmented_views(support_img, batch_id, support_mask)

            if support_img.dim() == 5 and support_img.size(1) != 1:
                B, K, C, H, W = support_img.shape
                support_img = support_img.view(B * K, C, H, W)  #(40,3,512,512)
                supp_feat_0 = self.layer0(support_img)  # [40,128,128,128]
                supp_feat_1 = self.layer1(supp_feat_0)  # [40,256,128,128]
                supp_feat_2 = self.layer2(supp_feat_1)  # [40,512,64,64]
                supp_feat_3 = self.layer3(supp_feat_2)  # [40,1024,64,64]
                # support_img = support_img.view(B,K, C, H, W)
                support_mask_ori = support_mask_ori.view(B*K, H, W)
                support_mask = F.interpolate(support_mask_ori.unsqueeze(1).float(),
                                             size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                             mode='nearest')  # [8,1,64,64]or[40,1,64,64]
            else:
                B, C, H, W = support_img.shape
                K =1
                # 通过逐层处理 support_img（支持图像）来提取特征 [8,5,3,512,512]
                supp_feat_0 = self.layer0(support_img)  # [8,128,128,128]---[4,64,256,256]
                supp_feat_1 = self.layer1(supp_feat_0)  # [8,256,128,128]---[4,128,128,128]
                supp_feat_2 = self.layer2(supp_feat_1)  # [8,512,128,128]---[4,64,256,256]
                supp_feat_3 = self.layer3(supp_feat_2)  # [8,1024,128,128]---[4,512,32,32]
                support_mask_ori = support_mask_ori.view(B * K, H, W)
                support_mask = F.interpolate(support_mask_ori.unsqueeze(1).float(),
                                             size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                             mode='nearest')  # [8,1,64,64]or[40,1,64,64]---[4,1,32,32]
            _supp_feat_4 = self.layer4(supp_feat_3 * support_mask)  # [8,2048,64,64]
            supp_feat_4 = self.layer4(supp_feat_3)  # [8,2048,64,64]---[4,512,32,32]
            if self.backbone_type == 'vgg16':
                supp_feat_2 = F.interpolate(supp_feat_2, size=(64, 64), mode='bilinear', align_corners=True)
                supp_feat_3 = F.interpolate(supp_feat_3, size=(64, 64), mode='bilinear', align_corners=True)
                supp_feat_4 = F.interpolate(supp_feat_4, size=(64, 64), mode='bilinear', align_corners=True)
                _supp_feat_4 = F.interpolate(_supp_feat_4, size=(64, 64), mode='bilinear', align_corners=True)
            # 将 supp_feat_3 和 supp_feat_2 按通道方向拼接，生成 supp_feat
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)  # [8,1536,64,64]---【4，768，64，64】
            # 将查询掩码和支持掩码的大小调整为 [8, 1, 64, 64]，以便与特征图尺寸对齐
            query_mask = F.interpolate(query_mask.unsqueeze(1).float(), size=(64, 64), mode='nearest')  # [8,1,64,64]
            support_mask = F.interpolate(support_mask_ori.unsqueeze(1).float(), size=(64, 64),
                                         mode='nearest')  # [8,1,64,64]
            # 通过 get_pseudo_mask 函数计算伪标签（pseudo mask），这是从支持图像和查询图像的特征中生成的掩码。
            pseudo_mask = self.get_pseudo_mask(_supp_feat_4, query_feat_4, support_mask)  # [8,1,64,64]
            # ----------------------增强图像视角处理-----------------------------
            if aug_imgs is not None:
                with torch.no_grad():  # 不参与训练
                    aug_feat_0 = self.layer0(aug_imgs)
                    aug_feat_1 = self.layer1(aug_feat_0)
                    aug_feat_2 = self.layer2(aug_feat_1)
                    aug_feat_3 = self.layer3(aug_feat_2)
                    if self.backbone_type == 'vgg16':
                        aug_feat_2 = F.interpolate(aug_feat_2, size=(64, 64), mode='bilinear', align_corners=True)
                        aug_feat_3 = F.interpolate(aug_feat_3, size=(64, 64), mode='bilinear', align_corners=True)

                    aug_feat = torch.cat([aug_feat_3, aug_feat_2], dim=1)  # [B_aug, 1536, 64, 64]

                    # 拼接原图 + 增强图用于多视角 GAT 构图
                    supp_feat_all = torch.cat([supp_feat, aug_feat], dim=0)  # [B + B_aug, 1536, 64, 64]
                    batch_id_all = torch.cat([batch_id, aug_ids], dim=0)  # [B + B_aug]
            else:
                supp_feat_all = supp_feat
                batch_id_all = batch_id

        """projection of sam and resnet feature"""
        # 代表 ConvG 的卷积操作
        query_feat = self.downsample_query(query_feat)  # [8,1536,128,128]---[8,256,64,64]
        supp_feat = self.downsample_query(supp_feat)  # [8,1536,64,64]---[8,256,64,64]
        supp_feat_all = self.downsample_query(supp_feat_all)

        # 背景的mask
        support_bg_mask = (support_mask == 0).float()  # 也保持 [B, 1, H, W]
        # resnet50的聚合
        #prototype = self.mask_feature(supp_feat, support_mask, support_se)  # [8,256,1,1]
        prototype = self.mask_feature(supp_feat, support_mask, support_mask)  # [8,256,1,1]
        prototype_bg = self.mask_feature(supp_feat, support_bg_mask, support_bg_mask)

        supp_feat_bin = prototype.repeat(1, 1, query_feat.shape[2], query_feat.shape[3])  # [8,256,64,64]
        supp_feat_bin_bg = prototype_bg.repeat(1, 1, query_feat.shape[2], query_feat.shape[3])
        # sam的聚合
        # prototype_sam = self.mask_feature(supp_sam_feat, support_mask, support_se)  # [8,256,1,1]
        prototype_sam = self.mask_feature(supp_sam_feat, support_mask, support_mask)  # [8,256,1,1]
        prototype_sam_bg = self.mask_feature(supp_sam_feat, support_bg_mask, support_bg_mask)

        supp_feat_bin_sam = prototype_sam.repeat(1, 1, query_feat.shape[2], query_feat.shape[3])  # [8,256,64,64]
        supp_feat_bin_sam_bg = prototype_sam_bg.repeat(1, 1, query_feat.shape[2], query_feat.shape[3])

        supp_sam_feat = self.downsample_sam_query(torch.cat([supp_sam_feat], 1))  # [8,256,64,64]
        query_sam_feat = self.downsample_sam_query(torch.cat([query_sam_feat], 1))  # [8,256,64,64]

        """feature projection for pseudo mask"""
        # 这里输入的support_mask要修改成带有背景的、以及sam_decoder生成的？
        # spt_prototype = self.mask_feature(supp_feat_4, support_mask, support_se)  # [8,2048,1,1]
        spt_prototype = self.mask_feature(supp_feat_4, support_mask, support_mask)  # [8,2048,1,1]



        # 加入get_distriministic_mask
        proto_query_res_dis,proto_query_res_fg,proto_query_res_bg = generate_disc_prompt(supp_feat_bin,support_mask,query_feat)
        proto_query_sam_dis,proto_query_sam_fg,proto_query_sam_bg = generate_disc_prompt(supp_sam_feat,support_mask,query_sam_feat)

        deterministic_dist = {
            "res": {  # 对应 supp_feat_bin / query_feat 分支
                "disc": proto_query_res_dis,  # 判别性掩码
                "fg": proto_query_res_fg,  # 前景相似度图
                "bg": proto_query_res_bg  # 背景相似度图
            },
            "sam": {  # 对应 supp_sam_feat / query_sam_feat 分支
                "disc": proto_query_sam_dis,
                "fg": proto_query_sam_fg,
                "bg": proto_query_sam_bg
            }
        }

        """generating prompts for sam mask decoder"""
        # 更新后的 output
        # 并根据需要返回注意力图（s_c_attn_map, q_c_attn_map）
        # 和伪标签（pseudo_mask_vis）以及计算的损失
        protos, attn_lst, dict_for_loss ,vis_q= self.transformer_decoder(query_feat,
                                                                   supp_feat,
                                                                   query_sam_feat,
                                                                   supp_sam_feat,
                                                                   support_mask,
                                                                   spt_prototype,
                                                                   query_feat_4,
                                                                   pseudo_mask,
                                                                   supp_feat_bin,
                                                                   supp_feat_bin_sam,
                                                                   supp_feat_all,
                                                                   batch_id_all,
                                                                   batch_id,
                                                                   # 新增
                                                                   deterministic_dist
                                                                   )
        # protos[8,50,256]  attn_list,dict_for_loss[]
        #为了可视化
        return protos, attn_lst, dict_for_loss,vis_q,deterministic_dist

    # [8,256,64,64]
    # mp操作
    def mask_feature(self, features, support_mask, se_mask):
        # # 加入背景信息
        # alpha = 0.1  # 控制背景信息的权重
        # background_mask = 1 - support_mask  # 获取背景掩码[8, 1, 64, 64]
        # enhanced_mask = support_mask + background_mask * alpha  # 融合前景和背景
        mask = support_mask  # [8,1,64,64]
        # device = features.device
        # support_mask = support_mask.to(device)
        # se_mask = se_mask.to(device)
        # alpha = 0.2
        # enhanced_mask = support_mask
        # mask = enhanced_mask  # [8,1,64,64]
        # 通过 mask 对 features（特征图）进行加权
        supp_feat = features * mask  # [8,256,64,64]*[8,1,64,64]----[8,256,64,64]
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]  # [64] [64]
        area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005  # [8,1,1,1]
        supp_feat = F.avg_pool2d(input=supp_feat,
                                 kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  # [8,256,1,1]
        # [8,256,1,1]
        return supp_feat

    def predict_mask_nshot(self, args, batch, sam_model,idx, nshot,input_point=None):

        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0  # 用于聚合所有 nshot 预测的结果
        protos_set = []  # 用来保存每次 nshot 预测的 原型，也就是用于计算相似度的特征向量。
        # 可视化
        features = []  # 原型特征
        labels = []  # 每个原型所属的类别（由 idx 决定）
        views = []  # 每个原型来自哪个视角（shot index）
        original_feat = []
        class_idx = idx  # before the loop
        for s_idx in range(nshot):
            query_sam_feat = sam_model.get_feat_from_np(batch['query_img'], batch['query_name'],
                                                        torch.tensor([0]).cuda())  # [1,256,64,64]
            supp_sam_feat = sam_model.get_feat_from_np(batch['support_imgs'].squeeze(1), batch['support_names'][0],
                                                       torch.tensor([0]).cuda())  # [1,256,64,64]
            protos_sub, support_mask, _,vis_q,deterministic_dist,multi_view_x_q, multi_view_x_s= self(args.condition, query_sam_feat, supp_sam_feat, batch['query_img'],
                                               batch['query_mask'], batch['support_imgs'][:, s_idx],
                                               batch['support_masks'][:, s_idx], False, idx)
            protos_set.append(protos_sub)  # ⬆️是VRP_encoder中的forward protp_sub[1,50,256],support_mask
            proto_feat = protos_sub.squeeze(0).mean(axis=0).cpu().numpy() # [50, 256]
            original_feat.append(vis_q)
            features.append(proto_feat)  # 原型 token
            labels.append(int(idx))
            views.append(int(s_idx))
        if nshot > 1:
            logit_masks = []
            for protos in protos_set:
                low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos, input_point)
                logit_mask = low_masks
                logit_masks.append(logit_mask)
            # 将所有预测结果的掩码进行聚合。首先将每次预测的 logit_mask 合并，并对其取均值来得到最终的掩码预测。
            logit_mask = torch.cat(logit_masks, dim=1).mean(1, keepdim=True)
            # 对预测结果进行 二值化，使用 sigmoid 激活函数并阈值化为 0 或 1，得到二值掩码 pred_mask。
            pred_mask = torch.sigmoid(logit_mask) >= 0.5  #

            pred_mask = pred_mask.float()
            # 累计所有 nshot 次的二值掩码，并返回最终的结果。
            logit_mask_agg += pred_mask.squeeze(1).clone()
            features_np = np.stack(features, axis=0) # shape: [nshot, 256]
            original_feat_np = torch.cat(original_feat, dim=0).cpu().numpy()
            labels_np = np.array(labels)
            views_np = np.array(views)
            tsne_vis_params = {
                "original_feat_np": original_feat_np,  # 原始特征，shape: [N, D]
                "features_np": features_np,  # 增强特征，shape: [N, D]
                "views_np": views_np,  # 视角索引，shape: [N]
                "class_idx": class_idx.item(),  # 类别编号（转换为 int）.
                "cross_atten": deterministic_dist['cross_atten']
            }
            self.global_tsne_counter += 1
            return logit_mask_agg, support_mask, logit_mask,deterministic_dist,tsne_vis_params


        else:
            protos = protos_sub  # [1,50,256]
        #  [1,1,512,512] , [1,512,512]
        low_masks, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos, input_point)
        logit_mask = low_masks
        if self.use_original_imgsize:
            org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
            logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)  # 【1，1，512，512】
        pred_mask = torch.sigmoid(logit_mask) >= 0.5

        pred_mask = pred_mask.float()

        logit_mask_agg += pred_mask.squeeze(1).clone()
        features_np = np.stack(features, axis=0)  # shape: [nshot, 256]
        original_feat_np = torch.cat(original_feat, dim=0).cpu().numpy()
        labels_np = np.array(labels)
        views_np = np.array(views)
        return logit_mask_agg, support_mask, logit_mask,deterministic_dist,_,multi_view_x_q, multi_view_x_s  # [1，512，512]--list{4}--[1,1,512,512]

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        loss_bce = self.bce_with_logits_loss(logit_mask.squeeze(1), gt_mask.float())
        loss_dice = dice_loss(logit_mask, gt_mask, bsz)
        return loss_bce + loss_dice

    def compute_pseudo_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        loss_bce = self.bce_with_logits_loss(logit_mask.squeeze(1), gt_mask.float())
        loss_dice = dice_loss(logit_mask, gt_mask, bsz)
        return loss_bce + loss_dice

    def train_mode(self):
        self.train()
        self.apply(fix_bn)
        self.layer0.eval(), self.layer1.eval(), self.layer2.eval(), self.layer3.eval(), self.layer4.eval()
        # [8,2048,64,64] [8,2048,64,64]  [8,1,64,64]
    def get_pseudo_mask(self, tmp_supp_feat, query_feat_4, mask):
        resize_size = tmp_supp_feat.size(2) # tmp_supp_feat [8,2048,64,64]
        tmp_mask = F.interpolate(mask, size=(resize_size, resize_size), mode='bilinear', align_corners=True)    #[8,1,64,64]
        #tmp_supp_feat:mask加权后的支持集特征, query_feat_4:查询视频特征, mask：支持集掩码
        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask  # 乘后的特征又乘一次？  [8,2048,64,64]
        q = query_feat_4    # [8.2048,64,64]
        s = tmp_supp_feat_4 # [8.2048,64,64]
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1) #[8,2048,4096]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)  # L2范式[8,1,4096]

        tmp_supp = s
        tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1) #[8,2048,4096]
        tmp_supp = tmp_supp.permute(0, 2, 1)    #[8,4096,2048]
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)    #[8,4096,1]

        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)  #[8,4096,4096]
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)   #逐像素之间的余弦相似度计算 [8,4096,1]*[8,1,4096]--选出最高的[8,4096]
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz) #[8,1,64,64]
        return corr_query
    def get_pseudo_mask_5shot(self, tmp_supp_feat, query_feat_4, mask,B,K):
        resize_size = tmp_supp_feat.size(2)  # 64
        tmp_mask = F.interpolate(mask, size=(resize_size, resize_size), mode='bilinear',
                                 align_corners=True)  # [8,1,64,64]

        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask  # [8,2048,64,64]
        q = query_feat_4    # [8,2048,64,64]
        s = tmp_supp_feat_4 # [40,2048,64,64]
        bsize, ch_sz, sp_sz, _ = q.size()[:]
        bsize_s,sch_sz, ssp_sz, _ = s.size()[:] # 40,2048,64,
        # [8,2048,64,64]
        tmp_query = q
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1)  # [8,2048,4096]
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)  # [8,1,4096]
        tmp_supp = s
        if K>1:
            tmp_supp = tmp_supp.unsqueeze(1)
            tmp_supp = tmp_supp.reshape(B, K, sch_sz, -1)
        else:
            tmp_supp = tmp_supp.reshape(bsize_s, sch_sz, -1)  # [8,2048,4096]
        tmp_supp = tmp_supp.permute(0, 2, 1)  # [40,4096,2048]
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)  # [8,4096,1]
        # tmp_supp:[40,4096,2048]  tmp_supp_norm:[40,4096,1]
        cosine_eps = 1e-7
        similarity = torch.bmm(tmp_supp, tmp_query) / (
                    torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)  # [8,4096,4096]
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)  # [8,4096]
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)  # [8,1,64,64]
        return corr_query