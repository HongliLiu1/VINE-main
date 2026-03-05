r""" Visual Prompt Encoder training (validation) code """
from collections import defaultdict

import os
import argparse
import sys

import torch.nn as nn
import torch
import torch.distributed as dist

# from model.VRP_encoder_v1 import VRP_encoder
from model.VRP_encoder_v1_newloss import VRP_encoder
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import VINEDataset
from SAM2pred import SAM_pred
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def visualize_disc_maps(P_q_fg, P_q_bg, P_q_disc, save_dir='./vis_disc_prompt', idx=0):
    os.makedirs(save_dir, exist_ok=True)
    maps = {'fg_sim': P_q_fg, 'bg_sim': P_q_bg, 'disc_prior': P_q_disc}
    import matplotlib.cm as cm
    B = P_q_fg.shape[0]  # batch size
    for i in range(B):
        fg = P_q_fg[i].cpu().numpy()
        bg = P_q_bg[i].cpu().numpy()
        disc = P_q_disc[i].cpu().numpy()

        cmap = cm.get_cmap('jet')

        fg_rgb = (cmap(fg)[..., :3] * 255).astype(np.uint8)  # shape: [H, W, 3]
        bg_rgb = (cmap(bg)[..., :3] * 255).astype(np.uint8)
        disc_rgb = (cmap(disc)[..., :3] * 255).astype(np.uint8)

        concat_img = np.concatenate([fg_rgb, bg_rgb, disc_rgb], axis=1)  # [H, W*3, 3]

        save_path = os.path.join(save_dir, f'disc_map_cat_{idx}.png')
        plt.imsave(save_path, concat_img)
        print(f"[Saved] {save_path}")


def plot_tsne_comparison_multiclass(original_feat_np, features_np, class_labels, save_path, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=min(perplexity, original_feat_np.shape[0] - 1), random_state=42)

    all_feats = np.concatenate([original_feat_np, features_np], axis=0)
    all_feats_2d = tsne.fit_transform(all_feats)

    original_feat_2d = all_feats_2d[:original_feat_np.shape[0]]
    features_2d = all_feats_2d[original_feat_np.shape[0]:]
    class_labels = np.array(class_labels)
    x_min = min(original_feat_2d[:, 0].min(), features_2d[:, 0].min()) - 1
    x_max = max(original_feat_2d[:, 0].max(), features_2d[:, 0].max()) + 1
    y_min = min(original_feat_2d[:, 1].min(), features_2d[:, 1].min()) - 1
    y_max = max(original_feat_2d[:, 1].max(), features_2d[:, 1].max()) + 1

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for c in np.unique(class_labels):
        idx = class_labels == c
        plt.scatter(original_feat_2d[idx, 0], original_feat_2d[idx, 1],
                    label=f'Class {c}', s=80)
    plt.margins(0.05)
    plt.title('Original Features')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    plt.legend()

    plt.subplot(1, 2, 2)
    for c in np.unique(class_labels):
        idx = class_labels == c
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    label=f'Class {c}', s=80)
    plt.title('Enhanced Prototypes')
    plt.margins(0.05)
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def test(args, epoch, model, sam_model, dataloader, training):
    r""" Train VRP_encoder model """
    training = False
    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    all_original_feats = []
    all_enhanced_feats = []
    all_views = []
    all_labels = []
    tsne_group_idx = 0
    seen_class_ids = set()
    N_PER_CLASS = 50
    classwise_original_feats = defaultdict(list)
    classwise_enhanced_feats = defaultdict(list)
    classwise_labels = defaultdict(list)
    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)
        # print(f"Query Image Shape: {batch['query_img'].shape}")     # query image [1,3,512,512]
        # print(f"Support Image Shape: {batch['support_imgs'].shape}")    # support image [1,1,3,512,512]
        if torch.all(batch['query_mask'] == 0):
            print(f"Batch index: {idx}")
            print(f"Query Mask Shape: {batch['query_mask'].shape}")
            print(f"Query Mask: {batch['query_mask']}")
            print(f"Query Image: {batch['query_img']}")
            print(f"Support Images: {batch['support_imgs']}")
            raise ValueError("Error: query_mask is all zeros!")
        # [1,512,512]---atten_list-list{4}
        pred_mask, attn_lst, _, deterministic_dist, tsne_vis_params = model.predict_mask_nshot(args, batch, sam_model,
                                                                                               batch['class_id'],
                                                                                               nshot=args.nshot)

        # original_feat_np = tsne_vis_params["original_feat_np"]
        # features_np = tsne_vis_params["features_np"]
        # views_np = tsne_vis_params["views_np"]
        # class_idx= tsne_vis_params["class_idx"]
        # classwise_original_feats[class_idx].append(original_feat_np)
        # classwise_enhanced_feats[class_idx].append(features_np)
        # classwise_labels[class_idx].append(np.full((features_np.shape[0],), int(class_idx)))
        #
        # seen_class_ids.add(class_idx)
        # if len(seen_class_ids) == 4:
        #     if all(sum(arr.shape[0] for arr in classwise_labels[c]) >= N_PER_CLASS for c in seen_class_ids):
        #         original_feat_np_cat = np.concatenate([
        #             np.concatenate(classwise_original_feats[c], axis=0)[:N_PER_CLASS]
        #             for c in seen_class_ids
        #         ])
        #         features_np_cat = np.concatenate([
        #             np.concatenate(classwise_enhanced_feats[c], axis=0)[:N_PER_CLASS]
        #             for c in seen_class_ids
        #         ])
        #         class_labels_cat = np.concatenate([
        #             np.concatenate(classwise_labels[c], axis=0)[:N_PER_CLASS]
        #             for c in seen_class_ids
        #         ])
        #
        #         tsne_vis_params = {
        #             "original_feat_np": original_feat_np_cat,
        #             "features_np": features_np_cat,
        #             "class_labels": class_labels_cat,
        #             "save_path": f"./tsne_vis_class8_pascal30/tsne_group_{tsne_group_idx}.png",
        #             "perplexity": 30
        #         }
        #
        #         # plot_tsne_comparison_multiclass(**tsne_vis_params)
        #         print(f"[t-SNE] Saved tsne_group_{tsne_group_idx}.png")
        #
        #         for c in seen_class_ids:
        #             classwise_original_feats[c].clear()
        #             classwise_enhanced_feats[c].clear()
        #             classwise_labels[c].clear()
        #         seen_class_ids.clear()
        #         tsne_group_idx += 1

        P_disc_sam, P_fg_sam, P_bg_sam = deterministic_dist["sam"]["disc"], deterministic_dist["sam"]["fg"], \
        deterministic_dist["sam"]["bg"]
        visualize_disc_maps(P_fg_sam, P_bg_sam, P_disc_sam, save_dir='./vis_disc_prompt_pascal2', idx=idx)
        if torch.all(pred_mask == 0):
            print("Warning: pred_mask is all zeros, skipping this sample...")
            continue
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=1)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

        # """visualization"""
        img_size = batch['query_img'].shape
        # # spt_img = batch['support_imgs'].squeeze(1).squeeze(0).cpu().numpy() #[3,512,512]
        # # spt_img = np.transpose(spt_img, (1, 2, 0))
        qry_img = batch['query_img'].cpu().squeeze(0).numpy()  # [4,3,512,512]
        qry_img = np.transpose(qry_img, (1, 2, 0))

        # """show attn map"""
        # target_img = spt_img
        # # target_img = np.transpose(target_img, (1, 2, 0))  # (3, 512, 512) → (512, 512, 3)
        # target_attn_map = attn_lst[0]
        # dir_path = './vis_s_attn'
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # attn_accum = np.zeros((img_size[2], img_size[3], 3))  # [512, 512, 3]
        # for j in range(target_attn_map.shape[1]):
        #     attn = target_attn_map[:,j,:]
        #     attn = (attn - attn.min())/(attn.max() - attn.min())
        #     attn = attn.reshape(64,64)
        #     attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear', align_corners=True).cpu().squeeze(0).squeeze(0).numpy()
        #
        #     cmap = plt.get_cmap('jet')
        #     attn_colormap = cmap(attn)
        #     attn_colormap = attn_colormap[..., :3]
        # alpha = 0.8
        # attn_accum /= target_attn_map.shape[1]
        # attn_accum = np.clip(attn_accum, 0, 1)
        #
        # combined_img = (1 - alpha) * target_img + alpha * attn_accum
        # combined_img = np.clip(combined_img, 0.0, 1.0)
        # plt.figure(figsize=(10, 10))
        # plt.imshow(combined_img)
        # plt.axis('off')
        # save_path = os.path.join(dir_path, f'attn_accum_{idx}.png')
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        # plt.close()
        # print(f"Saved accumulated attention visualization to: {save_path}")
        #
        # """show attn map"""
        # target_img = qry_img
        # target_attn_map = attn_lst[1]
        # dir_path = './vis_q_attn'
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # for j in range(target_attn_map.shape[1]):
        #     attn = target_attn_map[:,j,:]
        #     attn = (attn - attn.min())/(attn.max() - attn.min())
        #     attn = attn.reshape(64,64)
        #     attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear', align_corners=True).cpu().squeeze(0).squeeze(0).numpy()
        #
        #     cmap = plt.get_cmap('jet')
        #     attn_colormap = cmap(attn)
        #     attn_colormap = attn_colormap[..., :3]
        #
        #     alpha = 0.4
        #     combined_img = (1 - alpha) * target_img + alpha * attn_colormap
        #
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(combined_img)
        #     plt.axis('off')  # 축 제거
        #     plt.savefig(dir_path + '/{}_{}.png'.format(idx,j), bbox_inches='tight', pad_inches=0)
        #     plt.close()

        # """show attn map"""
        # target_img = qry_img
        # target_attn_map = attn_lst[4][0]
        # dir_path = './vis_q_cprior_attn'
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # for j in range(target_attn_map.shape[1]):
        #     attn = target_attn_map[:,j,:]
        #     attn = (attn - attn.min())/(attn.max() - attn.min())
        #     attn = attn.reshape(64,64)
        #     attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear', align_corners=True).cpu().squeeze(0).squeeze(0).numpy()
        #
        #     cmap = plt.get_cmap('jet')
        #     attn_colormap = cmap(attn)
        #     attn_colormap = attn_colormap[..., :3]
        #
        #     alpha = 0.8
        #     combined_img = target_img + alpha * attn_colormap
        #     combined_img = (combined_img - combined_img.min()) / (combined_img.max() - combined_img.min())
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(combined_img)
        #     plt.axis('off')  # 축 제거
        #     plt.savefig(dir_path + '/{}_{}.png'.format(idx,j), bbox_inches='tight', pad_inches=0)
        #     plt.close()

        # """show sub map"""
        # target_img = qry_img
        # target_attn_map = attn_lst[3]
        # dir_path = './vis_sub_map'
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # for j in range(target_attn_map.shape[1]):
        #     attn = target_attn_map[:,j,:]
        #     attn = (attn - attn.min())/(attn.max() - attn.min())
        #     attn = attn.reshape(64,64)
        #     attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear', align_corners=True).cpu().squeeze(0).squeeze(0).numpy()

        #     cmap = plt.get_cmap('jet')
        #     attn_colormap = cmap(attn)
        #     attn_colormap = attn_colormap[..., :3]

        #     alpha = 0.8
        #     combined_img = (1 - alpha) * target_img + alpha * attn_colormap

        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(combined_img)
        #     plt.axis('off')  # 축 제거
        #     plt.savefig(dir_path + '/{}_{}.png'.format(idx,j), bbox_inches='tight', pad_inches=0)
        #     plt.close()

        """show pseudo mask"""
        pseudo_mask_vis = attn_lst[2]
        for k in range(len(pseudo_mask_vis)):
            target_img = qry_img
            target_attn_map = pseudo_mask_vis[k]  # [1,1,64,64]
            dir_path = './vis_pseudo_mask_pascal2'
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            # [1,1,64,64]
            attn = target_attn_map[:, :]
            attn = (attn - attn.min()) / (attn.max() - attn.min())
            attn = attn.reshape(64, 64)  # [64,64]
            attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=img_size[2:], mode='bilinear',
                                 align_corners=True).cpu().squeeze(0).squeeze(0).numpy()
            cmap = plt.get_cmap('jet')
            attn_colormap = cmap(attn)
            attn_colormap = attn_colormap[..., :3]

            alpha = 0.8
            combined_img = (1 - alpha) * target_img + alpha * attn_colormap
            combined_img = (combined_img - combined_img.min()) / (combined_img.max() - combined_img.min())
            plt.figure(figsize=(10, 10))
            plt.imshow(combined_img)
            plt.axis('off')
            plt.savefig(dir_path + '/{}_{}.png'.format(idx, k), bbox_inches='tight', pad_inches=0)
            plt.close()

    average_meter.write_result('Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    # parser.add_argument('--datapath', type=str, default='/mnt/disk18/liuhongli/FCP/dataset')
    parser.add_argument('--datapath', type=str, default='/mnt/disk2/liuhongli/data_fcp')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='test_1_test')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--supp_ratio', type=float, default=0.0)
    parser.add_argument('--mask_loss_lower_bound', type=float, default=0.5)
    parser.add_argument('--pseudo_mask_upper_th', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--concat_th', type=bool, default=False)
    parser.add_argument('--nworker', type=int, default=4)
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='mask', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True,
                        help='Boundaries are not considered during pascal training')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--spt_num_query', type=int, default=50)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--attn_drop_out', type=float, default=0.3)
    parser.add_argument('--local_rank', type=int, default=0,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--load', type=str,
                        default="/mnt/disk1/liuhongli/FCP-main/logs/novelty2-layer2-pascal2/best_model.pt")
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--visualize', type=bool, default=True,
                        help='Boundaries are not considered during pascal training')
    parser.add_argument('--vispath', type=str, default='./vis_final_pascal2_seed321')
    args = parser.parse_args()
    print("Command-line arguments:", sys.argv)
    Logger.initialize(args, training=False)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12487'
    os.environ['WORLD_SIZE'] = '1'
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    local_rank = args.local_rank
    if local_rank != -1:
        import torch

        torch.cuda.set_device(local_rank)

    device = torch.device('cuda', local_rank)

    model = VRP_encoder(args, args.backbone, False)
    model.eval()

    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model.to(device)

    sam_model = SAM_pred()
    sam_model.to(device)
    model.to(device)

    if args.load == '':
        raise Exception('Pretrained model not specified.')

    checkpoint = torch.load(args.load)

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    if 'module.' in list(state_dict.keys())[0]:  # Check if the model was wrapped in DDP
        state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove 'module.' prefix

    for key in list(state_dict.keys()):
        if 'resnet.' in key:  # Example of matching layer names, such as resnet layers
            new_key = key.replace('resnet.', '')  # Adjust this line to match the layers in your model
            state_dict[new_key] = state_dict.pop(key)

    model.load_state_dict(state_dict, strict=True)
    Logger.log_params(model)
    Logger.log_params(sam_model)
    Evaluator.initialize(args)
    Visualizer.initialize(args.visualize, "./vis_final/")

    VINEDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_test = VINEDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val',
                                                  shot=args.nshot)

    # Test
    with torch.no_grad():
        test_miou, test_fb_iou = test(args, 0, model, sam_model, dataloader_test, False)
    Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')
