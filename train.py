import argparse
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
from model.VINE import VRP_encoder
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator, Evaluator_pseudo
from common import utils
from data.dataset import VINEDataset
from SAM2pred import SAM_pred
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = False
from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table

relu = nn.ReLU()
def setup_distributed(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return local_rank

def create_dataloader(args, fold, split):
    dataset = VINEDataset.build_dataset(args.benchmark, fold, split, shot=1)
    sampler = DistributedSampler(dataset, shuffle=(split == 'trn'))
    dataloader = DataLoader(dataset, batch_size=args.bsz, sampler=sampler,
                            num_workers=args.nworker, pin_memory=True)
    return dataloader
def l2norm(x):
    return F.normalize(x, p=2, dim=-1)


def cal_sim_loss(c_attn_map):
    c_attn_map = l2norm(c_attn_map)
    cur_loss = (c_attn_map @ c_attn_map.permute(0, 2, 1) - torch.eye(c_attn_map.shape[1]).cuda())
    cur_loss = (relu(cur_loss)).mean()
    return cur_loss


import torch
import torch.nn.functional as F

def train(args, epoch, model, sam_model, dataloader, optimizer, scheduler, training):
    utils.fix_randseed(args.seed + epoch) if training else utils.fix_randseed(args.seed)
    model.train_mode() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    average_meter_pseudo = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        batch = utils.to_cuda(batch)
        batch_id = batch['class_id']
        query_sam_feat = sam_model.get_feat_from_np(batch['query_img'], batch['query_name'], torch.tensor([0]).cuda())
        supp_sam_feat = sam_model.get_feat_from_np(batch['support_imgs'].squeeze(1), batch['support_names'][0],
                                                   torch.tensor([0]).cuda())

        protos, attn_lst, dict_for_loss,_ ,_= model(args.condition, query_sam_feat, supp_sam_feat,
                                                batch['query_img'], batch['query_mask'],
                                                batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1),training,
                                                batch_id )

        query_mask = F.interpolate(batch['query_mask'].unsqueeze(1).float(), size=(64, 64), mode='nearest').squeeze(1)
        pseudo_masks = dict_for_loss['pseudo_mask_loss']
        pseudo_mask_loss = 0
        for m, msk in enumerate(pseudo_masks):
            pseudo_mask_loss = pseudo_mask_loss + model.compute_pseudo_objective(msk, query_mask) / (
                    len(pseudo_masks) - m)

        contrastive_loss = utils.info_nce_loss(
            dict_for_loss['proto_q'],
            dict_for_loss['proto_s'],
            batch['class_id'],
            temperature=0.07,
            detach_query=True
        )
        logit_mask, pred_mask = sam_model(batch['query_img'], batch['query_name'], protos)
        pred_mask = (torch.sigmoid(logit_mask) > 0.5).float()
        prompt_loss = model.compute_objective(logit_mask, batch['query_mask'])

        loss = (prompt_loss * args.prompt_loss +
                pseudo_mask_loss * args.mask_loss +
                contrastive_loss * 1.0
                )

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.squeeze(1), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

        pseudo_masks[-1] = (pseudo_masks[-1] > 0.5).float()
        area_inter_pseudo, area_union_pseudo, area_pred_pseudo, area_gt_pseudo = Evaluator_pseudo.classify_prediction(
            pseudo_masks[-1].squeeze(1), query_mask, batch)
        average_meter_pseudo.update(area_inter_pseudo, area_union_pseudo, batch['class_id'], loss.detach().clone(),
                                    pred=area_pred_pseudo, gt=area_gt_pseudo)
        average_meter_pseudo.write_process(idx, len(dataloader), epoch, write_batch_idx=50, miou_only=True)

    average_meter.write_result('Training' if training else 'Validation', epoch)
    average_meter_pseudo.write_result('Training' if training else 'Validation', epoch, miou_only=True)

    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()
    miou_pseudo, _ = average_meter_pseudo.compute_iou()
    return avg_loss, miou, fb_iou, miou_pseudo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual Prompt Encoder Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='./dataset')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str,
                        default='PASCAL0')
    parser.add_argument('--bsz', type=int, default=4)
    parser.add_argument('--prompt_loss', type=float, default=0.5)
    parser.add_argument('--mask_loss', type=float, default=0.01)
    
    parser.add_argument('--attn_loss', type=float, default=0.01)
    parser.add_argument('--attn_drop_out', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--nworker', type=int, default=4)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--condition', type=str, default='mask', choices=['point', 'scribble', 'box', 'mask'])
    parser.add_argument('--use_ignore', type=bool, default=True,
                        help='Boundaries are not considered during pascal training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_query', type=int, default=50)
    parser.add_argument('--spt_num_query', type=int, default=50)
    parser.add_argument('--concat_th', type=bool, default=False)
    parser.add_argument('--use_log', action='store_true')
    parser.add_argument('--load', type=str, default="./dummy.pt")
    parser.add_argument('--eval_flops', action='store_true',
                        help='Only compute FLOPs/Params and exit')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101','swin','dino'])
    args = parser.parse_args()
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

    if utils.is_main_process():
        Logger.initialize(args, training=args.use_log)
    utils.fix_randseed(args.seed)
    model = VRP_encoder(args, args.backbone, False)
    if utils.is_main_process():
        Logger.log_params(model)
    sam_model = SAM_pred()
    sam_model.to(device)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([
        {'params': model.downsample_query.parameters(), "lr": args.lr},
        {'params': model.downsample_sam_query.parameters(), "lr": args.lr},
        {'params': model.transformer_decoder.parameters()},
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    Evaluator.initialize(args)
    Evaluator_pseudo.initialize(args)

    VINEDataset.initialize(img_size=512, datapath=args.datapath, use_original_imgsize=False)
    dataloader_trn = VINEDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn', shot=1)
    dataloader_val = VINEDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val', shot=1)
    print(len(dataloader_trn), len(dataloader_val))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader_trn))
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.epochs):

        trn_loss, trn_miou, trn_fb_iou, miou_pseudo = train(args, epoch, model, sam_model, dataloader_trn, optimizer,
                                                            scheduler, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou, miou_pseudo = train(args, epoch, model, sam_model, dataloader_val,
                                                                optimizer, scheduler, training=False)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            if utils.is_main_process():
                Logger.save_model_miou(model, epoch, val_miou)
        if utils.is_main_process():
            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')