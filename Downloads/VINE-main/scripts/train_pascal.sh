#!/bin/bash
GPUNUM=0,1
PORTNUM=$((10940+GPUNUM))

PROMPT_LOSS=1.0
mask_loss=0.5
attn_loss=0.05
attn_drop_out=0.5
num_layers=2
spt_num_query=50

for FOLDNUM in 1
do
CUDA_VISIBLE_DEVICES=$GPUNUM python3 -m torch.distributed.launch \
        --nproc_per_node=2 \
        --master_port=$PORTNUM train_gpu.py \
        --datapath ../dataset \
        --logpath pasacal-fold$FOLDNUM \
        --benchmark pascal \
        --backbone resnet50 \
        --fold $FOLDNUM \
        --seed 42 \
        --num_layers $num_layers \
        --prompt_loss $PROMPT_LOSS \
        --mask_loss $mask_loss \
        --attn_loss $attn_loss \
        --attn_drop_out $attn_drop_out \
        --condition mask \
        --num_query 50 \
        --spt_num_query $spt_num_query \
        --nworker 4 \
        --epochs 150 \
        --lr 2e-4 \
        --bsz 4 \
        # --use_log 
done