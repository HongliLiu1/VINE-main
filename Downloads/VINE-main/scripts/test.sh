num_layers=7

CUDA_VISIBLE_DEVICES=1 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=62203 test.py \
        --datapath /mnt/disk18/liuhongli/FCP/dataset \
        --logpath test_1_test \
        --benchmark pascal \
        --backbone resnet50 \
        --fold 1 \
        --condition mask \
        --num_query 50 \
        --num_layers $num_layers \
        --epochs 50 \
        --local-rank 1 \
        --lr 1e-4 \
        --bsz 1 \
        --load /mnt/disk18/liuhongli/FCP_v2/FCP-main/logs/_TRAIN__FOLD1_LAYER7__0228_113525.log/best_model.pt \
        --nshot 1