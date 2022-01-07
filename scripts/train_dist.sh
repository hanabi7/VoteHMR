batch_size=42

CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch \
    --nproc_per_node=2 --master_port 29502 /data1/liuguanze/human_point_cloud/src/train_dist.py \
    --dist \
    --print_freq 10880 \
    --batchSize $batch_size --isTrain \
    --dyna_use_male --dyna_use_female \
    --surreal_use_male --surreal_use_female \
    --lr_e 1e-4 \
    --loss_3d_weight_before 2000 \
    --loss_3d_weight_after 1000 \
    --loss_offset_weight 0.5 \
    --loss_dir_weight 1 \
    --loss_smpl_weight 4000 \
    --loss_vertex_weight 4000 \
    --loss_segment_weight 10 \
    --save_epoch_freq 5 \
    --total_epoch 60 \
    --continue_train \
    --which_epoch 50 \
    --use_generated_data_file \
    --gcn_feature_dim 131 \
    --port 8098 \
    --image_dir './images/original_heart/' \
    --checkpoints_dir './checkpoints/original_heart/' \
