batch_size=42

CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch \
    --nproc_per_node=2 --master_port 29502 /data1/liuguanze/human_point_cloud/src/train_dist.py \
    --dist \
    --print_freq 10880 \
    --batchSize $batch_size --isTrain \
    --surreal_use_male --surreal_use_female \
    --lr_e 1e-3 \
    --loss_3d_weight_before 1 \
    --loss_3d_weight_after 1 \
    --loss_offset_weight 0.5 \
    --loss_smpl_weight 1 \
    --loss_vertex_weight 10 \
    --loss_segment_weight 10 \
    --save_epoch_freq 5 \
    --total_epoch 60 \
    --use_generated_data_file \
    --gcn_feature_dim 131 \
    --port 8098 \
    --image_dir './images/original_heart/' \
    --checkpoints_dir './checkpoints/original_heart/' \
