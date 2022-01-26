batch_size=1
surreal_save_path='/the/path/to/the/processed/surreal/data/'

CUDA_VISIBLE_DEVICES=7 python /data1/liuguanze/human_point_cloud/src/train_dist.py \
    --print_freq 200 \
    --batchSize $batch_size --isTrain \
    --dyna_use_male --dyna_use_female \
    --surreal_use_male --surreal_use_female \
    --lr_e 5e-4 \
    --loss_3d_weight_before 2000 \
    --loss_3d_weight_after 1000 \
    --loss_smpl_weight 1000 \
    --loss_offset_weight 1 \
    --loss_segment_weight 100 \
    --loss_vertex_weight 1000 \
    --evaluate_epoch 10 \
    --save_epoch_freq 5 \
    --total_epoch 50 \
    --surreal_save_path $surreal_save_path \
    --gcn_feature_dim 131 \
    --port 8099 \
    --use_generated_data_file \
    --image_dir './images/model_check/' \
    --checkpoints_dir './checkpoints/model_check/' \
