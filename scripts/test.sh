batch_size=40

CUDA_VISIBLE_DEVICES=2 python3 /data1/liuguanze/human_point_cloud/src/test.py \
    --batchSize $batch_size \
    --dyna_use_male --dyna_use_female \
    --surreal_use_male --surreal_use_female \
    --which_epoch 50 \
    --use_generated_data_file \
    --gcn_feature_dim 131 \
    --checkpoints_dir './checkpoints/original_heart/' \
    --image_dir './tests/original_heart/' \
    --evaluate_dir './evaluate/original_heart/'
