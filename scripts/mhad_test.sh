batch_size=20

CUDA_VISIBLE_DEVICES=7 python3 /data3/data_backup/liuguanze/depth_point_cloud/src/mhad_test.py \
    --batchSize $batch_size \
    --mhad_path '/data/liuguanze/datasets/mhad/' \
    --which_epoch 60 \
    --use_downsample_evaluate \
    --evaluate_dir './evaluate/mhad/' \
    --use_refine_attention \
    --checkpoints_dir './checkpoints/normalize_noise_surreal/' \
    --image_dir './tests/mhad/' 
