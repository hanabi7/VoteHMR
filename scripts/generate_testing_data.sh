surreal_data_path='/the/path/to/the/download/surreal/data/'
surreal_save_path='/the/path/to/save/the/processed/data/'
dfaust_save_path='/the/path/to/save/the/processed/dfaust/data/'

python3 /data1/liuguanze/human_point_cloud/src/datasets/surreal_depth_image.py \
    --surreal_save_path $surreal_save_path \
    --surreal_dataset_path $surreal_data_path


python3 /data1/liuguanze/human_point_cloud/src/datasets/dfaust_render.py \
    --output_dir $dfaust_save_path 