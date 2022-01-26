surreal_data_path='/the/path/to/the/download/surreal/data/'
surreal_save_path='/the/path/to/save/the/processed/data/'
dfaust_data_path='/the/path/to/the/dfaust/data/'
dfaust_save_path='/the/path/to/save/the/processed/dfaust/data/'

python3 /data1/liuguanze/human_point_cloud/src/datasets/surreal_depth.py \
    --isTrain \
    --surreal_save_path $surreal_save_path \
    --surreal_dataset_path $surreal_data_path


python3 /data1/liuguanze/human_point_cloud/src/datasets/dfaust_render.py \
    --isTrain \
    --output_dir $dfaust_save_path 