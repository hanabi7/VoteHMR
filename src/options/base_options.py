from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--smpl_key_points_number', type=int, default=24, help='How many keypoints on smpl model.')
        self.parser.add_argument('--sample_points_number1', type=int, default=6890, help='N1 sampled points')
        self.parser.add_argument('--sample_points_number2', type=int, default=100, help='N2 sampled points')

        self.parser.add_argument('--isTrain', action='store_true', help='isTrain')
        self.parser.add_argument('--batchSize', type=int, default=80, help='isTrain')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')

        self.parser.add_argument('--pose_params_dim', type=int, default=72)
        self.parser.add_argument('--global_pose_dim', type=int, default=9)
        self.parser.add_argument('--shape_params_dim', type=int, default=10)
        self.parser.add_argument('--total_params_dim', type=int, default=82)
        self.parser.add_argument('--neutrMosh_path', type=str, default='/data1/liuguanze/neutrMosh/')
        self.parser.add_argument('--dyna_annotation_path', type=str, default='/data/liuguanze/datasets/dyna_datasets/',
                                 help='the annotation path of the dyna datasets')
        self.parser.add_argument('--dyna_use_male', action='store_true', help='whether to use the male model.')
        self.parser.add_argument('--dyna_use_female', action='store_true', help='whether to use the female model.')
        self.parser.add_argument('--surreal_annotation_path', type=str, default='/data/liuguanze/datasets/SURREAL/smpl_data/',
                                 help='the annotation path of the surreal datasets')
        self.parser.add_argument('--male_beta_stds', type=str, default='male_beta_stds.npy')
        self.parser.add_argument('--female_beta_stds', type=str, default='female_beta_stds.npy')
        self.parser.add_argument('--surreal_smpl_path', type=str, default='smpl_data.npz')
        self.parser.add_argument('--surreal_use_male', action='store_true')
        self.parser.add_argument('--surreal_use_female',action='store_true')
        self.parser.add_argument('--smpl_model_filename', type=str,
                                 default='/data1/liuguanze/depth_point_cloud/data/smpl_cocoplus_neutral_no_chumpy.pkl')
        self.parser.add_argument('--smpl_female_filename', type=str,
                                 default='/data1/liuguanze/depth_point_cloud/data/female_model.pkl')
        self.parser.add_argument('--smpl_male_filename', type=str,
                                 default='/data1/liuguanze/depth_point_cloud/data/male_model.pkl')
        self.parser.add_argument('--dist', action='store_true')
        self.parser.add_argument('--local_rank', type=int, default=0)
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--input_feature_dim', type=int, default=0, help='the input feature dim for the pointnet')
        self.parser.add_argument('--vote_factor', type=int, default=24, help='Number of votes generated from each seed point.')
        self.parser.add_argument('--sampling', type=str, default='vote_fps')
        self.parser.add_argument('--num_size_cluster', type=int, default=10)
        self.parser.add_argument('--num_proposal', type=int, default=24)
        # model_options
        self.parser.add_argument('--use_residual_mlp', action='store_true',
                                 help='whether to use residual mlp.')
        self.parser.add_argument('--use_virtual_nodes', action='store_true',
                                 help='whether to use the virtual nodes.')
        self.parser.add_argument('--num_virtual_nodes', type=int, default=6,
                                 help='numbers of the virtual nodes.')
        self.parser.add_argument('--use_dual_features', action='store_true',
                                 help='whether to use the dual features version.')
        self.parser.add_argument('--use_surface_points', action='store_true',
                                 help='whether to use the surface points version.')
        self.parser.add_argument('--surface_points_number', type=int, default=16)
        self.parser.add_argument('--use_graph_aggregate', action='store_true',
                                 help='whether to use graph aggregate.')
        self.parser.add_argument('--use_mesh_attention', action='store_true')
        self.parser.add_argument('--neighbor_number', type=int, default=10)
        self.parser.add_argument('--use_partial_conv', action='store_true')
        self.parser.add_argument('--down_sample_file', type=str,
                                 default='/data1/liuguanze/depth_point_cloud/data/down_sample.npy')
        # self.parser.add_argument('--voting_method', action='store_true',
        #                          help='generate 24 votes for each seed_points')
        self.parser.add_argument('--image_dir', type=str, default='./image',
                                 help='where to restore all the visualized point clouds')
        self.parser.add_argument('--gcn_feature_dim', type=int, default=131,
                                 help='the input feature dim of the gcn model input')
        self.parser.add_argument('--uv_processed_path', type=str,
                                 default='/data1/liuguanze/depth_point_cloud/data/UV_Processed.mat')
        self.parser.add_argument('--UV_symmetry_filename', type=str,
                                 default='/data1/liuguanze/depth_point_cloud/data/UV_symmetry_transforms.mat')
        self.parser.add_argument('--mean_param_file', type=str,
                                 default='/data1/liuguanze/depth_point_cloud/data/neutral_smpl_mean_params.h5', help='path of smpl face')
        self.parser.add_argument('--use_self_attention', action='store_true',
                                 help='whether to use the self attention module.')
        self.parser.add_argument('--use_refine_gcn', action='store_true',
                                 help='whether to use refined gcn module.')
        self.parser.add_argument('--use_ball_segment', action='store_true',
                                 help='whether to use ball segment')
        self.parser.add_argument('--has_direction_loss', action='store_true',
                                 help='whether ot use the direction loss')
        self.parser.add_argument('--use_segment', action='store_true')
        self.parser.add_argument('--use_part_drropout', action='store_true',
                                 help='whether to use the part_drop.')
        self.parser.add_argument('--use_no_voting', action='store_true',
                                 help='whether to use voting as a mean of cluster.')
        self.parser.add_argument('--use_no_global_edge_conv', action='store_true',
                                 help='whether to use the global edge conv module.')
        self.parser.add_argument('--use_no_completion', action='store_true',
                                 help='whether to use completion module')
        self.parser.add_argument('--use_no_gcn', action='store_true')
        self.parser.add_argument('--use_votenet_proposal', action='store_true')
        self.parser.add_argument('--use_adversarial_train', action='store_true')
        self.parser.add_argument('--use_downsample_evaluate', action='store_true')
        self.parser.add_argument('--use_refine_attention', action='store_true')
        self.parser.add_argument('--dropout_rate', type=float, default=0.3)
        self.parser.add_argument('--use_conditional_gan', action='store_true',
                                 help='whether to use the conditional gan.')
        # point cloud generate settings
        self.parser.add_argument('--noise_mean', type=float, default=0.1)
        self.parser.add_argument('--noise_var', type=float, default=0.1)

        # data sets settings
        self.parser.add_argument('--surreal_dataset_path', type=str,
                                 default='/data/liuguanze/datasets/surreal/SURREAL/data/cmu/')
        self.parser.add_argument('--surreal_save_path', type=str,
                                 default='/data/liuguanze/datasets/')
        self.parser.add_argument('--mosh_data_path', type=str,
                                 default='/data1/liuguanze/neutrMosh/')
        self.parser.add_argument('--mosh_save_path', type=str,
                                 default='/data1/liuguanze/neutrMosh/')
        self.parser.add_argument('--use_surreal', action='store_true')
        self.parser.add_argument('--surreal_train_sequence_number', type=int, default=10000)
        self.parser.add_argument('--surreal_train_sequence_sample_number', type=int, default=20)
        self.parser.add_argument('--surreal_test_sequence_number', type=int, default=100)
        self.parser.add_argument('--surreal_test_sequence_sample_number', type=int, default=100)
        self.parser.add_argument('--use_generated_data_file', action='store_true',
                                 help='whether to use the data file as datasets')
        self.parser.add_argument('--use_dfaus`t', action='store_true',
                                 help='can be used as only test_sets')
        self.parser.add_argument('--dfaust_save_path', type=str,
                                 default='/data2/liuguanze/dfaust/out/')
        self.parser.add_argument('--dfaust_data_path', type=str,
                                 default='/data2/liuguanze/dfaust/')
        self.parser.add_argument('--dfaust_depth_path', type=str,
                                 default='/data3/liuguanze/dfaust/')
        self.parser.add_argument('--use_h36m', action='store_true')
        self.parser.add_argument('--number_points_sample', type=int, default=2500)
        # pose2mesh
        self.parser.add_argument('--use_pose2mesh', action='store_true')
        self.parser.add_argument('--use_cheby_conv', action='store_true')
        self.parser.add_argument('--num_mesh_output_chan', type=int, default=3)
        self.parser.add_argument('--num_mesh_output_verts', type=int, default=1723)

        # occlusion methods
        self.parser.add_argument('--use_zero_padding', action='store_true',
                                 help='when the segment is missing, use zeros to padding the 131 dim feature')
        # whether to use the visdom visualizer
        self.parser.add_argument('--use_html', action='store_true',
                                 help='whether to use the html visualize')
        self.parser.add_argument('--server', type=str,
                                 default='http://localhost')
        self.parser.add_argument('--port', type=int, default=8097)
        self.parser.add_argument('--display_id', type=int, default=1)
        self.parser.add_argument('--display_winsize', type=int, default=256)
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=1,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--name', type=str, default='pointclouds2smpl')
        self.parser.add_argument('--env_name', type=str, default='stage')
        self.parser.add_argument('--evaluate_dir', type=str, default='./evaluate/test_name/')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        return self.opt
