import torch
import torch.nn as nn
import argparse
import numpy as np
import torch.optim as optim
import sys, os
import random
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
root_path = os.path.split(root_path)[0]
sys.path.append(root_path)
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
from src.datasets.mhad import MHAD
from src.utils.LaplacianLoss import LaplacianLoss
from src.models.votenet import VoteNet
from src.models.smpl import SMPL
import pickle
import trimesh


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--nepoch', type=int, default=120)
parser.add_argument('--model', type=str, default='')
parser.add_argument('--laplace', type=int, default=1, help='regularize towords 0 curvature, or template curvature')
# votenet initialization
parser.add_argument('--smpl_key_points_number', type=int, default=24, help='How many keypoints on smpl model.')
parser.add_argument('--sample_points_number1', type=int, default=6890, help='N1 sampled points')
parser.add_argument('--sample_points_number2', type=int, default=100, help='N2 sampled points')

parser.add_argument('--isTrain', action='store_true', help='isTrain')
parser.add_argument('--batchSize', type=int, default=80, help='isTrain')
parser.add_argument('--serial_batches', action='store_true',
                         help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')

parser.add_argument('--pose_params_dim', type=int, default=72)
parser.add_argument('--global_pose_dim', type=int, default=9)
parser.add_argument('--shape_params_dim', type=int, default=10)
parser.add_argument('--total_params_dim', type=int, default=82)
parser.add_argument('--neutrMosh_path', type=str, default='/data1/liuguanze/neutrMosh/')
parser.add_argument('--dyna_annotation_path', type=str, default='/data/liuguanze/datasets/dyna_datasets/',
                         help='the annotation path of the dyna datasets')
parser.add_argument('--dyna_use_male', action='store_true', help='whether to use the male model.')
parser.add_argument('--dyna_use_female', action='store_true', help='whether to use the female model.')
parser.add_argument('--surreal_annotation_path', type=str, default='/data/liuguanze/datasets/SURREAL/smpl_data/',
                         help='the annotation path of the surreal datasets')
parser.add_argument('--male_beta_stds', type=str, default='male_beta_stds.npy')
parser.add_argument('--female_beta_stds', type=str, default='female_beta_stds.npy')
parser.add_argument('--surreal_smpl_path', type=str, default='smpl_data.npz')
parser.add_argument('--surreal_use_male', action='store_true')
parser.add_argument('--surreal_use_female', action='store_true')
parser.add_argument('--smpl_model_filename', type=str,
                         default='/data1/liuguanze/depth_point_cloud/data/smpl_cocoplus_neutral_no_chumpy.pkl')
parser.add_argument('--smpl_female_filename', type=str,
                         default='/data1/liuguanze/depth_point_cloud/data/female_model.pkl')
parser.add_argument('--smpl_male_filename', type=str,
                         default='/data1/liuguanze/depth_point_cloud/data/male_model.pkl')
parser.add_argument('--dist', action='store_true')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--input_feature_dim', type=int, default=0, help='the input feature dim for the pointnet')
parser.add_argument('--vote_factor', type=int, default=24, help='Number of votes generated from each seed point.')
parser.add_argument('--sampling', type=str, default='vote_fps')
parser.add_argument('--num_size_cluster', type=int, default=10)
parser.add_argument('--num_proposal', type=int, default=24)
# model_options
parser.add_argument('--use_residual_mlp', action='store_true',
                         help='whether to use residual mlp.')
parser.add_argument('--use_virtual_nodes', action='store_true',
                         help='whether to use the virtual nodes.')
parser.add_argument('--num_virtual_nodes', type=int, default=6,
                         help='numbers of the virtual nodes.')
parser.add_argument('--use_dual_features', action='store_true',
                         help='whether to use the dual features version.')
parser.add_argument('--use_surface_points', action='store_true',
                         help='whether to use the surface points version.')
parser.add_argument('--surface_points_number', type=int, default=16)
parser.add_argument('--use_graph_aggregate', action='store_true',
                         help='whether to use graph aggregate.')
parser.add_argument('--neighbor_number', type=int, default=10)
parser.add_argument('--use_partial_conv', action='store_true')
parser.add_argument('--down_sample_file', type=str,
                         default='/data1/liuguanze/depth_point_cloud/data/down_sample.npy')
# parser.add_argument('--voting_method', action='store_true',
#                          help='generate 24 votes for each seed_points')
parser.add_argument('--image_dir', type=str, default='./image',
                         help='where to restore all the visualized point clouds')
parser.add_argument('--gcn_feature_dim', type=int, default=131,
                         help='the input feature dim of the gcn model input')
parser.add_argument('--uv_processed_path', type=str,
                         default='/data1/liuguanze/depth_point_cloud/data/UV_Processed.mat')
parser.add_argument('--UV_symmetry_filename', type=str,
                         default='/data1/liuguanze/depth_point_cloud/data/UV_symmetry_transforms.mat')
parser.add_argument('--mean_param_file', type=str,
                         default='/data1/liuguanze/depth_point_cloud/data/neutral_smpl_mean_params.h5',
                         help='path of smpl face')
parser.add_argument('--use_self_attention', action='store_true',
                         help='whether to use the self attention module.')
parser.add_argument('--use_refine_gcn', action='store_true',
                         help='whether to use refined gcn module.')
parser.add_argument('--use_ball_segment', action='store_true',
                         help='whether to use ball segment')
parser.add_argument('--has_direction_loss', action='store_true',
                         help='whether ot use the direction loss')
parser.add_argument('--use_segment', action='store_true')
parser.add_argument('--use_part_drropout', action='store_true',
                         help='whether to use the part_drop.')
parser.add_argument('--use_no_voting', action='store_true',
                         help='whether to use voting as a mean of cluster.')
parser.add_argument('--use_no_global_edge_conv', action='store_true',
                         help='whether to use the global edge conv module.')
parser.add_argument('--use_no_completion', action='store_true',
                         help='whether to use completion module')
parser.add_argument('--use_no_gcn', action='store_true')
parser.add_argument('--use_votenet_proposal', action='store_true')
parser.add_argument('--use_adversarial_train', action='store_true')
parser.add_argument('--use_downsample_evaluate', action='store_true')
parser.add_argument('--use_refine_attention', action='store_true')
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--use_conditional_gan', action='store_true',
                         help='whether to use the conditional gan.')
# point cloud generate settings
parser.add_argument('--noise_mean', type=float, default=0)
parser.add_argument('--noise_var', type=float, default=0.01)
# point cloud size settings
parser.add_argument('--shift_point_size', action='store_true')
parser.add_argument('--points_desired', type=int, default=1000)
# data sets settings
parser.add_argument('--surreal_dataset_path', type=str,
                         default='/data/liuguanze/datasets/surreal/SURREAL/data/cmu/')
parser.add_argument('--surreal_save_path', type=str,
                         default='/data/liuguanze/datasets/')
parser.add_argument('--dfaust_data_path', type=str,
                         default='/data2/liuguanze/dfaust/')
parser.add_argument('--dfaust_depth_path', type=str,
                         default='/data3/liuguanze/dfaust/')
parser.add_argument('--mosh_data_path', type=str,
                         default='/data1/liuguanze/neutrMosh/')
parser.add_argument('--mosh_save_path', type=str,
                         default='/data1/liuguanze/neutrMosh/')
parser.add_argument('--use_surreal', action='store_true')
parser.add_argument('--surreal_train_sequence_number', type=int, default=10000)
parser.add_argument('--surreal_train_sequence_sample_number', type=int, default=20)
parser.add_argument('--surreal_test_sequence_number', type=int, default=100)
parser.add_argument('--surreal_test_sequence_sample_number', type=int, default=100)
parser.add_argument('--use_generated_data_file', action='store_true',
                         help='whether to use the data file as datasets')
parser.add_argument('--use_dfaust', action='store_true',
                         help='can be used as only test_sets')
parser.add_argument('--number_points_sample', type=int, default=2500)
# pose2mesh
parser.add_argument('--use_pose2mesh', action='store_true')
parser.add_argument('--use_cheby_conv', action='store_true')
parser.add_argument('--num_mesh_output_chan', type=int, default=3)
parser.add_argument('--num_mesh_output_verts', type=int, default=1723)

# occlusion methods
parser.add_argument('--use_zero_padding', action='store_true',
                         help='when the segment is missing, use zeros to padding the 131 dim feature')
# whether to use the visdom visualizer
parser.add_argument('--use_html', action='store_true',
                         help='whether to use the html visualize')
parser.add_argument('--server', type=str,
                         default='http://localhost')
parser.add_argument('--port', type=int, default=8097)
parser.add_argument('--display_id', type=int, default=1)
parser.add_argument('--display_winsize', type=int, default=256)
parser.add_argument('--display_single_pane_ncols', type=int, default=1,
                         help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--name', type=str, default='pointclouds2smpl')
parser.add_argument('--env_name', type=str, default='stage')
parser.add_argument('--evaluate_dir', type=str, default='./evaluate/test_name/')
parser.add_argument('--mesh_visualize_freq', type=int, default=10000)
parser.add_argument('--mesh_dir', type=str, default='/mesh/normalized_version/')
parser.add_argument('--mhad_path', type=str, default='/data/liuguanze/datasets/mhad/')
opt = parser.parse_args()


def vertex_loss(gt_vertices, pred_vertices):
    vertex_diff = gt_vertices - pred_vertices
    abs_diff = torch.abs(vertex_diff)
    # abs_diff is a tensor of size [bs, 6890, 3]
    loss = torch.mean(abs_diff)
    return loss

def compute_score(points, faces, target):
    score = 0
    sommet_A = points[:,faces[:, 0]]
    sommet_B = points[:,faces[:, 1]]
    sommet_C = points[:,faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] -1)
    return torch.mean(score)


chamferLoss = dist_chamfer_3D.chamfer_3DDist()
learning_rate=0.001
opt.manual_seed = random.randint(1, 10000)
random.seed(opt.manual_seed)
opt.manual_seed(opt.manual_seed)
test_dataset = MHAD()
face_list = pickle.load(open('./data/face_list.pkl', 'rb'))
faces = face_list[1]
toref = opt.laplace # regularize towards 0 or template
mesh = trimesh.load('./data/template/template.ply', process=False)
# compute target edge

def init_regul(source):
    sommet_A_source = source.vertices[source.faces[:, 0]]
    sommet_B_source = source.vertices[source.faces[:, 1]]
    sommet_C_source = source.vertices[source.faces[:, 2]]
    target = []
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    return target

target = init_regul(mesh)
target = np.array(target)
target = torch.from_numpy(target).float().cuda()
target = target.unsqueeze(1).expand(3,opt.batchSize,-1)

vertices = mesh.vertices
vertices = [vertices for i in range(opt.batchSize)]
vertices = np.array(vertices)
vertices = torch.from_numpy(vertices).cuda()
laplaceloss = LaplacianLoss(faces, vertices, toref)
data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True)
len_dataset = len(data_loader)
# create network
network = VoteNet(opt).cuda()
smpl_model = SMPL('./data/smpl_cocoplus_neutral_no_chumpy.pkl', opt.batch_size)
pretrained_checkpoints = opt.pretrained_checkpoints
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
saved_weights = torch.load(pretrained_checkpoints)
network.load_state_dict(saved_weights)
lambda_laplace = 0.005
lambda_ratio = 0.005
dir_name = './checkpoints/weakly_supervised_fine_tune/last_fine_tune_network.pth'
for epoch in range(opt.nepoch):
    if epoch == 80:
        learning_rate = learning_rate / 10.0
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    if epoch == 90:
        learning_rate = learning_rate / 10.0
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    for index, data in enumerate(data_loader):
        optimizer.zero_grad()
        point_clouds = data['red_points']
        backbone_dict = network(point_clouds)
        pred_param = backbone_dict['pred_param']
        pred_shape = pred_param[:, :10]
        pred_pose = pred_param[:, 10:]
        pred_vertices, pred_joints_last = smpl_model(pred_shape, pred_pose)
        regul = laplaceloss(pred_vertices)
        dist1, dist2, idx1, idx2 = chamferLoss(point_clouds, pred_vertices)
        chamfer_loss = torch.mean(dist1)
        regul = laplaceloss(pred_vertices)
        loss_net = chamfer_loss + lambda_laplace * regul + lambda_ratio * compute_score(pred_vertices, mesh.faces, target)
        loss_net.backward()
    print('[%d: %d/%d] test smlp loss:  %f' % (epoch, index, len_dataset / 32, loss_net.item()))
torch.save(network.state_dict(), '%s/network_last.pth' % dir_name)





