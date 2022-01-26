# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F
import os
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from proposal_module import ProposalModule
from gcn import GCN
from src.utils.graph import Graph, normalize_undigraph, normalize_digraph
from src.utils.smpl_utils import smpl_structure
import deepdish
from scipy.io import loadmat
from src.models.self_attention import SelfAttention
from src.models.refine_attention import GlobalAttention
from src.utils.segmentation_generator import Segmentation


class VoteNet(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """
    def __init__(self, opt):
        super(VoteNet, self).__init__()
        # self.num_class = num_class
        # self.num_heading_bin = num_heading_bin
        self.num_size_cluster = opt.num_size_cluster
        self.use_surface_points = opt.use_surface_points
        self.gcn_feature_dim = 131
        if self.use_surface_points:
            self.gcn_feature_dim = 259
        self.global_pose_dim = opt.global_pose_dim
        # self.mean_size_arr = mean_size_arr
        # assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = opt.input_feature_dim
        self.num_proposal = opt.num_proposal
        self.vote_factor = opt.vote_factor
        self.use_ball_segment = opt.use_ball_segment
        self.use_segment = opt.use_segment
        self.use_partial_conv = opt.use_partial_conv
        self.sampling = opt.sampling
        self.use_refine_attention = opt.use_refine_attention
        self.use_votenet_proposal = opt.use_votenet_proposal
        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone()
        self.shape_params_dim = opt.shape_params_dim
        self.down_sample_file = opt.down_sample_file
        self.use_no_global_edge_conv = opt.use_no_global_edge_conv
        self.use_no_completion = opt.use_no_completion
        self.use_no_gcn = opt.use_no_gcn
        if self.use_no_global_edge_conv:
            self.use_refine_attention = False
        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 128)
        # self attention module
        # Vote aggregation and detection
        self.pnet = ProposalModule(opt, self.sampling, self.use_segment)
        self.isTrain = opt.isTrain
        self.smpl_parents = smpl_structure('smpl_parents')
        # smpl_parents is a tensor of size [2, 24]
        self.smpl_children_tree = [[idx for idx, val in enumerate(self.smpl_parents[0]) if val == i] for i in range(24)]
        smpl_chains = []
        for i in range(24):
            chain = [i]
            if i == 0:
                smpl_chains.append(chain)
                continue
            p_i = i
            for j in range(24):
                p_i = self.smpl_parents[0][p_i]
                chain.append(p_i)
                if p_i == 0:
                    smpl_chains.append(chain)
                    break
        self.smpl_chains = smpl_chains
        self.pose_regressors = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.gcn_feature_dim * 1 * 24, 9 * 24, kernel_size=1, groups=24)
            )
        # rotation2position matrix
        r2p_A = np.zeros((24, 24))
        for i in range(24):
            r2p_A[i, self.smpl_chains[i]] = 1
            r2p_A[i, i] = 0
        r2p_A = normalize_digraph(r2p_A, AD_mode=False)
        r2p_A = torch.from_numpy(r2p_A).float().unsqueeze(0)
        self.register_buffer('r2p_A', r2p_A)
        p2r_A = np.zeros((24, 24))
        for i in range(24):
            p2r_A[i, self.smpl_children_tree[i]] = 1
            p2r_A[i, self.smpl_parents[0][i]] = 1
            p2r_A[i, i] = 1
        p2r_A = normalize_digraph(p2r_A, AD_mode=False)
        p2r_A = torch.from_numpy(p2r_A).float().unsqueeze(0)
        self.register_buffer('p2r_A', p2r_A)
        # print('the shape of the p2r_A', self.p2r_A.shape)
        # shape_A matrix
        shape_A = np.zeros((24, 24))
        for i in range(24):
            shape_A[i, self.smpl_children_tree[i]] = 1
            shape_A[i, self.smpl_parents[0][i]] = 1
            shape_A[i, i] = 1
        shape_A = normalize_digraph(shape_A, AD_mode=False)
        shape_A = torch.from_numpy(shape_A).float().unsqueeze(0)
        self.register_buffer('shape_A', shape_A)
        self.coord_regressors = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.gcn_feature_dim*24, 3*24, kernel_size=1, groups=24)
        )
        self.p2r_gcn = GCN(self.gcn_feature_dim, self.gcn_feature_dim, self.gcn_feature_dim, num_layers=1, num_nodes=p2r_A.shape[1], normalize=False)
        # the shape prediction branch
        if self.use_no_global_edge_conv:
            self.shape_mlp = nn.Linear(
                self.gcn_feature_dim*24,
                10 + 9
            )
        else:
            self.attention_module = GlobalAttention(opt)
            if self.use_refine_attention:
                self.edge_conv = nn.Sequential(
                    nn.Conv2d(self.gcn_feature_dim*2, self.gcn_feature_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.gcn_feature_dim),
                    nn.ReLU(inplace=True)
                )
                self.weights_init(self.edge_conv)
                self.shape_mlp = nn.Linear(self.gcn_feature_dim*24, 10 + 9)
            else:
                self.shape_mlp = nn.Linear(self.gcn_feature_dim, 10 + 9)
        if not self.use_no_completion:
            self.pos_mlp = nn.Sequential(
                nn.Linear(self.gcn_feature_dim*24, self.gcn_feature_dim*24),
                nn.ReLU(inplace=True),
                nn.Linear(self.gcn_feature_dim*24, self.gcn_feature_dim*24),
                nn.ReLU(inplace=True)
            )
            self.weights_init(self.pos_mlp)
        self.weights_init(self.shape_mlp)

    def weights_init(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def forward(self, inputs, gt_segment):
        """ Forward pass of the network
        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs.shape[0]
        end_points = self.backbone_net(inputs, end_points)
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp4_xyz']
        # [bs, 2500, 3]
        features = end_points['fp4_features']
        # [bs, 2500, 128]
        end_points['concatenate_features'] = torch.cat((end_points['fp4_xyz'], end_points['fp4_features'].permute(0, 2, 1)), dim=2)
        xyz, features, offsets, classify_score = self.vgen(xyz, features)
        # xyz, features means the vote_xyz, and vote_features
        end_points['vote_xyz'] = xyz
        # vote_xyz is a tensor of size [batch_size, 2500, 3]
        end_points['vote_features'] = features
        # [bs, 128, 2500]
        end_points['pred_segmentation'] = classify_score
        # [bs, 2500, 24]
        pred_segmentation = torch.topk(end_points['pred_segmentation'], 1)[1].squeeze(1)
        pred_segmentation = pred_segmentation.view(batch_size, 1000)
        end_points = self.pnet(xyz, features, gt_segment, end_points)
        # end_points = self.pnet(xyz, features, pred_segmentation, end_points)
        # key_point_features [bs, 24, 131]
        key_points = end_points['pred_joints'].cuda()
        # key_point_features = end_points['pred_features'].cuda()
        concatenate_features = end_points['concatenate_features'].cuda()
        global_features = end_points['global_features'].cuda()
        # joints_segment is a tensor of size [batch_size, 24, 1]
        # --------- GRAPH CONVOLUTION ---------
        pose_feature_branch = concatenate_features
        pose_feature_branch = pose_feature_branch.view(batch_size, -1)
        if self.use_no_completion:
            key_points_feature_refined = pose_feature_branch
        else:
            key_points_feature_refined = self.pos_mlp(pose_feature_branch)
        key_points_feature_refined = key_points_feature_refined.view(batch_size, -1, 1, 1)
        # key_points_feature_refined is a tensor of size [batch_size, 24 * 131, 1, 1]
        key_points_refined = self.coord_regressors(key_points_feature_refined)
        key_points_refined = key_points_refined.view(batch_size, 24, -1)
        end_points['refined_joints'] = key_points_refined
        key_points_feature_refined = key_points_feature_refined.view(batch_size, 24, self.gcn_feature_dim)
        # print('the shape of the key_points_feature_refined:', key_points_feature_refined.shape)
        shape_feature_branch = key_points_feature_refined
        if self.use_no_gcn:
            pose_points_feature = key_points_feature_refined.view(batch_size, 24*self.gcn_feature_dim, 1, 1)
        else:
            pose_points_feature = self.p2r_gcn(key_points_feature_refined, self.p2r_A[0])
            pose_points_feature = pose_points_feature.view(batch_size, 24*self.gcn_feature_dim, 1, 1)
        smpl_pose = self.pose_regressors(pose_points_feature)
        smpl_pose = smpl_pose.view(batch_size, -1)
        # --------- Shape Prediction Convolution ---------
        if self.use_no_global_edge_conv:
            shape_feature_branch = shape_feature_branch.view(batch_size, -1)
            # [batch_size, 24*131]
            global_param = self.shape_mlp(shape_feature_branch)
        else:
            attention_feature, attention = self.attention_module(shape_feature_branch, global_features, shape_feature_branch)
            end_points['attention_feature'] = attention_feature
            if self.use_refine_attention:
                attention_feature = attention_feature.view(batch_size, 1, self.gcn_feature_dim).repeat(1, 24, 1)
                global_branch_feature = shape_feature_branch - attention_feature
                global_branch_feature = torch.cat((global_branch_feature, attention_feature), dim=2)
                # [batch_size, 24, 256]
                global_branch_feature = global_branch_feature.contiguous().view(batch_size, 24, 2*self.gcn_feature_dim, 1).permute(0, 2, 1, 3)
                global_branch_feature = self.edge_conv(global_branch_feature)
                global_branch_feature = global_branch_feature.view(batch_size, -1)
            else:
                global_branch_feature = F.relu(attention_feature)
            # shape_branch_feature is a tensor of size [batch_size, ]
            global_param = self.shape_mlp(global_branch_feature)
        smpl_shape = global_param[:, :self.shape_params_dim]
        global_pose = global_param[:, self.shape_params_dim:]
        smpl_pose = smpl_pose[:, self.global_pose_dim:]
        # print(smpl_shape)
        smpl_parameters = torch.cat((smpl_shape, global_pose, smpl_pose), dim=1)
        # print(smpl_parameters[0, :])

        end_points['pred_param'] = smpl_parameters

        return end_points
