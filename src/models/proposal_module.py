# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pointnet2.pointnet2_modules import PointnetSAModuleVotes
import pointnet2.pointnet2_utils
import src.pointnet2.pytorch_utils as pt_utils
from src.utils.smpl_utils import smpl_structure


class ProposalModule(nn.Module):
    def __init__(self, opt, sampling, use_segment, seed_feat_dim=128):
        super(ProposalModule, self).__init__()
        self.num_proposal = 1
        self.sampling = sampling
        self.use_segment = use_segment
        self.use_surface_points = opt.use_surface_points
        if self.use_surface_points:
            self.surface_points_number = opt.surface_points_number
            """self.surface_mlp = nn.Sequential(
                nn.Conv1d(128*2, 128*2, kernel_size=1, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True)
            )"""
        self.seed_feat_dim = seed_feat_dim
        # Smpl structure
        self.dp2smpl_mapping = smpl_structure('dp2smpl_mapping')

    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, D1,...DN]
        Return:
            new_points:, indexed points data, [B, D1,...DN, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def k_nearest_neighbor(self, xyz, part_xyz, num_sample):
        """
        Inputs:
            xyz: [batch_size, num_points, 3]
            part_xyz: [batch_size, 3]
        Outputs:
            group_idx: [batch_size, num_selected] long tensor
        """
        batch_size, num_points, c = xyz.shape
        dist = xyz - part_xyz.view(batch_size, 1, 3).repeat(1, num_points, 1)
        dist = torch.norm(dist, dim=-1)
        # [batch_size, num_points]
        dist = -dist
        value, index = torch.topk(dist, num_sample, dim=1)
        # print('the shape of the index:', index.shape)
        return index

    def forward(self, xyz, features, labels, end_points):
        """
        Args:
            xyz: [batch_size, 2500, 3]
            features: [batch_size, 128, 2500]
            labels: [bs, 2500]
        Returns:
            end_points: pred_joints: [batch_size, num_key_points, 3]
                        pred_features: [batch_size, num_key_points, 131]
        """
        batch_size, num_points, c = xyz.shape
        batch_size, feature_dim, num_points = features.shape
        labels_xyz = labels.view(batch_size, num_points, 1).repeat(1, 1, c)
        labels_features = labels.view(batch_size, 1, num_points).repeat(1, feature_dim, 1)
        joints = torch.ones((batch_size, 24, c), dtype=xyz.dtype)
        if self.use_surface_points:
            joints_features = torch.zeros((batch_size, 24, 256), dtype=features.dtype)
        else:
            joints_features = torch.zeros((batch_size, 24, 128), dtype=features.dtype)
        xyz_zeros = torch.zeros_like(xyz)
        features_zeros = torch.zeros_like(features)
        for i in range(24):
            segment_mask_xyz = (labels_xyz == i)
            segment_mask = (labels == i)
            # [batch_size, num_points]
            segment_mask = torch.sum(segment_mask, dim=1).float()
            # [batch_size]
            non_zero_mask = (segment_mask != 0)
            segment_mask = torch.where(segment_mask != 0, segment_mask, torch.ones_like(segment_mask))
            # [batch_size]
            non_zero_mask_xyz = non_zero_mask.view(batch_size, 1).repeat(1, 3)
            non_zero_mask_features = non_zero_mask.view(batch_size, 1).repeat(1, 128)
            segment_mask_features = (labels_features == i)
            # segment_mask_xyz [bs, 2500, 3]
            # segment_mask_features [bs, 128, 2500]
            selected_xyz = torch.where(segment_mask_xyz, xyz, xyz_zeros)
            selected_features = torch.where(segment_mask_features, features, features_zeros)
            # [bs, num_points, 3] [bs, 128, 2500]
            sum_selected_xyz = torch.sum(selected_xyz, dim=1).permute(1, 0)
            sum_selected_features = torch.sum(selected_features, dim=2).permute(1, 0)
            part_xyz = torch.div(sum_selected_xyz, segment_mask)
            # print('the shape of the part_xyz', part_xyz.shape) [batch_size, 3]
            part_features = torch.div(sum_selected_features, segment_mask)
            # print('the shape of the part_features', part_features.shape) [batch_size, 128]
            part_xyz = part_xyz.permute(1, 0)
            part_features = part_features.permute(1, 0)
            part_xyz_zeros = torch.zeros_like(part_xyz)
            part_features_zeros = torch.zeros_like(part_features)
            part_xyz = torch.where(non_zero_mask_xyz, part_xyz, part_xyz_zeros).contiguous()
            part_features = torch.where(non_zero_mask_features, part_features, part_features_zeros).contiguous()
            if self.use_surface_points:
                surface_points_features = selected_features - part_features.view(batch_size, 128, 1).repeat(1, 1, num_points)
                surface_points_features = torch.where(segment_mask_features, surface_points_features, features_zeros)
                # [batch_size, feature_dim, num_points]
                surface_points_features = surface_points_features.max(dim=-1, keepdim=False)[0]
                # [batch_size, feature_dim*2, surface_points_number]
                part_features = torch.cat((part_features, surface_points_features), dim=-1)
                # part_features for surface_points_version [batch_size, 256]
            joints[:, i, :] = part_xyz
            joints_features[:, i, :] = part_features
        end_points['pred_joints'] = joints
        end_points['pred_features'] = joints_features
        joints_features = torch.cat((joints, joints_features), dim=2)
        end_points['concatenate_features'] = joints_features

        return end_points

