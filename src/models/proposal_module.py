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

    def forward(self, xyz, features, labels, end_points):
        """
        Args:
            xyz: [batch_size, 2500, 3]
            features: [batch_size, 128, 2500]
            labels: [bs, 2500, 24]
        Returns:
            end_points: pred_joints: [batch_size, num_key_points, 3]
                        pred_features: [batch_size, num_key_points, 131]
        """
        labels = labels.permute(0, 2, 1)
        joints = torch.matmul(labels, xyz)
        joints_features = torch.matmul(labels, features)
        end_points['pred_joints'] = joints
        end_points['pred_features'] = joints_features
        joints_features = torch.cat((joints, joints_features), dim=2)
        end_points['concatenate_features'] = joints_features

        return end_points

