import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule


class Pointnet2Backbone(nn.Module):
    """
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0):
        super(Pointnet2Backbone, self).__init__()
        self.sa2 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.4,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.sa3 = PointnetSAModuleVotes(
            npoint=256,
            radius=0.8,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.sa4 = PointnetSAModuleVotes(
            npoint=128,
            radius=1.2,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.sa5 = PointnetSAModuleVotes(
            npoint=64,
            radius=1.2,
            nsample=16,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.fp0 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp1 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256 + 256, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[256 + 128, 128, 128])
        self.fp4 = PointnetFPModule(mlp=[128, 128, 128])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        """
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]
        xyz, features = self._break_up_pc(pointcloud)
        end_points['sa0_xyz'] = xyz
        end_points['sa0_features'] = features
        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,1023
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features
        xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features
        xyz, features, fps_inds = self.sa4(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features
        # sa4_features is a tensor of size [batch_size, num_points, features_dim]
        xyz, features, fps_inds = self.sa5(xyz, features)
        end_points['sa5_xyz'] = xyz
        end_points['sa5_features'] = features
        # --------- 4 FEATURE UPSAMPLING LAYERS --------
        features = self.fp0(end_points['sa4_xyz'], end_points['sa5_xyz'], end_points['sa4_features'],
                            end_points['sa5_features'])
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'],
                            features)
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        features = self.fp3(end_points['sa1_xyz'], end_points['sa2_xyz'], end_points['sa1_features'], features)
        features = self.fp4(end_points['sa0_xyz'], end_points['sa1_xyz'], end_points['sa0_features'], features)
        end_points['fp4_features'] = features
        # features is a tensor of size [batch_size, 2500, 128]
        end_points['fp4_xyz'] = end_points['sa0_xyz']
        global_features = end_points['sa5_features'].permute(0, 2, 1).contiguous().view(batch_size, 256, 64, 1)
        global_features = F.max_pool2d(global_features, kernel_size=[64, 1])
        end_points['global_features'] = global_features.view(batch_size, 256)
        # indices among the entire input point clouds
        return end_points
