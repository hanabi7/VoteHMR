import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys


class SelfAttention(nn.Module):
    def __init__(self, opt):
        super(SelfAttention, self).__init__()
        self.gcn_feature_dim = opt.gcn_feature_dim
        self.smpl_key_points_number = opt.smpl_key_points_number
        self.n_heads = self.smpl_key_points_number
        self.softmax = nn.Softmax(dim=-1)
        self.feature_projection = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=1, stride=1)
        )

    def forward(self, key_point_features):
        """
        self_attention module to cluster the vote position and features
        :param end_points:
                vote_xyz: [bs, 24, 3]
                vote_features: [bs, 128, 24]
        :return:
        """
        features = key_point_features
        batch_size, num_key_points, _ = features.shape
        # the features self_attention branch
        query = features.view(batch_size, num_key_points, 1, -1)
        weights = self.feature_projection(query).view(batch_size, 24, -1)
        features = features.permute(0, 2, 1).contiguous()
        # [bs, 131, 24]
        attention = torch.matmul(weights, features)
        # [bs, 24, 24]
        features = features.permute(0, 2, 1)
        # [bs, 24, 131]
        attention = self.softmax(attention)
        features = torch.matmul(attention, features)
        return features

