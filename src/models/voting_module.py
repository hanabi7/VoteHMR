# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Voting module: generate votes from XYZ and features of seed points.

Date: July, 2019
Author: Charles R. Qi and Or Litany
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class VotingModule(nn.Module):
    def __init__(self, vote_factor, seed_feature_dim):
        """ Votes generation from seed point features.

        Args:
            vote_facotr: int
                number of votes generated from each seed point
            seed_feature_dim: int
                number of channels of seed point features
            vote_feature_dim: int
                number of channels of vote features
        """
        super(VotingModule, self).__init__()
        self.vote_factor = vote_factor
        self.in_dim = seed_feature_dim
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)

        self.conv3 = torch.nn.Conv1d(self.in_dim, 3 + self.out_dim + 24, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        
    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        # print("the number of seed:", num_seed)
        # num_vote = num_seed*self.vote_factor
        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net)
        # (batch_size, 3+1+out_dim, num_seed)
        net = net.transpose(2, 1).view(batch_size, num_seed, 3 + self.out_dim + 24)
        offset = net[:, :, 0:3].view(batch_size, num_seed, 3)
        classify_score = net[:, :, 3:24+3]
        residual_features = net[:, :, 24+3:]  # (batch_size, num_seed, out_dim)
        vote_xyz = seed_xyz + offset
        vote_features = seed_features.transpose(2, 1) + residual_features
        # vote_features is a tensor of size [bs, num_seed, out_dim]
        vote_features = vote_features.contiguous().view(batch_size, num_seed, self.out_dim)
        vote_features = vote_features.transpose(2, 1).contiguous()
        # vote_features is a tensor of size [bs, out_dim, num_seed]

        return vote_xyz, vote_features, offset, classify_score
