import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys
import math


class GlobalAttention(nn.Module):
    def __init__(self, opt):
        super(GlobalAttention, self).__init__()
        self.use_surface_points = opt.use_surface_points
        self.gcn_feature_dim = opt.gcn_feature_dim
        if self.use_surface_points:
            self.gcn_feature_dim += 128
        self.smpl_key_points_number = opt.smpl_key_points_number
        self.n_heads = self.smpl_key_points_number
        self.use_refine_attention = opt.use_refine_attention
        self.softmax = nn.Softmax(dim=-1)
        self.feature_projection = nn.Sequential(
            nn.Linear(256, self.gcn_feature_dim),
            nn.ReLU(True),
            nn.Linear(self.gcn_feature_dim, self.gcn_feature_dim)
        )

    def forward(self, key, query, value, mask=None, dropout=None):
        # key [batch_size, 24, 131]
        # query [batch_size, 258]
        # return [batch_size, 131]
        d_k = query.size(-1)
        batch_size = query.shape[0]
        query = self.feature_projection(query)
        query = query.view(batch_size, 1, self.gcn_feature_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # print('the shape of the scores:', scores.shape)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.view(batch_size, self.gcn_feature_dim)
        return output, p_attn


class MeshAttention(nn.Module):
    def __init__(self, opt):
        super(MeshAttention, self).__init__()
        self.opt = opt
        self.gcn_feature_dim = opt.gcn_feature_dim
        self.neighbor_number = opt.neighbor_number
        self.key_embedding = nn.Sequential(
            nn.Conv1d(in_channels=self.gcn_feature_dim, out_channels=self.gcn_feature_dim, kernel_size=1, stride=1)
        )
        self.query_embedding = nn.Linear(self.gcn_feature_dim, self.gcn_feature_dim)

    def knn(self, x, k):
        """
        x: [batch_size, 3, num_points]
        k:
        """
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

    def get_graph_feature(self, feature, idx):
        batch_size, num_points, feature_dim = feature.shape
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
        idx = idx + idx_base.cuda()
        idx = idx.view(-1)
        feature = feature.view(batch_size*num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.neighbor_number, feature_dim)
        return feature

    def forward(self, end_points):
        # xyz is a tensor of size [batch_size, 2500, 3]
        # features is a tensor of size [batch_size, 128, 2500]
        # concatenate_features is a tensor of size [batch_size, 2500, 131]
        idx = self.knn(end_points['fp4_xyz'].permute(0, 2, 1), k=self.neighbor_number)
        _, feature_dim, _ = end_points['fp4_features'].shape
        graph_concatenate_features = self.get_graph_feature(end_points['concatenate_features'], idx)
        graph_features = self.get_graph_feature(end_points['fp4_features'].permute(0, 2, 1).contiguous(), idx)
        batch_size, num_points, k, gcn_feature_dim = graph_concatenate_features.shape
        # graph_features [batch_size, num_points, k, feature_dim]
        offset_graph_features = graph_concatenate_features - end_points['concatenate_features'].view(batch_size, num_points, 1, gcn_feature_dim).repeat(1, 1, k, 1)
        offset_graph_features = offset_graph_features.view(batch_size*num_points, k, gcn_feature_dim)
        # offset_graph_features [batch_size, num_points, k, feature_dim]
        graph_features = graph_features.view(batch_size*num_points, k, feature_dim).contiguous()
        query = self.query_embedding(end_points['concatenate_features'].view(batch_size*num_points, gcn_feature_dim))
        query = query.view(batch_size*num_points, 1, gcn_feature_dim)
        d_k = query.size(-1)
        key = self.key_embedding(offset_graph_features.permute(0, 2, 1))
        # query [batch_points, feature_dim] key [batch_points, feature_dim, k]
        scores = torch.matmul(query, key) / math.sqrt(d_k)
        # scores should be tensor of size [batch_points, 1, k]
        attention = F.softmax(scores, dim=-1)
        # print("the shape of the scores:", attention.shape)
        output = torch.matmul(attention, graph_features)
        output = output.view(batch_size, num_points, feature_dim).permute(0, 2, 1).contiguous()
        return output


class VoteAttention(nn.Module):
    def __init__(self):
        super(VoteAttention, self).__init__()
        self.query_embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=2500,
                out_channels=24,
                kernel_size=1,
                stride=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=24,
                out_channels=24,
                kernel_size=1,
                stride=1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seed_features):
        """
            seed_features: [batch_size, 2500, 131]
            output: [batch_size, 24, 131]
        """
        query = self.query_embedding(seed_features)
        # query is a tensor of size [batch_size, 24, 131]
        scores = torch.matmul(query, seed_features.permute(0, 2, 1))
        # scores is a tensor of size [batch_size, 24, 2500]
        attention = self.softmax(scores)
        # attention is a tensor of size [batch_size, 24, 2500]
        output = torch.matmul(attention, seed_features)
        return output, attention
