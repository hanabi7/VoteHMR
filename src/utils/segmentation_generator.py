import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import torch
import pickle


class Segmentation():
    def __init__(self, opt):
        # smpl model initialize
        self.original_smpl_female_filename = opt.original_smpl_female_filename
        self.smpl_key_points_number = opt.smpl_key_points_number
        with open(self.original_smpl_female_filename, 'rb') as f:
            self.params = pickle.load(f, encoding='latin1')

    def square_distance(self, src, dst):
        """
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
    	     = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # 2*(xn * xm + yn * ym + zn * zm)
        dist += torch.sum(src ** 2, -1).view(B, N, 1)  # xn*xn + yn*yn + zn*zn
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # xm*xm + ym*ym + zm*zm
        return dist

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

    def query_ball_point(self, radius, nsample, xyz, new_xyz):
        """
        Input:
            radius: local region radius
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        # sqrdists: [B, S, N] 记录中心点与所有点之间的欧几里德距离
        sqrdists = self.square_distance(new_xyz, xyz)
        # 找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
        group_idx[sqrdists > radius ** 2] = N
        # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前nsample个点
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
        # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        # 找到group_idx中值等于N的点
        mask = group_idx == N
        # 将这些点的值替换为第一个点的值
        group_idx[mask] = group_first[mask]
        return group_idx

    def get_vertices_bound_to_jnts(self, skinning_weights, jnts):
        weights_of_interest = skinning_weights[:, jnts]
        return np.where(weights_of_interest > 0.5)

    def part_segmentation(self):
        weights = np.asarray(self.params['weights'])
        num_points = 6890
        labels = torch.zeros(num_points)
        for i in range(24):
            part_indices = self.get_vertices_bound_to_jnts(weights, i)
            part_indices = part_indices[0]
            # part indices is a tensor of size (num_selected, )
            part_indices = torch.from_numpy(part_indices)
            labels[part_indices] = i
        return labels

    def num_seed_sample(self, seed_inds):
        """
        Input:
            seed_inds: [bs, num_seed] 0, ..., 6890
            labels: [6890]
        Return:
            seed_labels: [bs, num_seed]
        """
        seed_inds = seed_inds.long()
        labels = self.part_segmentation()
        seed_labels = labels[seed_inds]
        return seed_labels

    def segmentation_part(self, key_points, vote_xyz, vote_feature, labels, num_sample=50):
        """
        Input:
            vote_xyz: [bs, 6890, 24, 3]
            vote_feature: [bs, 128, 6890]
            labels: [6890]
            key_points: [bs, 24, 3]
            num_sample: num_points sampled for ball query
        Return:
            seg_xyz: [bs, 24, num_sample, 3]
            seg_feature: [bs, 24, 256, num_sample]
        """
        sum = 0
        batch_size, num_seed, _, _ = vote_xyz.shape
        seg_xyz = torch.zeros(batch_size, 24, num_sample, 3)
        seg_feature = torch.zeros(batch_size, 24, 128, num_sample)
        vote_feature = vote_feature.permute(0, 2, 1)
        # vote_feature [bs, num_seed, 256]
        for i in range(self.smpl_key_points_number):
            # print(i)
            part_sample_index = (labels == i)
            # [6890]
            part_vote_xyz = vote_xyz[:, :, i, :]
            # [bs, 6890, 3]
            part_sample_xyz = part_vote_xyz[:, part_sample_index, :]
            # [bs, num_part_sample, 3]
            part_sample_feature = vote_feature[:, part_sample_index, :]
            # [bs, num_part_sample, 256]
            # print('num of sampled points:', part_sample_feature.shape[1])
            part_key_point = key_points[:, i, :].view(batch_size, 1, 3)
            part_index = self.query_ball_point(30, num_sample, part_sample_xyz, part_key_point)
            part_index = part_index.view(batch_size, num_sample)
            ball_query_sample_xyz = self.index_points(part_sample_xyz, part_index)
            # print('the shape of the ball_query_sample_xyz:', ball_query_sample_xyz.shape)
            # [bs, num_sample, 3]
            ball_query_sample_feature = self.index_points(part_sample_feature, part_index)
            # [bs, num_sample, 256]
            ball_query_sample_feature = ball_query_sample_feature.permute(0, 2, 1)
            # ball_query_sample_feature is a tensor of size[bs, 256, num_sample]
            seg_xyz[:, i, :, :] = ball_query_sample_xyz
            seg_feature[:, i, :, :] = ball_query_sample_feature
        return seg_xyz, seg_feature
