from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import shutil
import os.path as osp
from collections import OrderedDict
import deepdish
import itertools
import torch.nn.functional as F
import torch.nn as nn
import torch
import pdb
import cv2
from .smpl import batch_rodrigues
from scipy.io import loadmat


class LossUtil(object):
    def __init__(self, opt):
        self.pose_params_dim = opt.pose_params_dim
        self.shape_params_dim = opt.shape_params_dim
        self.total_params_dim = opt.total_params_dim
        self.isTrain = opt.isTrain
        if opt.dist:
            self.batchSize = opt.batchSize // torch.distributed.get_world_size()
        else:
            self.batchSize = opt.batchSize

    def _keypoint_3d_loss(self, target_keypoint, pred_keypoint):
        abs_loss = torch.abs((target_keypoint - pred_keypoint))
        weighted_loss = abs_loss
        loss = torch.mean(weighted_loss)
        return loss

    def _vertex_loss(self, pred_vertex, target_vertex):
        vertex_diff = pred_vertex - target_vertex
        abs_diff = torch.abs(vertex_diff)
        # abs_diff is a tensor of size [bs, 6890, 3]
        loss = torch.mean(abs_diff)
        return loss

    def orthogonal_loss(self, para):
        device_id = para.get_device()
        Rs_pred = para[:, 10:].contiguous().view(-1, 3, 3)
        Rs_pred_transposed = torch.transpose(Rs_pred, 2, 1)
        Rs_mm = torch.bmm(Rs_pred, Rs_pred_transposed)
        tensor_eyes = torch.eye(3).expand_as(Rs_mm).cuda(device_id)
        return F.mse_loss(Rs_mm, tensor_eyes)

    def _smpl_params_loss(self, smpl_params, pred_smpl_params):
        # smpl_params is a tensor of size [batch_size, 82]
        # pred_smpl_params is a tensor of size [batch_size, 226]
        # square loss
        batch_size, _ = smpl_params.shape
        gt_shape = smpl_params[:, :self.shape_params_dim]
        gt_pose = smpl_params[:, self.shape_params_dim:]
        reshape_pose = gt_pose.contiguous().view(self.batchSize*24, -1)
        gt_pose_rodrigues = batch_rodrigues(reshape_pose).view(self.batchSize, -1)
        gt_param = torch.cat([gt_shape, gt_pose_rodrigues], dim=1)
        params_diff = gt_param - pred_smpl_params
        square_loss = torch.mul(params_diff, params_diff)
        # square_loss = square_loss * smpl_params_weight

        loss = torch.mean(square_loss)
        return loss

    def orthogonal_loss(self, para):
        device_id = para.get_device()
        Rs_pred = para[:, 10:].contiguous().view(-1, 3, 3)
        Rs_pred_transposed = torch.transpose(Rs_pred, 1, 2)
        Rs_mm = torch.bmm(Rs_pred, Rs_pred_transposed)
        tensor_eyes = torch.eye(3).expand_as(Rs_mm).cuda(device_id)
        return F.mse_loss(Rs_mm, tensor_eyes)

    def of_l1_loss(
            self, vote_xyz, smpl_joints, labels=None
    ):
        """
        Inputs
            vote_xyz:[bs, 2500, 3]
            smpl_joints:[bs, n_kpts, 3]
            labels: [bs, 2500] 0, ..., 23
            gt_xyz: [bs, 2500, 3]
        """
        if labels is not None:
            bs, num_seed, c = vote_xyz.size()
            labels = labels.long()
            gt_xyz = torch.zeros(bs, num_seed, 3).cuda()
            for i in range(bs):
                part_joints = smpl_joints[i, :, :]
                part_labels = labels[i, :]
                gt_xyz[i, :, :] = part_joints[part_labels, :]
            # [batch_size, batch_size, 24, 3]
            # [bs, 2500, 3]
            diff = vote_xyz - gt_xyz
            abs_diff = torch.abs(diff)
            in_loss = abs_diff
            in_loss = torch.sum(
                in_loss.view(bs, -1), 1
            )
            in_loss = torch.mean(in_loss)
        else:
            bs, num_seed, num_keypoints, c = vote_xyz.size()
            smpl_joints = smpl_joints.view(bs, 1, num_keypoints, 3)
            smpl_joints = smpl_joints.repeat(1, num_seed, 1, 1)
            smpl_joints = smpl_joints.permute(0, 2, 1, 3).contiguous()
            # smpl_joints is a tensor of size [bs, num_keypoints, num_seed, 3]
            vote_xyz = vote_xyz.view(bs, num_seed, num_keypoints, 3)
            vote_xyz = vote_xyz.permute(0, 2, 1, 3).contiguous()
            diff = vote_xyz - smpl_joints
            abs_diff = torch.abs(diff)
            # abs_diff is a tensor of size [bs, n_kpts, n_pts, 3]
            in_loss = abs_diff
            in_loss = torch.sum(
                in_loss.view(bs, -1), 1
            )
            in_loss = torch.mean(in_loss)
        return in_loss

    def _segmentation_loss(self, class_score, gt_label):
        """
        Input:
            class_score: [bs, 2500, 24]
            gt_label: [bs, 2500]
        Return: loss
        """
        entropy = nn.CrossEntropyLoss()
        batch_size, num_points, C = class_score.shape
        input = class_score.contiguous().view(-1, C)
        # print(input) cuda error
        gt_labels = gt_label.view(-1).long().cuda()
        loss = entropy(input, gt_labels)
        return loss

    def _offset_dir_loss(self, gt_offsets, pt_offsets):
        """
        :param gt_offsets:[bs, 6890, 3]
        :param pt_offsets:[bs, 6890, 3]
        :return:
        """
        gt_offsets = gt_offsets.contiguous().view(-1, 3)
        pt_offsets = pt_offsets.contiguous().view(-1, 3)
        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=-1)
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=-1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff)
        return offset_dir_loss

    def adversarial_loss(self, real_result, fake_result):
        m = nn.Sigmoid()
        criterion = nn.BCELoss()
        batch_size = real_result.shape[0]
        real_label = torch.ones(batch_size).cuda()
        fake_label = torch.zeros(batch_size).cuda()
        # real_label [batch_size]
        real_loss = criterion(m(real_result), real_label)
        fake_loss = criterion(m(fake_result), fake_label)
        loss = real_loss + fake_loss
        return loss

    def generate_loss(self, real_result):
        m = nn.Sigmoid()
        criterion = nn.BCELoss()
        batch_size = real_result.shape[0]
        real_label = torch.ones(batch_size).cuda()
        real_loss = criterion(m(real_result), real_label)
        return real_loss
