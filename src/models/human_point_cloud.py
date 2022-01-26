import torch
import torch.nn as nn
from src.models.loss import LossUtil
from src.models.votenet import VoteNet
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import os.path as osp
import os
import shutil
import PIL
import deepdish
import numpy as np
import pickle
from src.models.smpl import SMPL, batch_skew
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.utils.visual_utils import Visualizer
from src.utils.segmentation_generator import Segmentation
from src.models.adversarial import Adversarial
from src.utils.renderer import *


class HumanPointCloud(nn.Module):
    """
    Human Regression module that regresses a smpl model from the input point cloud
    """
    def __init__(self, opt):
        super(HumanPointCloud, self).__init__()
        self.opt = opt
        self.input_channels = 3
        self.smpl_key_points_number = opt.smpl_key_points_number
        self.sample_points_number1 = opt.sample_points_number1
        self.sample_Points_number2 = opt.sample_points_number2
        self.shape_params_dim = opt.shape_params_dim
        self.pose_params_dim = opt.pose_params_dim
        self.total_params_dim = opt.total_params_dim
        self.input_feature_dim = opt.input_feature_dim
        self.use_segment = opt.use_segment
        self.use_ball_segment = opt.use_ball_segment
        self.use_adversarial_train = opt.use_adversarial_train
        self.use_downsample_evaluate = opt.use_downsample_evaluate
        self.use_no_voting = opt.use_no_voting
        self.use_conditional_gan = opt.use_conditional_gan
        if self.use_downsample_evaluate:
            self.down_sample_matrix = np.load(opt.down_sample_file)
            self.down_sample_matrix = torch.from_numpy(self.down_sample_matrix).cuda().float()
        self.votenet = VoteNet(opt).cuda()
        self.visualizer = Visualizer(opt)
        if opt.dist:
            self.votenet = DistributedDataParallel(
                self.votenet, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
            if self.use_adversarial_train or self.use_conditional_gan:
                self.adversarial_model = DistributedDataParallel(
                    self.adversarial_model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        self.loss_utils = LossUtil(opt)
        # load mean params
        self.mean_param_file = osp.join(opt.mean_param_file)
        self.has_offset_loss = True
        self.image_dir = osp.join(opt.image_dir)
        self.has_joint3d_loss_before = True
        self.has_smpl_loss = True
        self.has_vertex_loss = True
        self.has_joint3d_loss_after = False
        self.has_segment_loss = True
        self.has_joint3d_loss_last = True
        self.has_generate_loss = False
        self.has_adversarial_loss = False
        if self.use_adversarial_train or self.use_conditional_gan:
            self.has_adversarial_loss = True
            self.has_generate_loss = True
        self.has_orthogonal_loss = True
        if self.use_no_voting:
            self.has_segment_loss = False
            self.has_offset_loss = False
        self.has_direction_loss = opt.has_direction_loss
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = osp.join(opt.checkpoints_dir)
        self.Tensor = torch.cuda.FloatTensor
        self.annotation_path = opt.surreal_annotation_path
        self.smpl_male_filename = opt.smpl_male_filename
        if self.isTrain:
            self.optimizer_E = torch.optim.Adam(
                self.votenet.parameters(), lr=opt.lr_e)
            if self.use_adversarial_train or self.use_conditional_gan:
                self.optimizer_D = torch.optim.Adam(
                    self.adversarial_model.parameters(), lr=opt.lr_e
                )
            self.continue_train = opt.continue_train
        else:
            which_epoch = opt.which_epoch
            self.load_network(self.votenet, 'votenet', which_epoch)
        #model detail
        self.dist = opt.dist
        if self.dist:
            self.batchSize = opt.batchSize // torch.distributed.get_world_size()
        else:
            self.batchSize = opt.batchSize
        self.load_mean_params()
        self.smpl_model = SMPL(self.smpl_male_filename, self.batchSize)
        # iuv maps
        if self.isTrain:
            if self.continue_train:
                which_epoch = opt.which_epoch
                saved_info = self.load_info(which_epoch)
                opt.epoch_count = saved_info['epoch']
                self.optimizer_E.load_state_dict(saved_info['optimizer_E'])
                if self.use_adversarial_train or self.use_conditional_gan:
                    self.optimizer_D.load_state_dict(saved_info['optimizer_D'])
                self.load_network(self.votenet, 'votenet', which_epoch)
                if self.use_adversarial_train or self.use_conditional_gan:
                    self.load_network(self.adversarial_model, 'adv_net', which_epoch)
                if opt.process_rank <= 0:
                    print('resume from epoch {}'.format(which_epoch))
        male_model = pickle.load(open('./data/male_model.pkl', 'rb'), encoding='latin1')
        self.faces = male_model['f']

    def set_input(self, data_dict):
        if 'gt_pose' in data_dict:
            gt_pose = data_dict['gt_pose'].cuda()
        if 'gt_shape' in data_dict:
            gt_shape = data_dict['gt_shape'].cuda()
        batch_size, _ = gt_pose.shape
        reshape_pose = gt_pose.view(-1, 3)
        Rs = self.batch_rodrigues(reshape_pose).view(batch_size, -1)
        gt_vertices, J_transformed = self.smpl_model(gt_shape, Rs)
        gt_joints3d = data_dict['gt_joints3d'].cuda()
        # J_transformed is a tensor of size [batch_size, 24, 3]
        rot_pose = J_transformed[:, 0, :]
        # rot_pose is a tensor of size [batch_size, 3]
        rot_joints = gt_joints3d[:, 0, :]
        # rot_joints is a tensor of size [batch_size, 3]
        trans = rot_joints - rot_pose
        # [batch_size, 3]
        gt_pose_rodrigues = Rs.view(batch_size, -1)
        repeat_trans = trans.view(batch_size, 1, 3)
        repeat_trans_verts = repeat_trans.repeat(1, 6890, 1)
        repeat_trans_joints = repeat_trans.repeat(1, 24, 1)
        # [batch_size, 6890, 3]
        gt_vertices = gt_vertices + repeat_trans_verts
        gt_smpl_joints = J_transformed + repeat_trans_joints
        gt_param = torch.cat((gt_shape, gt_pose), dim=1)
        data_dict['gt_param'] = gt_param
        data_dict['gt_pose'] = gt_pose_rodrigues
        data_dict['gt_vertices'] = gt_vertices
        data_dict['trans'] = trans
        data_dict['gt_smpl_joints'] = gt_smpl_joints
        return data_dict

    def batch_rodrigues(self, pose_params):
        with torch.cuda.device(pose_params.get_device()):
            # pose_params shape is (bs*24, 3)
            # angle shape is (batchSize*24, 1)
            angle = torch.norm(pose_params + 1e-8, p=2, dim=1).view(-1, 1)
            # r shape is (batchSize*24, 3, 1)
            r = torch.div(pose_params, angle).view(angle.size(0), -1, 1)
            # r_T shape is (batchSize*24, 1, 3)
            r_T = r.permute(0, 2, 1)
            # cos and sin is (batchSize*24, 1, 1)
            cos = torch.cos(angle).view(angle.size(0), 1, 1)
            sin = torch.sin(angle).view(angle.size(0), 1, 1)
            # outer is (bs*24, 3, 3)
            outer = torch.matmul(r, r_T)
            eye = torch.eye(3).view(1, 3, 3)
            # eyes is (bs*24, 3, 3)
            eyes = eye.repeat(angle.size(0), 1, 1).cuda()
            # r_sk is (bs*24, 3, 3)
            r_sk = batch_skew(r, r.size(0))
            R = cos * eyes + (1 - cos) * outer + sin * r_sk
            # R shape is (bs*24, 3, 3)
            return R

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

    def load_mean_params(self):
        mean_params = np.zeros((1, self.total_params_dim))
        mean_vals = deepdish.io.load(self.mean_param_file)
        # initialize scale at 0.9
        mean_params[0, 0] = 0.9
        # set pose
        mean_pose = mean_vals['pose']
        mean_pose[:3] = 0.
        mean_pose[0] = np.pi
        mean_shape = mean_vals['shape']
        mean_params = np.hstack((mean_shape, mean_pose))
        mean_params = mean_params.reshape(1, 82)
        mean_params = np.repeat(mean_params, self.batchSize, axis=0)
        self.mean_params = torch.from_numpy(mean_params).float().cuda()
        self.mean_params.requires_grad = False

    def forward(self, data_dict):
        """
        inputs is a data dictionary
        """
        if 'gt_param' in data_dict.keys():
            gt_param = data_dict['gt_param'].cuda()
        input = data_dict['point_cloud'].float().cuda()
        if 'gt_joints3d' in data_dict.keys():
            gt_joints3d = data_dict['gt_joints3d'].cuda()
        gt_segment = data_dict['gt_segment'].cuda()
            # print('gt_segment max', gt_segment.max())
        if 'gt_vertices' in data_dict.keys():
            gt_vertices = data_dict['gt_vertices'].cuda()
        if 'gt_shape' in data_dict:
            gt_shape = data_dict['gt_shape'].cuda()
        if 'trans' in data_dict.keys():
            trans = data_dict['trans'].cuda()
        if 'gt_smpl_joints' in data_dict.keys():
            gt_smpl_joints = data_dict['gt_smpl_joints'].cuda()
        if 'gt_pose' in data_dict.keys():
            gt_pose_rodrigues = data_dict['gt_pose'].cuda()
        batch_size, _ = gt_param.shape
        backbone_dict = self.votenet(input, gt_segment)
        pred_joints = backbone_dict['pred_joints'].cuda()
        refined_joints = backbone_dict['refined_joints']
        if self.use_no_voting:
            self.vote_attention = backbone_dict['vote_attention']
            # vote_attention is a tensor of size [batch_size, 24, 2500]
            self.vote_attention = self.vote_attention.permute(0, 2, 1)
            # vote_attention is a tensor of size [batch_size, 2500, 24]
        self.pred_segmentation = backbone_dict['pred_segmentation']
        pred_param = backbone_dict['pred_param']
        if not self.use_no_voting:
            vote_xyz = backbone_dict['vote_xyz']
            self.vote_xyz = vote_xyz
        pred_param = pred_param.float().cuda()
        self.pred_joints = pred_joints
        self.point_clouds = input
        self.pred_joints = backbone_dict['pred_joints']
        self.refined_joints = backbone_dict['refined_joints']
        self.gt_vertices = gt_vertices
        self.gt_joints3d = gt_joints3d
        self.pred_vertices, self.pred_joints_last = self.return_smpl_vertices(
            pred_param[:, :self.shape_params_dim],
            pred_param[:, self.shape_params_dim:],
            trans
        )
        self.gt_smpl_joints = gt_smpl_joints
        self.gt_segment = gt_segment
        # compute adversarial model
        if self.isTrain and self.use_adversarial_train:
            real_label = self.adversarial_model(gt_pose_rodrigues[:, 9:])
            fake_label = self.adversarial_model(pred_param[:, self.shape_params_dim + 9:].detach())
            self.adversarial_loss = self.loss_utils.adversarial_loss(real_label, fake_label)
            self.generate_loss = self.loss_utils.generate_loss(fake_label)
        if self.isTrain and self.use_conditional_gan:
            global_features = backbone_dict['global_features']
            real_label = self.adversarial_model(torch.cat((gt_shape, gt_pose_rodrigues[:, ]), dim=1), global_features)
            fake_label = self.adversarial_model(pred_param.detach(), global_features)
            self.adversarial_loss = self.loss_utils.adversarial_loss(real_label, fake_label)
            self.generate_loss = self.loss_utils.generate_loss(fake_label)
        # compute loss item
        if self.has_offset_loss:
            self.of_l1_loss = self.loss_utils.of_l1_loss(vote_xyz, gt_joints3d, gt_segment)
        if self.has_joint3d_loss_before:
            self.joints3d_loss_before = self.loss_utils._keypoint_3d_loss(gt_joints3d, refined_joints)
        if self.has_smpl_loss:
            self.smpl_loss = self.loss_utils._smpl_params_loss(gt_param, pred_param)
        if self.has_vertex_loss:
            self.vertex_loss = self.loss_utils._vertex_loss(
                pred_vertex=self.pred_vertices,
                target_vertex=self.gt_vertices
            )
        if self.has_segment_loss:
            self.segment_loss = self.loss_utils._segmentation_loss(
                self.pred_segmentation, gt_segment
            )
        if self.has_joint3d_loss_last:
            self.joints3d_loss_last = self.loss_utils._keypoint_3d_loss(
                gt_smpl_joints,
                self.pred_joints_last
            )
        if self.has_orthogonal_loss:
            self.orthogonal_loss = self.loss_utils.orthogonal_loss(pred_param)
        # compute t-pose smpl verts
        if not self.isTrain:
            tpose = torch.zeros(self.batchSize, 24*3)
            tpose[:, 0] = 3.39
            reshape_tpose = tpose.view(batch_size*24, -1).cuda()
            Rs_tpose = self.batch_rodrigues(reshape_tpose).view(batch_size, -1)
            self.pred_verts_tpose, _ = self.smpl_model(pred_param[:, :self.shape_params_dim], Rs_tpose)
            gt_shape_param = gt_param[:, :self.shape_params_dim]
            self.gt_verts_tpose, _ = self.smpl_model(gt_shape_param, Rs_tpose)
            # gt_param tensor of size [batch_size, 82]
            gt_shape = gt_param[:, :self.shape_params_dim]
            gt_pose = gt_param[:, self.shape_params_dim:].contiguous().view(batch_size*24, -1).cuda()
            rs_gt_pose = self.batch_rodrigues(gt_pose).view(batch_size, -1)
            rs_predict_pose = pred_param[:, self.shape_params_dim:]
            self.pred_verts_tshape, _ = self.smpl_model(gt_shape, rs_gt_pose)
            self.gt_verts_tshape, _ = self.smpl_model(gt_shape, rs_predict_pose)
            if self.use_downsample_evaluate:
                self.pred_vertices = torch.matmul(self.down_sample_matrix, self.pred_vertices)
                self.gt_vertices = torch.matmul(self.down_sample_matrix, self.gt_vertices)

    def test(self, data):
        with torch.no_grad():
            self.forward(data)

    def basic_visualize(self, epoch, total_steps):
        # basic visualize of the point clouds, comparison, segmentation, error, joint_prediction
        if not osp.exists(self.image_dir):
            os.mkdir(self.image_dir)
        image_dir = self.image_dir + str(epoch) + '/'
        if not osp.exists(image_dir):
            os.mkdir(image_dir)
        gt_vertice = self.gt_vertices[0, :, :]
        gt_joints3d = self.gt_joints3d[0, :, :]
        point_clouds = self.point_clouds[0, :, :]
        pred_joints = self.pred_joints[0, :, :]
        refined_joints = self.refined_joints[0, :, :]
        pred_vertices = self.pred_vertices[0, :, :]
        if self.use_no_voting:
            vote_attention = self.vote_attention[0, :, :]
        else:
            vote_xyz = self.vote_xyz[0, :, :]
        labels = self.gt_segment[0, :]
        if not self.use_no_voting:
            pred_segmentation = self.pred_segmentation[0, :, :]
            pred_segmentation = torch.topk(pred_segmentation, 1)[1].squeeze(1)
        gt_smpl_joints = self.gt_smpl_joints[0, :, :]
        # dirs
        gt_dir = image_dir + '_' + str(total_steps) + 'gt_vertices.png'
        basic_dir = image_dir + '_' + str(total_steps) + 'pred_joint_comparison.png'
        refined_joints_dir = image_dir + '_' + str(total_steps) + 'refine_joint_comparison.png'
        comparison_dir = image_dir + '_' + str(total_steps) + 'body_comparison.png'
        segment_dir = image_dir + '_' + str(total_steps) + 'segment_comparison.png'
        error_dir = image_dir + '_' + str(total_steps) + 'error.png'
        check_dir = image_dir + '_' + str(total_steps) + 'gt_check.png'
        self.visualizer.point_clouds_compare(point_clouds, gt_vertice, gt_dir)
        self.visualizer.point_cloud_visualize(point_clouds, gt_joints3d, pred_joints, basic_dir)
        self.visualizer.point_cloud_visualize(point_clouds, gt_joints3d, refined_joints, refined_joints_dir)
        self.visualizer.point_clouds_compare(point_clouds, pred_vertices, comparison_dir)
        if not self.use_no_voting:
            self.visualizer.segmentation_visualize(point_clouds, labels, pred_segmentation, segment_dir)
        self.visualizer.error_visualize(pred_vertices, gt_vertice, error_dir)
        self.visualizer.key_points_check(gt_vertice, gt_smpl_joints, check_dir)
        if not self.isTrain:
            if not self.use_no_voting:
                self.get_current_votes(point_clouds, vote_xyz, labels, pred_segmentation, total_steps)
            else:
                self.get_current_attention(vote_attention, point_clouds, total_steps)
            self.get_current_mesh_errors(gt_vertice, pred_vertices, total_steps)

    def save(self, label, epoch):
        self.save_network(self.votenet, 'votenet', label)
        save_info = {'epoch': epoch,
                     'optimizer_E': self.optimizer_E.state_dict()}
        if self.use_adversarial_train or self.use_conditional_gan:
            self.save_network(self.adversarial_model, 'adv_net', label)
            save_info = {'epoch': epoch,
                         'optimizer_E': self.optimizer_E.state_dict(),
                         'optimizer_D': self.optimizer_D.state_dict()}
        self.save_info(save_info, label)

    def get_current_visuals(self, epoch, total_steps):
        image_dir = self.image_dir + str(epoch) + '/'
        # dirs
        gt_dir = image_dir + '_' + str(total_steps) + 'gt_vertices.png'
        pred_joint_comparison_dir = image_dir + '_' + str(total_steps) + 'pred_joint_comparison.png'
        refine_joint_comparison_dir = image_dir + '_' + str(total_steps) + 'refine_joint_comparison.png'
        body_comparison_dir = image_dir + '_' + str(total_steps) + 'body_comparison.png'
        error_dir = image_dir + '_' + str(total_steps) + 'error.png'
        gt_vertices = PIL.Image.open(gt_dir).convert("RGB")
        gt_vertices = np.asarray(gt_vertices)
        pred_joint_comparison = PIL.Image.open(pred_joint_comparison_dir).convert("RGB")
        pred_joint_comparison = np.asarray(pred_joint_comparison)
        refine_joint_comparison = PIL.Image.open(refine_joint_comparison_dir).convert("RGB")
        refine_joint_comparison = np.asarray(refine_joint_comparison)
        body_comparison = PIL.Image.open(body_comparison_dir).convert("RGB")
        body_comparison = np.asarray(body_comparison)
        error = PIL.Image.open(error_dir).convert("RGB")
        error = np.asarray(error)
        visuals_dict = OrderedDict([('gt_vertices', gt_vertices)])
        visuals_dict['pred_joint_comparison'] = pred_joint_comparison
        visuals_dict['refine_joint_comparison'] = refine_joint_comparison
        visuals_dict['body_comparison'] = body_comparison
        visuals_dict['error'] = error
        return visuals_dict

    def get_current_errors(self):
        loss_dict = OrderedDict()
        total_loss = 0
        if self.has_offset_loss:
            of_l1_loss = self.of_l1_loss.item()
            loss_dict['of_l1_loss'] = of_l1_loss
            total_loss += of_l1_loss
        if self.has_joint3d_loss_before:
            joints3d_loss_before = self.joints3d_loss_before.item()
            loss_dict['joints3d_loss_before'] = joints3d_loss_before
            # loss_dict = OrderedDict([('of_l1_loss', of_l1_loss)])
            total_loss += joints3d_loss_before
        if self.has_smpl_loss:
            smpl_loss = self.smpl_loss.item()
            loss_dict['smpl_loss'] = smpl_loss
            total_loss += smpl_loss
        if self.has_vertex_loss:
            vertex_loss = self.vertex_loss.item()
            loss_dict['vertex_loss'] = vertex_loss
            total_loss += vertex_loss
        if self.has_segment_loss:
            segment_loss = self.segment_loss.item()
            loss_dict['segment_loss'] = segment_loss
            total_loss += segment_loss
        if self.has_joint3d_loss_last:
            joints3d_loss_last = self.joints3d_loss_last.item()
            loss_dict['joints3d_loss_last'] = joints3d_loss_last
            total_loss += joints3d_loss_last
        if self.has_direction_loss:
            offset_dir_loss = self.offset_dir_loss.item()
            loss_dict['offset_dir_loss'] = offset_dir_loss
            total_loss += offset_dir_loss
        if self.has_orthogonal_loss:
            orthogonal_loss = self.orthogonal_loss.item()
            loss_dict['orthogonal_loss'] = orthogonal_loss
            total_loss += orthogonal_loss
        if self.has_generate_loss:
            generate_loss = self.generate_loss.item()
            loss_dict['generate_loss'] = generate_loss
            total_loss += generate_loss
        if self.has_adversarial_loss:
            adversarial_loss = self.adversarial_loss.item()
            loss_dict['adversarial_loss'] = adversarial_loss
        loss_dict['total_loss'] = total_loss
        return loss_dict

    def get_current_attention(self, vote_attention, point_cloud, total_steps):
        # vote_attention [2500, 24]
        # point_cloud [2500]
        vote_attention = vote_attention.detach().cpu().numpy()
        point_cloud = point_cloud.detach().cpu().numpy()
        num_point, num_joints = vote_attention.shape
        attention_dir = self.image_dir + str(total_steps) + '/'
        cmap = plt.get_cmap('seismic')
        if not osp.exists(self.image_dir):
            os.mkdir(self.image_dir)
        if not osp.exists(attention_dir):
            os.mkdir(attention_dir)
        for i in range(num_joints):
            part_attention = vote_attention[:, i]
            part_attention = part_attention / part_attention.max()
            part_color = cmap(part_attention) * 255
            # print('the content of the part_color:', part_color)
            part_file_name = attention_dir + str(i) + '.ply'
            write_point_cloud(point_cloud, part_file_name, colors=part_color)

    def get_current_votes(self, point_clouds, vote_xyz, gt_segment, pred_segment, total_step):
        # vote_xyz [batch, 2500, 3]
        # gt_segment [batch, 2500]
        vote_xyz = vote_xyz.detach().cpu().numpy()
        gt_segment = gt_segment.detach().cpu().numpy()
        point_clouds = point_clouds.detach().cpu().numpy()
        pred_segment = pred_segment.detach().cpu().numpy()
        if not osp.exists(self.image_dir):
            os.mkdir(self.image_dir)
        pure_pc_dir = self.image_dir + str(total_step) + '_pure_pc.ply'
        vote_visual_dir = self.image_dir + str(total_step) + '_vote_seg.ply'
        point_cloud_visual_dir = self.image_dir + str(total_step) + '_pc_gt_seg.ply'
        segment_visual_dir = self.image_dir + str(total_step) + '_pc_pred_seg.ply'
        write_point_cloud(point_clouds, pure_pc_dir, labels=None)
        write_point_cloud(point_clouds, point_cloud_visual_dir, labels=gt_segment)
        write_point_cloud(vote_xyz, vote_visual_dir, labels=gt_segment)
        write_point_cloud(point_clouds, segment_visual_dir, labels=pred_segment)

    def get_current_mesh_errors(self, gt_vertice, pred_vertice, total_step):
        gt_vertice = gt_vertice.detach().cpu().numpy()
        pred_vertice = pred_vertice.detach().cpu().numpy()
        vertice_diff = np.linalg.norm(gt_vertice - pred_vertice, axis=1)
        # self.faces [num_faces, 3] vertice_diff [6890]
        gt_vertice_dir = self.image_dir + str(total_step) + 'gt_vertice.ply'
        pred_vertice_dir = self.image_dir + str(total_step) + 'pred_vertice.ply'
        error_vertice_dir = self.image_dir + str(total_step) + 'mesh_error.ply'
        face_color = self.vertice_to_face(vertice_diff)
        write_mesh(gt_vertice, self.faces, gt_vertice_dir)
        write_mesh(pred_vertice, self.faces, pred_vertice_dir)
        write_mesh(pred_vertice, self.faces, error_vertice_dir, face_colors=face_color)

    def vertice_to_face(self, vertice_diff):
        face_num = self.faces.shape[0]
        # print('the max value of the vertice_diff:', vertice_diff.max())
        vertice_diff = vertice_diff / 0.50
        vertice_mask = vertice_diff > 1
        vertice_diff[vertice_mask] == 1
        face_diff = np.zeros(face_num, dtype=float)
        cmap = plt.get_cmap('Reds')
        for i in range(face_num):
            single_face = self.faces[i]
            face_diff[i] = vertice_diff[single_face[2]]
            face_color = cmap(face_diff)
        face_color = face_color * 255
        return face_color

    def backward_E(self):
        # offset loss
        if self.has_offset_loss:
            self.of_l1_loss *= self.opt.loss_offset_weight
            # self.loss = self.of_l1_loss
        # keypoints 3d loss
        if self.has_joint3d_loss_before:
            # self.joints3d_loss = return_dict['joints3d_loss']
            self.joints3d_loss_before *= self.opt.loss_3d_weight_before
            # self.loss += self.joints3d_loss_before
        if self.has_smpl_loss:
            self.smpl_loss *= self.opt.loss_smpl_weight
            # self.loss += self.smpl_loss
        if self.has_vertex_loss:
            self.vertex_loss *= self.opt.loss_vertex_weight
            # self.loss += self.vertex_loss
        if self.has_segment_loss:
            self.segment_loss *= self.opt.loss_segment_weight
            # self.loss += self.segment_loss
        if self.has_joint3d_loss_last:
            self.joints3d_loss_last *= self.opt.loss_3d_weight_after
            # self.loss += self.joints3d_loss_last
        if self.has_direction_loss:
            self.offset_dir_loss *= self.opt.loss_dir_weight
            # self.loss += self.offset_dir_loss
        if self.has_orthogonal_loss:
            self.orthogonal_loss *= self.opt.loss_orthogonal_weight
            # self.loss += self.orthogonal_loss
        # print('the shape of the loss:', self.loss.shape)
        if self.has_generate_loss:
            self.generate_loss *= self.opt.loss_generate_weight
            # self.loss += self.generate_loss
        if self.use_no_voting:
            self.loss = self.joints3d_loss_before + self.smpl_loss + self.vertex_loss + self.joints3d_loss_last + self.orthogonal_loss
        elif self.use_adversarial_train or self.use_conditional_gan:
            self.loss = self.of_l1_loss + self.joints3d_loss_before + self.smpl_loss + self.vertex_loss + self.segment_loss \
            + self.joints3d_loss_last + self.orthogonal_loss + self.generate_loss
        else:
            self.loss = self.of_l1_loss + self.joints3d_loss_before + self.smpl_loss + self.vertex_loss + self.segment_loss \
            + self.joints3d_loss_last + self.orthogonal_loss

        self.loss.backward(retain_graph=True)

    def backward_D(self):
        if self.has_adversarial_loss:
            self.adversarial_loss *= self.opt.loss_adversarial_weight
        if self.use_adversarial_train or self.use_conditional_gan:
            self.adversarial_loss.backward()

    def optimize_parameters(self):
        torch.autograd.set_detect_anomaly(True)
        self.optimizer_E.zero_grad()
        self.backward_E()
        self.optimizer_E.step()
        if self.use_adversarial_train or self.use_conditional_gan:
            self.adversarial_model.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not osp.exists(self.save_dir):
            os.mkdir(self.save_dir)
        save_path = osp.join(self.save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

        backup_filename = '%s_net_%s.pth' % ("latest", network_label)
        backup_path = osp.join(self.save_dir, backup_filename)
        shutil.copy2(save_path, backup_path)

    def save_info(self, save_info, epoch_label):
        save_filename = '{}_info.pth'.format(epoch_label)
        save_path = osp.join(self.save_dir, save_filename)
        torch.save(save_info, save_path)

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = osp.join(self.save_dir, save_filename)
        if self.opt.dist:
            network.module.load_state_dict(torch.load(
                save_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))
        else:
            saved_weights = torch.load(save_path)
            network.load_state_dict(saved_weights)

    def load_info(self, epoch_label):
        save_filename = '{}_info.pth'.format(epoch_label)
        save_path = osp.join(self.save_dir, save_filename)
        if self.opt.dist:
            saved_info = torch.load(save_path, map_location=lambda storage, loc: storage.cuda(
                torch.cuda.current_device()))
        else:
            saved_info = torch.load(save_path)
        return saved_info

    def update_learning_rate(self, epoch):
        old_lr = self.opt.lr_e
        lr = 0.5*(1.0 + np.cos(np.pi*epoch/self.opt.total_epoch)) * old_lr
        if lr <= 0.15 * old_lr:
            lr = 0.15 * old_lr
        for param_group in self.optimizer_E.param_groups:
            param_group['lr'] = lr
        if self.use_adversarial_train or self.use_conditional_gan:
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
        if self.opt.process_rank <= 0:
            print("Current Learning Rate:{0:.2E}".format(lr))

    def eval(self):
        self.votenet.eval()

    def return_smpl_vertices(self, shape_param, pose_param, trans):
        # shape_param, pose_param cpu tensors
        # trans is a tensor of size [batch_size, 3]
        batch_size = shape_param.shape[0]
        pose_param_rodrigues = pose_param.view(batch_size, -1).cuda()
        # pose_param_rodrigues is a tensor of size [batch_size, 24*9]
        shape_param = shape_param.view(batch_size, 10).cuda()
        vertices, joints = self.smpl_model(shape_param, pose_param_rodrigues)
        vertices = vertices.view(batch_size, 6890, 3)
        joints = joints.view(batch_size, 24, 3)
        repeat_trans = trans.view(batch_size, 1, 3)
        vertices_trans = repeat_trans.repeat(1, 6890, 1)
        joints_trans = repeat_trans.repeat(1, 24, 1)
        vertices = vertices + vertices_trans
        joints = joints + joints_trans
        return vertices, joints

