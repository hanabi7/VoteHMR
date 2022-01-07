import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import os.path as osp
import pickle
import cv2

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.smpl_key_points_number = opt.smpl_key_points_number
        self.smpl_model_filename = opt.smpl_model_filename
        with open(self.smpl_model_filename, 'rb') as f:
            params = pickle.load(f, encoding='latin1')
        self.kintree_table = params['kintree_table']

    def smpl_view_set_axis_full_body(self, ax, azimuth=0):
        ## Manually set axis
        ax.view_init(0, azimuth)
        max_range = 0.55
        ax.set_xlim(- max_range, max_range)
        ax.set_ylim(- max_range, max_range)
        ax.set_zlim(-0.3 - max_range, -0.3 + max_range)
        ax.axis('off')

    def key_points_check(self, gt_vertices, gt_keypoints, dir):
        """
        Inputs:
            gt_vertices: [6890, 3] tensor
            gt_keypoints: [24, 3] tensor
            dir:
        """
        fig = plt.figure(figsize=[20, 10])
        gt_vertices = gt_vertices.detach().cpu().numpy()
        gt_keypoints = gt_keypoints.detach().cpu().numpy()
        visual_keypoints = gt_keypoints
        x, y, z = gt_vertices[:, 0], gt_vertices[:, 1], gt_vertices[:, 2]
        key_x, key_y, key_z = visual_keypoints[:, 0], visual_keypoints[:, 1], visual_keypoints[:, 2]
        y = -y
        key_y = - key_y
        visual_keypoints[:, 1] = - visual_keypoints[:, 1]
        ax = fig.add_subplot(131, projection='3d')
        ax.view_init(0, 0)
        ax.scatter(x, z, y, s=0.1, c='k')
        ax.scatter(key_x, key_z, key_y)
        self.draw_joints3d(visual_keypoints, ax, self.kintree_table)
        ax.axis('off')
        plt.title('the ground truth key points after.')
        plt.savefig(dir)
        plt.close()

    def segmentation_visualize(self, gt_point_clouds, labels, pred_segment, dir):
        """
        Inputs:
            gt_point_clouds: [2500, 3] tensor
            joints: [24, 3] tensor
            labels: [2500] labels
            pr_segment: [2500] labels
            dir:
        """
        fig = plt.figure(figsize=[20, 10])
        labels = labels.detach().cpu().numpy()
        pred_segment = pred_segment.detach().cpu().numpy()
        # joints = joints.detach().cpu().numpy()
        labels = labels / labels.max()
        pred_segment = pred_segment / pred_segment.max()
        point_clouds = gt_point_clouds.detach().cpu().numpy()
        x, y, z = point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2]
        y = -y
        ax = fig.add_subplot(131, projection='3d')
        ax.view_init(0, 0)
        ax.scatter(x, z, y, s=0.1, c=labels)
        ax.axis('off')
        plt.title('the ground_truth segmentation')
        ax = fig.add_subplot(132, projection='3d')
        ax.view_init(0, 0)
        ax.scatter(x, z, y, s=0.1, c=pred_segment)
        ax.axis('off')
        plt.title('the predicted segmentation')
        plt.savefig(dir)
        plt.close()

    def point_cloud_visualize(self, point_clouds, gt_keypoints, pred_keypoints, dir):
        """
        Input:
            point_clouds: [2500, 3]
            key_points: [24, 3]
            dir: image_dir must mkdir before assign this function
        """
        fig = plt.figure(figsize=[20, 10])
        point_clouds = point_clouds.detach().cpu().numpy()
        gt_keypoints = gt_keypoints.detach().cpu().numpy()
        pred_keypoints = pred_keypoints.detach().cpu().numpy()
        visual_gt_keypoints = gt_keypoints
        visual_pred_keypoints = pred_keypoints
        x, y, z = point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2]
        joints_x, joints_y, joints_z = gt_keypoints[:, 0], gt_keypoints[:, 1], gt_keypoints[:, 2]
        pred_x, pred_y, pred_z = pred_keypoints[:, 0], pred_keypoints[:, 1], pred_keypoints[:, 2]
        joints_y = - joints_y
        pred_y = - pred_y
        visual_gt_keypoints[:, 1] = - visual_gt_keypoints[:, 1]
        visual_pred_keypoints[:, 1] = - visual_pred_keypoints[:, 1]
        y = -y
        ax = fig.add_subplot(131, projection='3d')
        ax.view_init(0, 0)
        ax.scatter(x, z, y, s=0.1, c='k')
        ax.scatter(joints_x, joints_z, joints_y)
        self.draw_joints3d(visual_gt_keypoints, ax, self.kintree_table)
        ax.axis('off')
        plt.title('The ground_truth key points')
        ax = fig.add_subplot(132, projection='3d')
        ax.view_init(0, 0)
        ax.scatter(x, z, y, s=0.1, c='k')
        ax.scatter(pred_x, pred_z, pred_y)
        self.draw_joints3d(visual_pred_keypoints, ax, self.kintree_table)
        ax.axis('off')
        plt.title('The predicted key points')
        plt.savefig(dir)
        plt.close()

    def vote_visualize(self, point_cloud, vote_xyz, dir):
        point_cloud = point_cloud.detach().cpu().numpy()
        vote_xyz = vote_xyz.detach().cpu().numpy()
        x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
        vote_x, vote_y, vote_z = vote_xyz[:, 0], vote_xyz[:, 1], vote_xyz[:, 2]
        y = -y
        vote_y = - vote_y
        fig = plt.figure(figsize=[20, 7])
        ax = fig.add_subplot(131, projection='3d')
        ax.view_init(0, 0)
        ax.scatter(x, z, y, s=0.1, c='k')
        ax.axis('off')
        ax = fig.add_subplot(132, projection='3d')
        ax.view_init(0, 0)
        ax.scatter(vote_x, vote_z, vote_y, s=0.1, c='k')
        ax.axis('off')
        plt.savefig(dir)
        plt.close()

    def draw_joints3d(self, joints3d, ax=None, kintree_table=None, color='g'):
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(131)
        for i in range(1, kintree_table.shape[1]):
            j1 = kintree_table[0][i]
            j2 = kintree_table[1][i]
            ax.plot([joints3d[j1, 0], joints3d[j2, 0]],
                    [joints3d[j1, 2], joints3d[j2, 2]],
                    [joints3d[j1, 1], joints3d[j2, 1]],
                    color=color, linestyle='-', linewidth=2)

    def error_visualize(self, pred_vertice, gt_vertice, dir):
        # must mkdir before using this function
        pred_vertice = pred_vertice.detach().cpu().numpy()
        gt_vertice = gt_vertice.detach().cpu().numpy()
        pred_x, pred_y, pred_z = pred_vertice[:, 0], pred_vertice[:, 1], pred_vertice[:, 2]
        x, y, z = gt_vertice[:, 0], gt_vertice[:, 1], gt_vertice[:, 2]
        pred_y = - pred_y
        y = - y
        offset_error = abs(gt_vertice - pred_vertice)
        offset_error = offset_error / offset_error.max()
        # offsets error between [0, 1]
        fig = plt.figure(figsize=[20, 7])
        ax = fig.add_subplot(131, projection='3d')
        ax.view_init(0, 0)
        ax.scatter(pred_x, pred_z, pred_y, s=0.1, c='k')
        ax.axis('off')
        plt.title('Pred Vertices:')
        ax = fig.add_subplot(132, projection='3d')
        ax.scatter(x, z, y, s=0.1, c='k')
        ax.view_init(0, 0)
        ax.axis('off')
        plt.title('Ground_truth Vertices:')
        ax = fig.add_subplot(133, projection='3d')
        ax.scatter(x, z, y, s=0.1, c=offset_error, cmap='coolwarm')
        ax.view_init(0, 0)
        ax.axis('off')
        plt.title('Errors on Vertices:')
        plt.savefig(dir)
        plt.close()

    def edge_conv_visualization(self, attention_feature, refine_key_points, dir):
        # attention_feature [131]
        # refine_key_points [24, 3]
        attention_feature = attention_feature.detach().cpu().numpy()
        refine_key_points = refine_key_points.detach().cpu().numpy()
        num_points = refine_key_points.shape[0]
        attention_coordinates = attention_feature[:3]
        joint_x, joint_y, joint_z = refine_key_points[:, 0], refine_key_points[:, 1], refine_key_points[:, 2]
        center_x, center_y, center_z = attention_coordinates[0], attention_coordinates[1], attention_coordinates[2]
        fig = plt.figure(figsize=[20, 7])
        ax = fig.add_subplot(131, projection='3d')
        ax.scatter(joint_x, joint_z, joint_y, s=1, c='k')
        ax.scatter(center_x, center_z, center_y, s=2, c='b')
        for i in range(num_points):
            ax.plot([refine_key_points[i, 0], center_x],
                    [refine_key_points[i, 2], center_z],
                    [refine_key_points[i, 1], center_y],
                    color='g', linestyle='-', linewidth=2)
        ax.view_init(0, 0)
        ax.axis('off')
        plt.savefig(dir)

    # save image to the disk
    def point_clouds_compare(self, point_clouds, pred_vertice, dir):
        """
        Inputs:
            point_clouds: [2500, 3] tensor
            pred_vertice: [6890, 3]
            dir: must mkdir before assign this function
        """
        point_clouds = point_clouds.detach().cpu().numpy()
        pred_vertice = pred_vertice.detach().cpu().numpy()
        x, y, z = point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2]
        pred_x, pred_y, pred_z = pred_vertice[:, 0], pred_vertice[:, 1], pred_vertice[:, 2]
        y = - y
        pred_y = - pred_y
        fig = plt.figure(figsize=[20, 10])
        ax = fig.add_subplot(131, projection='3d')
        ax.scatter(x, z, y, s=0.1, c='k')
        ax.view_init(0, 0)
        ax.axis('off')
        plt.title('Gt points:')
        ax = fig.add_subplot(132, projection='3d')
        ax.scatter(pred_x, pred_z, pred_y, s=0.1, c='k')
        ax.view_init(0, 0)
        ax.axis('off')
        plt.title('Pred points:')
        plt.savefig(dir)
        plt.close()

