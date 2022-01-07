from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import shutil
import os.path as osp
import numpy as np
import pickle
import copy
import cv2
import time
import torch
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
from src.utils.visual_utils import Visualizer
from mpl_toolkits.mplot3d import Axes3D
from src.options.test_options import TestOptions
from src.utils.renderer import *
import src.utils.parallel_io as pio
import os.path as osp
import os


def smpl_view_set_axis_full_body(ax, azimuth=0):
    ## Manually set axis
    ax.view_init(0, azimuth)
    max_range = 0.55
    ax.set_xlim(- max_range, max_range)
    ax.set_ylim(- max_range, max_range)
    ax.set_zlim(-0.2 - max_range, -0.2 + max_range)
    ax.axis('off')


class Evaluator(object):
    def __init__(self, opt, model_root=''):
        if len(model_root) > 0:
            self.model_root = model_root
        with open('./data/male_model.pkl', 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        print('the keys of the model:', model.keys())
        self.faces = model['f']
        self.count1 = 0
        self.pred_results = list()
        self.baseline_results = list()
        self.comparison_results = list()
        self.visualizer = Visualizer(opt)
        self.evaluate_dir = opt.evaluate_dir

    def clear(self):
        self.pred_results = list()
        self.baseline_results = list()

    def pred_update(self, pred_joints, gt_vertices, pred_vertices, pred_joints_last, gt_smpl_joints=None):
        # pred_smpl_joints is a tensor of size [batch_size, 24, 3]
        # smpl_joints is a tensor of size [batch_size, 24, 3]
        # segmentation is a tensor of size [batch_size, 6890, 24]
        batch_size = pred_joints.shape[0]
        for i in range(batch_size):
            single_data = dict(
                gt_vertice=gt_vertices[i],
                pred_vertice=pred_vertices[i],
                pred_joints_last=pred_joints_last[i],
                gt_smpl_joints=gt_smpl_joints[i]
            )
            # pve
            verts1 = single_data['gt_vertice']
            verts2 = single_data['pred_vertice']
            single_data['pve'] = np.average(
                np.linalg.norm(verts1 - verts2, axis=1))
            # mpjpe
            smpl_joints1 = single_data['gt_smpl_joints']
            smpl_joints2 = single_data['pred_joints_last']
            single_data['mpjpe'] = np.average(
                np.linalg.norm(smpl_joints1 - smpl_joints2, axis=1))
            self.pred_results.append(single_data)

    def baseline_update(self, pred_joints, gt_vertices, pred_vertices, pred_joints_last, gt_smpl_joints=None):
        # pred_smpl_joints is a tensor of size [batch_size, 24, 3]
        # smpl_joints is a tensor of size [batch_size, 24, 3]
        # segmentation is a tensor of size [batch_size, 6890, 24]
        batch_size = pred_joints.shape[0]
        for i in range(batch_size):
            single_data = dict(
                gt_vertice=gt_vertices[i],
                pred_vertice=pred_vertices[i],
                pred_joints_last=pred_joints_last[i],
                gt_smpl_joints=gt_smpl_joints[i]
            )
            # pve
            verts1 = single_data['gt_vertice']
            verts2 = single_data['pred_vertice']
            single_data['pve'] = np.average(
                np.linalg.norm(verts1 - verts2, axis=1))
            # mpjpe
            smpl_joints1 = single_data['gt_smpl_joints']
            smpl_joints2 = single_data['pred_joints_last']
            single_data['mpjpe'] = np.average(
                np.linalg.norm(smpl_joints1 - smpl_joints2, axis=1))
            self.baseline_results.append(single_data)

    def remove_redunc(self):
        print("Number of test data:", len(self.pred_results))

    def comparison(self):
        test_data_length = len(self.pred_results)
        for index in range(test_data_length):
            error_diff = abs(self.pred_results[index]['pve'] - self.baseline_results[index]['pve'])
            if error_diff >= 0.020:
                pred_vertices = self.pred_results[index][]

    @property
    def pve(self):
        res = np.average([data['pve']
                          for data in self.pred_results])
        return res

    def mpjpe(self):
        res = np.average([data['mpjpe']
                          for data in self.pred_results])
        return res

    def mpjpe_max(self):
        res = np.max(data['mpjpe'] for data in self.pred_results)
        return res

    def pve_max(self):
        res = np.max([data['pve'] for data in self.pred_results])
        # self.pve_max_image()
        return res

    def get_current_mesh_errors(self, gt_vertice, pred_vertice, total_step):
        vertice_diff = np.linalg.norm(gt_vertice - pred_vertice, axis=1)
        # self.faces [num_faces, 3] vertice_diff [6890]
        index_dir = self.evaluate_dir + str(total_step)
        if not osp.exists(index_dir):
            os.mkdir(index_dir)
        gt_vertice_dir = index_dir + '/gt_vertice.stl'
        pred_vertice_dir = index_dir + '/pred_vertice.stl'
        error_vertice_dir = index_dr + '/mesh_error.stl'
        face_color = self.vertice_to_face(vertice_diff)
        write_mesh(gt_vertice, self.faces, gt_vertice_dir)
        write_mesh(pred_vertice, self.faces, pred_vertice_dir)
        write_mesh(gt_vertice, self.faces, error_vertice_dir, face_colors=face_color)

    def vertice_to_face(self, vertice_diff):
        face_num = self.faces.shape[0]
        face_matrix = np.zeros(face_num)
        for i in range(face_num):
            single_face = self.faces[i]
            face_matrix[i] = vertice_diff[single_face[2]]
        face_matrix = face_matrix / face_matrix.max()
        face_matrix = face_matrix.reshape(face_num, 1)
        face_matrix = face_matrix.repeat(3, axis=1)
        face_matrix = 1 - face_matrix
        return face_matrix

    def pve_max_count(self):
        if not osp.exists(self.evaluate_dir):
            os.mkdir(self.evaluate_dir)
        count1 = 0
        count2 = 0
        count3 = 0
        for data in self.pred_results:
            if data['pve'] >= 0.100:
                count1 = count1 + 1
                count1_dir = self.evaluate_dir + '100/'
                if not osp.exists(count1_dir):
                    os.mkdir(count1_dir)
                image_dir_1 = count1_dir + str(count1) + 'pc_visualize.png'
                image_dir_2 = count1_dir + str(count1) + 'errors_compare.png'
                gt_vertice = data['gt_vertice']
                gt_vertice = torch.from_numpy(gt_vertice)
                pred_vertice = data['pred_vertice']
                pred_vertice = torch.from_numpy(pred_vertice)
                point_cloud = data['point_cloud']
                point_cloud = torch.from_numpy(point_cloud)
                gt_joints = data['gt_joints']
                gt_joints = torch.from_numpy(gt_joints)
                pred_joints = data['pred_joints']
                pred_joints = torch.from_numpy(pred_joints)
                self.visualizer.point_cloud_visualize(point_cloud, gt_joints, pred_joints, image_dir_1)
                self.visualizer.error_visualize(pred_vertice, gt_vertice, image_dir_2)
        print('number of pve bigger than 100 mm:', count1)
        print('number of pve bigger than 50 mm:', count2)
        print('number of pve bigger than 20 mm:', count3)

    def save_to_pkl(self, res_pkl_file):
        saved_data = dict(
            model_root=self.model_root,
            pred_results=self.pred_results
        )
        pio.save_pkl_single(res_pkl_file, saved_data, protocol=2)

    def load_from_pkl(self, pkl_path):
        saved_data = pio.load_pkl_single(pkl_path)
        self.pred_results = saved_data['pred_results']
        self.model_root = saved_data['model_root']


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    evaluate_file_path = opt.save_evaluate_dir + opt.evaluate_file_name
    evaluator = Evaluator(opt)
    evaluator.load_from_pkl(evaluate_file_path)
    test_data_length = len(evaluator.pred_results)
    mesh_dir = opt.save_evaluate_dir + '/mesh/'
    if not osp.exists(mesh_dir):
        os.mkdir(mesh_dir)
    for index in range(test_data_length):
        single_data = evaluator.pred_results[index]
        if single_data['pve'] > 0.08:
            evaluator.get_current_mesh_errors(single_data['gt_vertice'], single_data['pred_vertice'], i)
            evaluator.get_current_vote(single_data['point_cloud'], single_data['vote_xyz'], single_data['pred_segment'],
                                  single_data['gt_segment'], i)
