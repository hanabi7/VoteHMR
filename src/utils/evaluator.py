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
        self.visualizer = Visualizer(opt)
        self.evaluate_dir = opt.evaluate_dir

    def clear(self):
        self.pred_results = list()

    def update(self, point_clouds, pred_joints, gt_joints, gt_vertices, pred_vertices, pred_joints_last, vote_xyz=None, gt_segment=None, pred_segmentation=None,
               pred_tpose_verts=None, gt_tpose_verts=None, pred_verts_tshape=None, gt_verts_tshape=None, gt_smpl_joints=None):
        # pred_smpl_joints is a tensor of size [batch_size, 24, 3]
        # smpl_joints is a tensor of size [batch_size, 24, 3]
        # segmentation is a tensor of size [batch_size, 6890, 24]
        batch_size = pred_joints.shape[0]
        for i in range(batch_size):
            if vote_xyz is not None:
                single_data = dict(
                    point_cloud=point_clouds[i],
                    gt_joints=gt_joints[i],
                    pred_joints=pred_joints[i],
                    gt_vertice=gt_vertices[i],
                    pred_vertice=pred_vertices[i],
                    pred_joints_last=pred_joints_last[i],
                    vote_xyz=vote_xyz[i],
                    gt_segment=gt_segment[i],
                    pred_segment=pred_segmentation[i],
                    pred_tpose_verts=pred_tpose_verts[i],
                    gt_tpose_verts=gt_tpose_verts[i],
                    pred_verts_tshape=pred_verts_tshape[i],
                    gt_verts_tshape=gt_verts_tshape[i],
                    gt_smpl_joints=gt_smpl_joints[i]
                )
            else:
                single_data = dict(
                    point_cloud=point_clouds[i],
                    gt_joints=gt_joints[i],
                    pred_joints=pred_joints[i],
                    gt_vertice=gt_vertices[i],
                    pred_vertice=pred_vertices[i],
                    pred_joints_last=pred_joints_last[i],
                    pred_tpose_verts=pred_tpose_verts[i],
                    gt_tpose_verts=gt_tpose_verts[i],
                    pred_verts_tshape=pred_verts_tshape[i],
                    gt_verts_tshape=gt_verts_tshape[i],
                    gt_smpl_joints=gt_smpl_joints[i]
                )
            # pve
            verts1 = single_data['gt_vertice']
            verts2 = single_data['pred_vertice']
            single_data['pve'] = np.average(
                np.linalg.norm(verts1 - verts2, axis=1))
            # mpjpe
            smpl_joints1 = single_data['gt_joints']
            smpl_joints2 = single_data['pred_joints']
            single_data['mpjpe'] = np.average(
                np.linalg.norm(smpl_joints1 - smpl_joints2, axis=1))
            # mpjpe_after
            smpl_joints1 = single_data['gt_smpl_joints']
            smpl_joints2 = single_data['pred_joints_last']
            single_data['mpjpe_after'] = np.average(
                np.linalg.norm(smpl_joints1 - smpl_joints2, axis=1))
            # accuracy [6890]
            tpose_verts1 = single_data['pred_tpose_verts']
            tpose_verts2 = single_data['gt_tpose_verts']
            single_data['tpose_pve'] = np.average(
                np.linalg.norm(tpose_verts1 - tpose_verts2, axis=1)
            )
            tshape_verts1 = single_data['pred_verts_tshape']
            tshape_verts2 = single_data['gt_verts_tshape']
            single_data['tshape_pve'] = np.average(
                np.linalg.norm(tshape_verts1 - tshape_verts2, axis=1)
            )
            self.pred_results.append(single_data)

    def remove_redunc(self):
        print("Number of test data:", len(self.pred_results))

    @property
    def pve(self):
        res = np.average([data['pve']
                          for data in self.pred_results])
        return res

    @property
    def mpjpe(self):
        res = np.average([data['mpjpe']
                          for data in self.pred_results])
        return res

    @property
    def pve_tpose(self):
        res = np.average([data['tpose_pve']
                          for data in self.pred_results])
        return res

    def pve_tshape(self):
        res = np.average([data['tshape_pve']
                          for data in self.pred_results
                          ])
        return res

    def mpjpe_after(self):
        res = np.average([data['mpjpe_after']
                          for data in self.pred_results])
        return res

    def mpjpe_max(self):
        res = np.max(data['mpjpe'] for data in self.pred_results)
        return res

    def pve_max(self):
        res = np.max([data['pve'] for data in self.pred_results])
        # self.pve_max_image()
        return res

    def vote_max_image(self):
        idx = np.argmax([data['mpjpe'] for data in self.pred_results])
        point_cloud = self.pred_results[idx]['point_cloud']
        point_cloud = torch.from_numpy(point_cloud)
        gt_joints = self.pred_results[idx]['gt_joints']
        gt_joints = torch.from_numpy(gt_joints)
        pred_joints = self.pred_results[idx]['pred_joints']
        pred_joints = torch.from_numpy(pred_joints)
        """
        gt_vertice = self.pred_results[idx]['gt_vertice']
        gt_vertice = torch.from_numpy(gt_vertice)
        pred_vertice = self.pred_results[idx]['pred_vertice']
        pred_vertice = torch.from_numpy(pred_vertice)
        """
        if not osp.exists('/data1/liuguanze/depth_point_cloud/worst/'):
            os.mkdir('/data1/liuguanze/depth_point_cloud/worst/')
        image_dir_1 = '/data1/liuguanze/depth_point_cloud/worst/' + str(idx) + '_vote_max_error_joints' + '.png'
        # image_dir_2 = '/data1/liuguanze/depth_point_cloud/worst/' + str(idx) + '_vote_max_error_vertices' + '.png'
        self.visualizer.point_cloud_visualize(point_cloud, gt_joints, pred_joints, image_dir_1)

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

    def get_current_vote(self, point_cloud, vote_xyz, pred_segment, gt_segment, total_steps):
        """
        Input: point_cloud: [2500, 3]
        Input: vote_xyz:  [2500, 3]
        Input: pred_segment: [2500]
        Input: gt_segment: [2500]
        """
        index_dir = self.evaluate_dir + str(total_steps)
        if not osp.exists(index_dir):
            os.mkdir(index_dir)
        vote_result_dir = index_dir + '/vote_result.ply'
        gt_segment_dir = index_dir + '/gt_segment.ply'
        pred_segment_dir = index_dir + '/pred_segment.ply'
        write_point_cloud(vote_xyz, vote_result_dir, labels=gt_segment)
        write_point_cloud(point_cloud, gt_segment_dir, labels=gt_segment)
        write_point_cloud(point_cloud, pred_segment_dir, labels=pred_segment)

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

    def pve_max_image(self):
        idx = np.argmax([data['pve'] for data in self.pred_results])
        point_cloud = self.pred_results[idx]['point_cloud']
        point_cloud = torch.from_numpy(point_cloud)
        gt_joints = self.pred_results[idx]['gt_joints']
        gt_joints = torch.from_numpy(gt_joints)
        pred_joints = self.pred_results[idx]['pred_joints']
        pred_joints = torch.from_numpy(pred_joints)
        gt_vertice = self.pred_results[idx]['gt_vertice']
        gt_vertice = torch.from_numpy(gt_vertice)
        pred_vertice = self.pred_results[idx]['pred_vertice']
        pred_vertice = torch.from_numpy(pred_vertice)
        if not osp.exists('/data1/liuguanze/depth_point_cloud/worst/'):
            os.mkdir('/data1/liuguanze/depth_point_cloud/worst/')
        image_dir_1 = '/data1/liuguanze/depth_point_cloud/worst/' + str(idx) + '_pve_max_error_joints' + '.png'
        image_dir_2 = '/data1/liuguanze/depth_point_cloud/worst/' + str(idx) + '_pve_max_error_vertices' + '.png'
        self.visualizer.point_cloud_visualize(point_cloud, gt_joints, pred_joints, image_dir_1)
        self.visualizer.error_visualize(pred_vertice, gt_vertice, image_dir_2)
        # self.segmentation_visualize(gt_vertice, segmentation, image_dir, 0)

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
            """
            if data['pve'] >= 0.050 and data['pve'] < 0.100:
                count2 = count2 + 1
                count2_dir = self.evaluate_dir + '50/'
                if not osp.exists(count2_dir):
                    os.mkdir(count2_dir)
                image_dir_1 = count2_dir + str(count2) + 'pc_visualize.png'
                image_dir_2 = count2_dir + str(count2) + 'errors_compare.png'
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
            if data['pve'] >= 0.020 and data['pve'] < 0.050:
                count3 = count3 + 1"""
        print('number of pve bigger than 100 mm:', count1)
        print('number of pve bigger than 50 mm:', count2)
        print('number of pve bigger than 20 mm:', count3)

    def MPJPE_max_count(self):
        count = 0
        for data in self.pred_results:
            if data['mpjpe'] > 0.145:
                count = count + 1
        print('number of mpjpe bigger than 145mm:', count)

    def visualize_result(self, res_dir):
        num_process = 8
        num_each = len(self.pred_results) // num_process
        process_list = list()
        for i in range(num_process):
            start = i*num_each
            end = (i+1)*num_each if i<num_process-1 else len(self.pred_results)
            p = mp.Process(target=self.visualize_result_single, args=(start, end, res_dir, render_util))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

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


def main():

    evaluator = Evaluator()
    if len(sys.argv) > 1:
        epoch = sys.argv[1]
        pkl_path = 'evaluate_results/estimator_{}.pkl'.format(epoch)
        res_dir = 'evaluate_results/images/{}'.format(epoch)
    else:
        pkl_path = 'evaluate_results/estimator.pkl'
        res_dir = 'evaluate_results/images/default'
    ry_utils.renew_dir(res_dir)
    start = time.time()
    evaluator.load_from_pkl(pkl_path)
    end = time.time()
    print("Load evaluate results complete, time costed : {0:.3f}s".format(end-start))
    evaluator.visualize_result(res_dir)


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


