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
from mpl_toolkits.mplot3d import Axes3D
from src.options.test_options import TestOptions
from src.utils.renderer import *
import src.utils.parallel_io as pio
import os.path as osp
import os
from chamfer_distance import ChamferDistance


class MhadEvaluator(object):
    def __init__(self):
        self.face_list = pickle.load(open('./data/face_list.pkl', 'rb'))
        self.faces = self.face_list[1]
        self.mesh_count = 0
        self.pred_results = list()
        self.chamfer_dist = ChamferDistance()
        self.evaluate_dir = './evaluate/mhad/'
        if not osp.exists(self.evaluate_dir):
            os.mkdir(self.evaluate_dir)

    def clear(self):
        self.pred_results = list()

    def update(self, point_clouds, pred_vertices, vote_xyz, pred_segmentation):
        batch_size = point_clouds.shape[0]
        point_clouds = torch.from_numpy(point_clouds)
        pred_vertices = torch.from_numpy(pred_vertices)
        dist = self.chamfer_dist(point_clouds, pred_vertices)
        dist2 = dist[1]
        print('the shape of the dist2:', dist2.shape)
        point_clouds = point_clouds.detach().numpy()
        pred_vertices = pred_vertices.detach().numpy()
        dist2 = dist2.detach().numpy()
        for i in range(batch_size):
            single_data = dict(
                point_cloud=point_clouds[i],
                pred_vertice=pred_vertices[i],
                vote_xyz=vote_xyz[i],
                pred_segment=pred_segmentation[i],
                mesh_error=dist2[i]
            )
            # pve
            single_data['p2v_distance'] = np.average(dist2)
            self.pred_results.append(single_data)
            self.get_current_mesh_errors(pred_vertices[i], self.mesh_count)
            self.get_current_vote(point_clouds[i], vote_xyz[i], pred_segmentation[i], self.mesh_count)
            self.mesh_count += 1

    def p2v(self):
        res = np.average([data['p2v_distance']
                          for data in self.pred_results])
        return res

    def p2v_max(self):
        res = np.max(data['p2v_distance'] for data in self.pred_results)
        return res

    def get_current_mesh_errors(self, pred_vertice, total_step):
        index_dir = self.evaluate_dir + str(total_step)
        if not osp.exists(index_dir):
            os.mkdir(index_dir)
        pred_vertice_dir = index_dir + '/pred_vertice.ply'
        write_mesh(pred_vertice, self.faces, pred_vertice_dir)

    def get_current_vote(self, point_cloud, vote_xyz, pred_segment, total_steps):
        """
        Input: point_cloud: [2500, 3]
        Input: vote_xyz:  [2500, 3]
        Input: pred_segment: [2500]
        Input: gt_segment: [2500]
        """
        index_dir = self.evaluate_dir + str(total_steps)
        if not osp.exists(index_dir):
            os.mkdir(index_dir)
        pc_result_dir = index_dir + '/pc_result.ply'
        vote_result_dir = index_dir + '/vote_result.ply'
        pred_segment_dir = index_dir + '/pred_segment.ply'
        write_point_cloud(point_cloud, pc_result_dir)
        write_point_cloud(vote_xyz, vote_result_dir, labels=pred_segment)
        write_point_cloud(point_cloud, pred_segment_dir, labels=pred_segment)
