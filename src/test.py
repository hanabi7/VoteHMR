from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, shutil
import os.path as osp
import time
from datetime import datetime
import torch
import numpy
import random
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from src.options.test_options import TestOptions
from src.datasets.data_loader import CreateDataLoader
from src.models.human_point_cloud import HumanPointCloud
from src.utils.evaluator import Evaluator
import cv2
import numpy as np
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Timer(object):
    def __init__(self, num_batch):
        self.start = time.time()
        self.num_batch = num_batch

    def click(self, batch_id):
        start, num_batch = self.start, self.num_batch
        end = time.time()
        cost_time = (end - start) / 60
        speed = (batch_id + 1) / cost_time
        res_time = (num_batch - (batch_id + 1)) / speed
        print("we have process {0}/{1}, it takes {2:.3f} mins, remain needs {3:.3f} mins".format(
            (batch_id + 1), num_batch, cost_time, res_time))
        sys.stdout.flush()


def main():
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    visualize_eval = opt.visualize_eval
    opt.process_rank = -1
    data_loader = CreateDataLoader(opt)
    test_dataset = data_loader.load_data()
    evaluator = Evaluator(opt, opt.smpl_model_filename)
    test_res_dir = 'evaluate_results'
    evaluator.clear()
    model = HumanPointCloud(opt)
    model.eval()
    epoch = opt.which_epoch
    timer = Timer(len(test_dataset))
    # print('max value of the down sample matrix:', torch.sum(down_sample_matrix))
    for i, data in enumerate(test_dataset):
        if data['gt_joints3d'].shape[0] == opt.batchSize:
            data = model.set_input(data)
            # print('gt_vertices:', gt_vertices.shape)
            model.test(data)
            gt_vertices = model.gt_vertices.detach().cpu().numpy()
            pred_vertices = model.pred_vertices.detach().cpu().numpy()
            point_clouds = data['point_cloud'].detach().cpu().numpy()
            gt_joints = data['gt_joints3d'].detach().cpu().numpy()
            pred_joints = model.refined_joints.detach().cpu().numpy()
            pred_joints_last = model.pred_joints_last.detach().cpu().numpy()
            if not opt.use_no_voting:
                vote_xyz = model.vote_xyz.detach().cpu().numpy()
                gt_segment = model.gt_segment.detach().cpu().numpy()
                pred_segmentation = model.pred_segmentation.detach().cpu().numpy()
            pred_tpose_verts = model.pred_verts_tpose.detach().cpu().numpy()
            gt_tpose_verts = model.gt_verts_tpose.detach().cpu().numpy()
            gt_verts_tshape = model.gt_verts_tshape.detach().cpu().numpy()
            pred_verts_tshape = model.pred_verts_tshape.detach().cpu().numpy()
            gt_smpl_joints = model.gt_smpl_joints.detach().cpu().numpy()
            if not opt.use_no_voting:
                evaluator.update(point_clouds, pred_joints, gt_joints, gt_vertices, pred_vertices, pred_joints_last, vote_xyz, gt_segment,
                                 pred_segmentation, pred_tpose_verts, gt_tpose_verts, pred_verts_tshape, gt_verts_tshape, gt_smpl_joints)
            else:
                evaluator.update(point_clouds, pred_joints, gt_joints, gt_vertices, pred_vertices, pred_joints_last,
                                 pred_tpose_verts=pred_tpose_verts, gt_tpose_verts=gt_tpose_verts, pred_verts_tshape=pred_verts_tshape, gt_verts_tshape=gt_verts_tshape, gt_smpl_joints=gt_smpl_joints)
            timer.click(i)
            model.basic_visualize(0, i)
    evaluator.remove_redunc()
    if opt.save_evaluate_result:
        res_pkl_file = osp.join(opt.save_evaluate_dir, 'estimator_{}.pkl'.format(epoch))
        evaluator.save_to_pkl(res_pkl_file)
    # res_pkl_file = osp.join(test_res_dir, 'estimator_{}.pkl'.format(epoch))
    # backup_pkl_file = osp.join(test_res_dir, 'estimator.pkl')
    print("Test of epoch: {} complete".format(epoch))
    print("PVE:{}".format(evaluator.pve))
    print("MPJPE:{}".format(evaluator.mpjpe))
    print("PVE_MAX：{}".format(evaluator.pve_max()))
    print("MPJPE_AFTER:{}".format(evaluator.mpjpe_after()))
    print("MPJPE_MAX:{}".format(evaluator.mpjpe_max()))
    evaluator.vote_max_image()
    evaluator.pve_max_image()
    evaluator.pve_max_count()
    evaluator.MPJPE_max_count()
    print("TPVE:{}".format(evaluator.pve_tpose))
    print("Tshape:{}".format(evaluator.pve_tshape()))
    # print("ACCURACY:{}".format(evaluator.accuracy()))
    print('------------------')
    sys.stdout.flush()


if __name__ == '__main__':
    main()
