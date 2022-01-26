from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, shutil
import os.path as osp
import time
from datetime import datetime
from torch.utils.data import DataLoader
import torch
import numpy
import random
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from src.options.test_options import TestOptions
from src.datasets.mhad import MHAD
from src.models.human_point_cloud import HumanPointCloud
from src.utils.mhad_evaluate import MhadEvaluator
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
    opt.process_rank = -1
    test_datasets = MHAD(opt)
    dataloader = DataLoader(
        test_datasets,
        batch_size=opt.batchSize,
        shuffle=False,
        drop_last=opt.isTrain
    )
    evaluator = MhadEvaluator()
    evaluator.clear()
    model = HumanPointCloud(opt)
    model.eval()
    timer = Timer(len(dataloader))
    for i, data in enumerate(dataloader):
        if data.shape[0] == opt.batchSize:
            model.mhad_test(data)
            pred_vertices = model.pred_vertices.detach().cpu().numpy()
            point_clouds = model.point_clouds.detach().cpu().numpy()
            vote_xyz = model.vote_xyz.detach().cpu().numpy()
            pred_segmentation = torch.topk(model.pred_segmentation, 1)[1].squeeze(-1)
            pred_segmentation = pred_segmentation.detach().cpu().numpy()
            evaluator.update(point_clouds, pred_vertices, vote_xyz, pred_segmentation)
            timer.click(i)
    print('------------------')
    print('p2v distance of the mhad datasets:', evaluator.p2v())
    print('p2v max distance of the mhad datasets:', evaluator.p2v_max())
    sys.stdout.flush()


if __name__ == '__main__':
    main()
