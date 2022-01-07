from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import shutil
import time
from datetime import datetime
import torch
import numpy
import random
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
from src.utils.train_utils import *
from src.options.train_options import TrainOptions
from src.datasets.data_loader import CreateDataLoader
from torch.multiprocessing import Process
from src.utils.evaluator import Evaluator
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
import numpy as np
import pdb
from src.utils.visdom_utils import VisdomObserver
from src.models.human_point_cloud import HumanPointCloud
from multiprocessing import set_start_method


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

def main():
    opt = TrainOptions().parse()
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    # distributed learning initiate
    if opt.dist:
        init_dist()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = -1
    opt.process_rank = rank
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    model = HumanPointCloud(opt)
    if rank <= 0:
        visdomport = VisdomObserver(opt)
        total_steps = 0
        print_count = 0
        loss_stat = LossStat(len(data_loader))
    for epoch in range(opt.epoch_count, opt.total_epoch+1):
        epoch_iter = 0
        torch.manual_seed(int(time.time()))
        numpy.random.seed(int(time.time()))
        random.seed(int(time.time()))
        if rank <= 0:
            loss_stat.set_epoch(epoch)
        for i, data in enumerate(dataset):
            data = model.set_input(data)
            # print('the shape of the point clouds:', data['point_cloud'].shape)
            model.forward(data)
            model.optimize_parameters()
            if rank <= 0:
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                # get training losses
                errors = model.get_current_errors()
                loss_stat.update(errors)
                if total_steps/opt.print_freq > print_count:
                    model.basic_visualize(epoch, total_steps)
                    # visuals = model.get_current_visuals(epoch, total_steps)
                    # visdomport.display_current_results(visuals)
                    visuals = model.get_current_visuals(epoch, total_steps)
                    visdomport.plot_current_errors(epoch, float(epoch_iter)/dataset_size, errors)
                    visdomport.display_current_results(visuals)
                    loss_stat.print_loss(epoch_iter)
                    print_count += 1
        if rank <= 0:
            if epoch % opt.save_epoch_freq == 0:
                print("saving model at the end of epoch {epoch}, iters {total_steps}")
                model.save(epoch, epoch)
        model.update_learning_rate(epoch)


if __name__ == '__main__':
    main()
