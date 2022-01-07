from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import shutil
import time
from datetime import datetime


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossStat(object):
    def __init__(self, num_data):
        self.total_losses = AverageMeter()
        self.of_l1_loss = AverageMeter()
        self.joints3d_loss_before = AverageMeter()
        self.smpl_loss = AverageMeter()
        self.vertex_loss = AverageMeter()
        self.joints3d_loss_after = AverageMeter()
        self.segment_loss = AverageMeter()
        self.joints3d_loss_last = AverageMeter()
        self.offset_dir_loss = AverageMeter()
        self.chamfer_loss = AverageMeter()
        self.adversarial_loss = AverageMeter()
        self.generate_loss = AverageMeter()
        self.num_data = num_data
        self.has_offset_loss = False
        self.has_joint3d_loss_before = False
        self.has_smpl_loss = False
        self.has_vertex_loss = False
        self.has_joint3d_loss_after = False
        self.has_segment_loss = False
        self.has_joint3d_loss_last = False
        self.has_direction_loss = False
        self.has_chamfer_loss = False
        self.has_adversarial_loss = False
        self.has_generate_loss = False

    def set_epoch(self, epoch):
        self.epoch = epoch

    def update(self, errors):
        self.total_losses.update(errors['total_loss'])
        if 'of_l1_loss' in errors:
            self.of_l1_loss.update(errors['of_l1_loss'])
            self.has_offset_loss = True
        if 'joints3d_loss_before' in errors:
            self.joints3d_loss_before.update(errors['joints3d_loss_before'])
            self.has_joint3d_loss_before = True
        if 'smpl_loss' in errors:
            self.smpl_loss.update(errors['smpl_loss'])
            self.has_smpl_loss = True
        if 'vertex_loss' in errors:
            self.vertex_loss.update(errors['vertex_loss'])
            self.has_vertex_loss = True
        if 'joints3d_loss_after' in errors:
            self.joints3d_loss_after.update(errors['joints3d_loss_after'])
            self.has_joint3d_loss_after = True
        if 'segment_loss' in errors:
            self.segment_loss.update(errors['segment_loss'])
            self.has_segment_loss = True
        if 'joints3d_loss_last' in errors:
            self.joints3d_loss_last.update(errors['joints3d_loss_last'])
            self.has_joint3d_loss_last = True
        if 'offset_dir_loss' in errors:
            self.offset_dir_loss.update(errors['offset_dir_loss'])
            self.has_direction_loss = True
        if 'chamfer_loss' in errors:
            self.chamfer_loss.update(errors['chamfer_loss'])
            self.has_chamfer_loss = True
        if 'generate_loss' in errors:
            self.generate_loss.update(errors['generate_loss'])
            self.has_generate_loss = True
        if 'adversarial_loss' in errors:
            self.adversarial_loss.update(errors['adversarial_loss'])
            self.has_adversarial_loss = True

    def print_loss(self, epoch_iter):
        print_content = 'Epoch:[{}][{}/{}]\t' + \
                        'Total Loss {tot_loss.val:.4f}({tot_loss.avg:.4f})\t'
        print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                             tot_loss=self.total_losses)
        if self.has_joint3d_loss_before:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Joint3D Before Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.joints3d_loss_before)

        if self.has_offset_loss:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Offset Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.of_l1_loss)
        if self.has_smpl_loss:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Smpl Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.smpl_loss)
        if self.has_vertex_loss:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Vertex Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.vertex_loss)
        if self.has_direction_loss:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Dir Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.offset_dir_loss)
        if self.has_segment_loss:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Segment Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.segment_loss)
        if self.has_joint3d_loss_last:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Joint3D Last Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.joints3d_loss_last)
        if self.has_chamfer_loss:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Chamfer Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.chamfer_loss)
        if self.has_adversarial_loss:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Adversarial Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.adversarial_loss)
        if self.has_generate_loss:
            print_content += '\nEpoch:[{}][{}/{}]\t' + \
                             'Generate Loss {dp_align_loss.val:.4f}({dp_align_loss.avg:.4f})\t'
            print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
                                                 dp_align_loss=self.generate_loss)
        print(print_content)


class TimeStat(object):
    def __init__(self, total_epoch=100):
        self.data_time = AverageMeter()
        self.forward_time = AverageMeter()
        self.visualize_time = AverageMeter()
        self.total_time = AverageMeter()
        self.total_epoch = total_epoch

    def epoch_init(self, epoch):
        self.data_time_epoch = 0.0
        self.forward_time_epoch = 0.0
        self.visualize_time_epoch = 0.0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.forward_start_time = -1
        self.visualize_start_time = -1
        self.epoch = epoch

    def stat_data_time(self):
        self.forward_start_time = time.time()
        self.data_time_epoch += (self.forward_start_time - self.start_time)

    def stat_forward_time(self):
        self.visualize_start_time = time.time()
        self.forward_time_epoch += (self.visualize_start_time - self.forward_start_time)

    def stat_visualize_time(self):
        visualize_end_time = time.time()
        self.start_time = visualize_end_time
        self.visualize_time_epoch += visualize_end_time - self.visualize_start_time

    def stat_epoch_time(self):
        epoch_end_time = time.time()
        self.epoch_time = epoch_end_time - self.epoch_start_time

    def print_stat(self):
        self.data_time.update(self.data_time_epoch)
        self.forward_time.update(self.forward_time_epoch)
        self.visualize_time.update(self.visualize_time_epoch)

        time_content = f"End of epoch {self.epoch} / {self.total_epoch} \t" \
                       f"Time Taken: data {self.data_time.avg:.2f}, " \
                       f"forward {self.forward_time.avg:.2f}, " \
                       f"visualize {self.visualize_time.avg:.2f}, " \
                       f"Total {self.epoch_time:.2f} \n"
        time_content += f"Epoch {self.epoch} compeletes in {datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}"
        print(time_content)