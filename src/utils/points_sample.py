import torch
import numpy as np
from scipy.io import loadmat
from PIL import Image
import pandas as pd
import os.path as osp
import os
import time
from scipy.io import loadmat
import torch
import random
import argparse
from src.utils.visual_utils import Visualizer


class PointsSample():
    def __init__(self, opt):
        self.number_points_sample = opt.number_points_sample
        self.noise_mean = opt.noise_mean
        self.noise_var = opt.noise_var

    def get_intrinsic(self):
        # These are set in Blender (datageneration/main_part1.py)
        res_x_px = 320  # *scn.render.resolution_x
        res_y_px = 240  # *scn.render.resolution_y
        f_mm = 60  # *cam_ob.data.lens
        sensor_w_mm = 32  # *cam_ob.data.sensor_width
        sensor_h_mm = sensor_w_mm * res_y_px / res_x_px  # *cam_ob.data.sensor_height (function of others)

        scale = 1  # *scn.render.resolution_percentage/100
        skew = 0  # only use rectangular pixels
        pixel_aspect_ratio = 1

        # From similar triangles:
        # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
        fx_px = f_mm * res_x_px * scale / sensor_w_mm
        fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

        # Center of the image
        u = res_x_px * scale / 2
        v = res_y_px * scale / 2

        # Intrinsic camera matrix
        K = np.array([[fx_px, skew, u], [0, fy_px, v], [0, 0, 1]])
        return K

    def get_extrinsic(self, T):
        # Take the first 3 columns of the matrix_world in Blender and transpose.
        # This is hard-coded since all images in SURREAL use the same.
        R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
        # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
        #                               (0., -1, 0., -1.0),
        #                               (-1., 0., 0., 0.),
        #                               (0.0, 0.0, 0.0, 1.0)))

        # Convert camera location to translation vector used in coordinate changes
        T_world2bcam = -1 * np.dot(R_world2bcam, T)

        # Following is needed to convert Blender camera to computer vision camera
        R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
        T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

        # Put into 3x4 matrix
        RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
        return RT, R_world2cv, T_world2cv

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [N, C]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [npoint]
        """
        device = xyz.device
        N, C = xyz.shape
        centroids = torch.zeros(npoint, dtype=torch.long).to(device)
        distance = torch.ones(N).to(device) * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
        for i in range(npoint):
            # 更新第i个最远点
            centroids[i] = farthest
            # 取出这个最远点的xyz坐标
            centroid = xyz[farthest, :].view(1, 3)
            # 计算点集中的所有点到这个最远点的欧式距离
            dist = torch.sum((xyz - centroid) ** 2, -1)
            # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
            mask = dist < distance
            distance[mask] = dist[mask]
            # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
            farthest = torch.max(distance, -1)[1]
        return centroids

    def point_clouds_sampler(self, point_clouds, segment):
        """
        :param point_clouds: [number_points, 3] tensor
        :param segment: [number_points]  tensor
        :param n_sample:
        :return:
        """
        num_availabel = point_clouds.shape[0]
        if num_availabel >= self.number_points_sample:
            """centroids = self.farthest_point_sample(point_clouds, n_sample)
            # centroids is a tensor of size [n_points]
            centroids = centroids.long()"""
            sample_index = random.sample(range(0, num_availabel), self.number_points_sample)
            sample_index = np.array(sample_index)
            sample_index = torch.from_numpy(sample_index).long()
            sample_point_clouds = point_clouds[sample_index, :]
            sample_segment = segment[sample_index]
        else:
            num_offset = self.number_points_sample - num_availabel
            rand_inds = np.random.randint(0, high=num_availabel, size=num_offset)
            rand_inds = torch.from_numpy(rand_inds).long()
            sample_point_clouds = point_clouds[rand_inds, :]
            sample_segment = segment[rand_inds]
            sample_point_clouds = torch.cat((point_clouds, sample_point_clouds), dim=0)
            sample_segment = torch.cat((segment, sample_segment), dim=0)
        return sample_point_clouds, sample_segment

    def gaussion_random_generator(self, point_clouds):
        noise = np.random.normal(self.noise_mean, self.noise_var, point_clouds.shape)
        noise = torch.from_numpy(noise)
        out = point_clouds + noise
        return out

    def point_cloud_generate(self, depth_image, seg_image, camLoc):
        """
        Inputs
            depth_image: [240, 320]
            seg_image:   [240, 320]
            camDist:     [3]
        Return:
            Pointclouds: [number_sample, 3]
            Gt_segment: [number_sample]
        """
        width, height = depth_image.shape
        # width 240, height 320
        seg_mask = np.where(seg_image != 0)
        y = seg_mask[0]
        x = seg_mask[1]
        number_points = x.shape[0]
        z = depth_image[y, x]
        x = x.reshape(number_points, 1)
        y = y.reshape(number_points, 1)
        z = z.reshape(number_points, 1)
        x = (x - 160) * z / 600
        y = (y - 120) * z / 600
        cam_coordinates = np.concatenate((x, y, z), axis=1)
        wrd_x = - cam_coordinates[:, 2] + camLoc[0]
        wrd_y = cam_coordinates[:, 1] + camLoc[1]
        wrd_z = - cam_coordinates[:, 0] + camLoc[2]
        wrd_x = wrd_x.reshape(number_points, 1)
        wrd_y = wrd_y.reshape(number_points, 1)
        wrd_z = wrd_z.reshape(number_points, 1)
        point_clouds = np.concatenate((wrd_x, wrd_y, wrd_z), axis=1)
        gt_segment = seg_image[seg_mask[0], seg_mask[1]]
        gt_segment = gt_segment.reshape(number_points)
        gt_segment = torch.from_numpy(gt_segment)
        point_clouds = torch.from_numpy(point_clouds)
        if number_points > 0:
            point_clouds, gt_segment = self.point_clouds_sampler(point_clouds, gt_segment)
            return point_clouds, gt_segment
        else:
            return point_clouds, gt_segment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synth dataset images.')
    parser.add_argument('--number_points_sample', type=int, default=2500)
    parser.add_argument('--smpl_key_points_number', type=int, default=24)
    parser.add_argument('--smpl_model_filename', type=str,
                        default='/data1/liuguanze/depth_point_cloud/data/smpl_cocoplus_neutral_no_chumpy.pkl')
    opt = parser.parse_args()
    point_sampler = PointsSample(opt)
    info_path = '/data2/liuguanze/SURREAL/data/cmu/train/run2/ung_94_16/ung_94_16_c0001_info.mat'
    depth_path = '/data2/liuguanze/SURREAL/data/cmu/train/run2/ung_94_16/ung_94_16_c0001_depth.mat'
    segm_path = '/data2/liuguanze/SURREAL/data/cmu/train/run2/ung_94_16/ung_94_16_c0001_segm.mat'
    info_file = loadmat(info_path)
    depth_file = loadmat(depth_path)
    segm_file = loadmat(segm_path)
    idx = 1
    camLoc = info_file['camLoc']
    joints3d = info_file['joints3D'][:, :, idx]
    depth_image = depth_file['depth_%d' % (idx + 1)]
    segm_image = segm_file['segm_%d' % (idx + 1)]
    point_cloud, segmentation = point_sampler.point_cloud_generate(depth_image, segm_image, camLoc)
    visualizer = Visualizer(opt)
    point_cloud = point_cloud.cuda()
    joints3d = torch.from_numpy(joints3d).cuda().transpose(1, 0)
    dir = '/data1/liuguanze/point_sample_check.png'
    visualizer.point_cloud_visualize(point_cloud, joints3d, joints3d, dir)