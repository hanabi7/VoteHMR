import sys
import os.path as osp
import os
import h5py
import numpy as np
import torch
import random
import torch.utils.data as data
from scipy.io import loadmat
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
root_path = os.path.split(root_path)[0]
sys.path.append(root_path)
from src.datasets.smpl import SMPL
import math
import argparse
import transforms3d


class SurrealDepth(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.data_list = []
        self.use_generated_data_file = opt.use_generated_data_file
        self.surreal_save_path = opt.surreal_save_path
        if self.isTrain:
            self.split = 'train'
        else:
            self.split = 'test'
        self.surreal_split_name = ['run0', 'run1', 'run2']
        self.surreal_train_sequence_number = opt.surreal_train_sequence_number
        self.surreal_train_sequence_sample_number = opt.surreal_train_sequence_sample_number
        self.surreal_test_sequence_number = opt.surreal_test_sequence_number
        self.surreal_test_sequence_sample_number = opt.surreal_test_sequence_sample_number
        self.data_path = opt.surreal_dataset_path + self.split
        if not self.use_generated_data_file:
            cmu_keys, cmu_keys_name = self.load_surreal_data(self.data_path)
            print('the lens of cmu_keys:', len(cmu_keys))
            complete_sequence = self.sequence_complete(cmu_keys, cmu_keys_name)
            random.shuffle(complete_sequence)
            if self.split == 'train':
                sequence = complete_sequence[:self.surreal_train_sequence_number]
            else:
                sequence = complete_sequence[:self.surreal_test_sequence_number]
            number_sequence = len(sequence)
            for i in range(number_sequence):
                self.single_sequence_process(sequence[i])
                print(i)
            self.generate_datasets()
            # print('the content of cmu_keys:', cmu_keys)
        else:
            if self.split == 'train':
                file_name = self.surreal_save_path + self.split + '_annotation.npy'
                self.data_list = np.load(file_name, allow_pickle=True)
            else:
                file_name = self.surreal_save_path + self.split + '_annotation.npy'
                self.data_list = np.load(file_name, allow_pickle=True)

    def rotateBody(self, RzBody, pelvisRotVec):
        angle = np.linalg.norm(pelvisRotVec)
        Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
        globRotMat = np.dot(RzBody, Rpelvis)
        R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
        globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
        globRotVec = globRotAx * globRotAngle
        return globRotVec

    def sequence_complete(self, cmu_keys, cmu_keys_name):
        length = len(cmu_keys)
        complete_sequence = []
        for i in range(length):
            sequence_path = cmu_keys[i]
            sequence_name = cmu_keys_name[i]
            number_files = len(os.listdir(sequence_path))
            number_sequence = int(number_files / 4)
            for j in range(number_sequence):
                depth_filename = sequence_name + '_c00%02d' % (j + 1) + '_depth.mat'
                info_filename = sequence_name + '_c00%02d' % (j + 1) + '_info.mat'
                depth_filepath = sequence_path + '/' + depth_filename
                info_filepath = sequence_path + '/' +info_filename
                if os.path.exists(depth_filepath):
                    file_dict = dict(
                        depth=depth_filepath,
                        info=info_filepath,
                        sequence_name=sequence_name + '%02d' % (j+1)
                    )
                    complete_sequence.append(file_dict)
        print('sequence_lenghth:', len(complete_sequence))
        return complete_sequence

    def data_generate(self, depth_filename, info_filename, sample_number, sequence_name):
        depth_file = loadmat(depth_filename)
        info_file = loadmat(info_filename)
        pose = info_file['pose']
        # [10, 100]
        shape = info_file['shape']
        # [72, 100]
        number_frame = pose.shape[1]
        joints3d = info_file['joints3D']
        # joints3d [3, 24, 100]
        joints2D = info_file['joints2D']
        # joints2d [2, 24, 100]
        zrot = info_file['zrot']
        zrot = zrot[0][0]
        RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                           (math.sin(zrot), math.cos(zrot), 0),
                           (0, 0, 1)))
        # gender is a numpy array of size [number_frame, 1]
        if joints3d.ndim == 3:
            for i in range(min(sample_number, number_frame)):
                idx = i
                depth_image = depth_file['depth_%d' % (idx + 1)]
                depth_image = self.depth_image_process(depth_image)
                gt_joints2d = torch.from_numpy(joints2D[:, :, idx]).transpose(1, 0)
                gt_joints3d = torch.from_numpy(joints3d[:, :, idx]).transpose(1, 0)
                pose_param = pose[:, idx]
                pose_param[0:3] = self.rotateBody(RzBody, pose_param[0:3])
                pose_param = torch.from_numpy(pose_param)
                shape_param = torch.from_numpy(shape[:, idx])
                sequence_path = self.surreal_save_path + self.split + '/' + sequence_name + '/'
                if not osp.exists(osp.join(self.surreal_save_path, self.split)):
                    os.mkdir(osp.join(self.surreal_save_path, self.split))
                if not osp.exists(sequence_path):
                    os.mkdir(sequence_path)
                depth_image_path = sequence_path + '_' + str(i) + '.npy'
                np.save(depth_image_path, depth_image)
                single_data = dict(
                    depth_image_path=depth_image_path,
                    gt_joints2d=gt_joints2d,
                    gt_joints3d=gt_joints3d,
                    gt_pose=pose_param,
                    gt_shape=shape_param
                )
                self.data_list.append(single_data)
            print('sequence_complete')

    def depth_image_process(self, depth_image):
        max_value = depth_image.max()
        segment_mask = depth_image == max_value
        depth_image[segment_mask] = 0
        return depth_image

    def single_sequence_process(self, sequence_path):
        """
        :param data_path: the data_path for the small sequence
        :return:
        """
        depth_path = sequence_path['depth']
        info_path = sequence_path['info']
        sequence_name = sequence_path['sequence_name']
        if self.split == 'train':
            self.data_generate(depth_path, info_path, self.surreal_train_sequence_sample_number, sequence_name)
        else:
            self.data_generate(depth_path, info_path, self.surreal_test_sequence_sample_number, sequence_name)

    def load_surreal_data(self, path):
        """
        Input:
            path: /data/liuguanze/datasets/surreal/SURREAL/data/cmu/spilt
        Return:
            cmu_keys [list of sequences path]
        """
        cmu_keys = []
        cmu_keys_name = []
        for filename in self.surreal_split_name:
            added_path = path + '/' + filename + '/'
            for sequence_name in os.listdir(added_path):
                # print('the content of the sequence_path:', sequence_path)
                sequence_path = added_path + sequence_name
                if os.path.isdir(sequence_path):
                    cmu_keys_name.append(sequence_name)
                    cmu_keys.append(sequence_path)
        return cmu_keys, cmu_keys_name
        # number of sequences of the split of the surreal datasets

    def __getitem__(self, index):
        single_data = self.data_list[index]
        depth_image = np.load(single_data['depth_image_path'], allow_pickle=True)
        depth_image = torch.from_numpy(depth_image)
        single_data['depth_image'] = depth_image

    def __len__(self):
        return len(self.data_list)

    def generate_datasets(self):
        file_name = self.surreal_save_path + self.split + '_annotation.npy'
        np.save(file_name, self.data_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--use_generated_data_file', action='store_true')
    parser.add_argument('--surreal_save_path', type=str, default='/data1/liuguanze/datasets/depth_image/')
    parser.add_argument('--surreal_train_sequence_number', type=int, default=10000)
    parser.add_argument('--surreal_train_sequence_sample_number', type=int, default=20)
    parser.add_argument('--surreal_test_sequence_number', type=int, default=100)
    parser.add_argument('--surreal_test_sequence_sample_number', type=int, default=100)
    parser.add_argument('--surreal_dataset_path', type=str, default='/data1/liuguanze/datasets/SURREAL/data/cmu/')
    opt = parser.parse_args()
    depth_image = SurrealDepth(opt)
