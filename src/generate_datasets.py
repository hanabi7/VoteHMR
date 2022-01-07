import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
import argparse
from src.models.smpl import batch_rodrigues


class MotionData(data.Dataset):
    def __init__(self, opt):
        super(MotionData, self).__init__()
        self.opt = opt
        self.sample_rate = opt.sample_rate
        self.cmu_keys = []
        self.gender = ['male']
        self.neutrMosh_path = opt.neutrMosh_path
        self.data_file = np.load(self.neutrMosh_path, allow_pickle=True)
        for seq in self.data_file.files:
            if seq.startswith('pose_'):
                self.cmu_keys.append(seq.replace('pose_', ''))
        nseqs = 0
        cmu_parms = {}
        for name in self.cmu_keys:
            for seq in self.data_file.files:
                if seq == ('pose_' + name):
                    cmu_parms[seq.replace('pose_', '')] = {
                        'poses': self.data_file[seq][::self.sample_rate],
                        'trans': self.data_file[seq.replace('pose_', 'trans_')][::self.sample_rate]}
                    nseqs += len(self.data_file[seq])
        # load all SMPL shapes
        fshapes = []
        for g in self.gender:
            fshapes.append(self.data_file['%sshapes' % g])
        fshapes = np.concatenate(fshapes, axis=0)
        self.poses = []
        nseqs = len(cmu_parms.keys())
        for key in cmu_parms.keys():
            self.poses.append(cmu_parms[key]['poses'])
        self.poses = np.concatenate(self.poses)
        print('the shape of the cmu_poses:', self.poses.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neutrMosh_path', type=str, default='/data/dataset/liuguanze/surreal/SURREAL/smpl_data/smpl_data.npz')
    parser.add_argument('--sample_rate', type=int, default=10)
    opt = parser.parse_args()
    motion_datasets = MotionData(opt)
