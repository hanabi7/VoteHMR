#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : nn/dataset.py.bak
# Author            : Anonymous
# Date              : 18.06.2020
# Last Modified Date: 18.06.2020
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
import os
import os.path as osp

# load poses and shapes
def load_body_data(mocap_fpath, gender=['male'], sample_rate=20):
    smpl_data = np.load(mocap_fpath)
    # load MoSHed data from CMU Mocap (only the given idx is loaded)
    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))

    nseqs = 0
    cmu_parms = {}
    for name in cmu_keys:
        for seq in smpl_data.files:
            if seq == ('pose_' + name):
                cmu_parms[seq.replace('pose_', '')] = {
                    'poses':smpl_data[seq][::sample_rate],
                    'trans':smpl_data[seq.replace('pose_','trans_')][::sample_rate]}
                nseqs += len(smpl_data[seq])
    # cmu_parms is a dict
    # load all SMPL shapes
    fshapes = []
    for g in gender:
        fshapes.append(smpl_data['%sshapes' % g])
    fshapes = np.concatenate(fshapes, axis=0)

    #  print("******************************")
    #  print("cmu_sequences#%d"%(len(cmu_keys)))
    #  print("cmu_frames#%d"%(nseqs))
    #  print("cmu_fshapes#%s"%str(fshapes.shape))
    return (cmu_parms, fshapes)

##  data loader, generated randomly
class HumanDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        if self.isTrain:
            split = "Train"
            self.nsample = opt.train_sample_number
        else:
            split = "test"
            self.nsample = opt.test_sample_number
        self.surreal_use_female = opt.surreal_use_female
        self.surreal_smpl_path = opt.surreal_smpl_path
        self.annotation_path = opt.surreal_annotation_path
        self.surreal_smpl_filename = osp.join(self.annotation_path, self.surreal_smpl_path)
        self.datasets_path = opt.datasets_path
        # self.render = load_model(cfg.DATA.SMPL_MODEL_PATH)
        # print("Load smpl render.")
        if not osp.exists(self.datasets_path):
            self.load_smpl_data()

        self.shapes, self.poses = self.load_synthesis_dataset(split)
        self.shapes = torch.from_numpy(self.shapes).float()
        self.shapes = self.shapes[:self.nsample, :]
        self.poses = torch.from_numpy(self.poses).float()
        self.poses = self.poses[:self.nsample, :]
        print("**************************************************")
        print("#nsample: ", self.nsample)

    def filter_positive_elbow_and_knee(self, shapes, poses):
        """
        Refer to Keep it SMPL to filter invalid poses.
        """
        idxs = []
        part_id = [4, 5, 18, 19]
        for ii in range(poses.shape[0]):
            pose = poses[ii].reshape(24, 3)
            #  if np.any(np.sum(pose[part_id], axis=1)<0.1):
            if np.any(np.all(pose[part_id]>0.1, axis=1)):
                pass
            else:
                idxs.append(ii)
        print("Valid poses: ", len(idxs))
        return shapes[idxs], poses[idxs]

    def load_smpl_data(self):
        cmu_seqs, fshapes = load_body_data(
            self.surreal_smpl_filename, sample_rate=100)
        train_start_pos = 0
        train_end_pos = 0.816326530
        test_start_pos = 0.836734594
        test_end_pos = 1

        ## shapes for the split
        #  fshapes = fshapes[int(start_pos*len(fshapes)):int(end_pos*len(fshapes))]

        ## pose for the split
        nseqs = len(cmu_seqs.keys())
        # print('keys:', cmu_seqs.keys())
        # nseqs: number of seqs used in the smpl_data.npz
        st0 = np.random.get_state()
        np.random.seed(100)
        keys_rand = [list(cmu_seqs.keys())[idx] for idx in np.random.randint(0,nseqs,size=(nseqs,))]
        np.random.set_state(st0)
        train_poses = []
        test_poses = []
        for k in keys_rand[int(train_start_pos*nseqs): int(train_end_pos*nseqs)]:
            train_poses.append(cmu_seqs[k]["poses"])
            #  trans.append(cmu_seqs[k]["trans"])
        for k in keys_rand[int(test_start_pos*nseqs): int(test_end_pos*nseqs)]:
            test_poses.append(cmu_seqs[k]["poses"])
        train_poses = np.concatenate(train_poses, axis=0)
        test_poses = np.concatenate(test_poses, axis=0)
        train_sampled_poses = train_poses.shape[0]
        test_sampled_poses = test_poses.shape[0]
        ## shuffle poses and generate shapes
        np.random.shuffle(train_poses)
        np.random.shuffle(test_poses)
        train_idxs = np.random.randint(0, fshapes.shape[0], size=(train_sampled_poses,))
        test_idxs = np.random.randint(0, fshapes.shape[0], size=(test_sampled_poses, ))
        train_fshapes = fshapes[train_idxs]
        test_fshapes = fshapes[test_idxs]
        train_fshapes, train_poses = self.filter_positive_elbow_and_knee(train_fshapes, train_poses)
        test_fshapes, test_poses = self.filter_positive_elbow_and_knee(test_fshapes, test_poses)
        data_dict = {}
        data_dict['train_poses'] = train_poses
        data_dict['train_shapes'] = train_fshapes
        data_dict['test_poses'] = test_poses
        data_dict['test_shapes'] = test_fshapes
        np.save(self.datasets_path, data_dict)

    def load_synthesis_dataset(self, split):
        data_dict = np.load(self.datasets_path, allow_pickle=True)
        if split == 'Train':
            poses = data_dict.item().get('train_poses')
            shapes = data_dict.item().get('train_shapes')
        else:
            poses = data_dict.item().get('test_poses')
            shapes = data_dict.item().get('test_shapes')
        return shapes, poses

    def __getitem__(self, index):
        gt_param = torch.cat((self.shapes[index, :], self.poses[index, :]), dim=0)
        single_dict = dict(
            gt_param=gt_param
        )
        return single_dict

    def __len__(self):
        return self.poses.shape[0]
