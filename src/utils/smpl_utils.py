import h5py
import torch
import numpy as np
# from config import args
import json
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import math
from scipy import interpolate

total_theta_count = 85


def smpl_structure(inquiry_key):
    knowledge = {}

    knowledge['limb_pairs'] = [(0, 1), (1, 4), (4, 7), (7, 10),
                               (0, 2), (2, 5), (5, 8), (8, 11),
                               (0, 3), (3, 6), (6, 9),
                               (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
                               (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
                               (9, 12), (12, 15)]  #,
                               # (12, 17), (12, 16)]

    knowledge['smpl_parents'] = [[0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
                                 [3, 3, 3, 3, 0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 9, 9, 9, 13, 14, 16, 17, 18, 19]]

    knowledge['smpl_children'] = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
                                  [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 10, 11, 15, 16, 17, 15, 18, 19, 20, 21, 22, 23, 22, 23]]

    knowledge['m_l_r'] = [[0, 3, 6, 9, 12, 15],
                          [1, 4, 7, 10, 13, 16, 18, 20, 22],
                          [2, 5, 8, 11, 14, 17, 19, 21, 23]]

    knowledge['branch'] = [[0, 3, 6, 9, 12, 15],
                           [1, 4, 7, 10],
                           [13, 16, 18, 20, 22],
                           [2, 5, 8, 11],
                           [14, 17, 19, 21, 23]]

    knowledge['smpl2dp_part'] = [[1, 2],  # 0: 'Pelvis'
                                 [8, 10],  # 1: 'L_Hip'
                                 [7, 9],  # 2: 'R_Hip'
                                 [1, 2],  # 3: 'Spine1'
                                 [8, 10, 12, 14],  # 4:  'L_Knee'
                                 [7, 9, 11, 13],  # 5:  'R_Knee'
                                 [1, 2],  # 6:  'Spine2'
                                 [12, 14, 5],  # 7:  'L_Ankle'
                                 [11, 13, 6],  # 8:  'R_Ankle'
                                 [1, 2],  # 9:  'Spine3'
                                 [12, 14, 5],  # 10: 'L_Foot'
                                 [11, 13, 6],  # 11: 'R_Foot'
                                 [1, 2, 23, 24],  # 12: 'Neck'
                                 [15, 17],  # 13: 'L_Collar'
                                 [16, 18],  # 14: 'R_Collar'
                                 [23, 24],  # 15: 'Head'
                                 [15, 17],  # 16: 'L_Shoulder'
                                 [16, 18],  # 17: 'R_Shoulder'
                                 [15, 17, 19, 21],  # 18: 'L_Elbow'
                                 [16, 18, 20, 22],  # 19: 'R_Elbow'
                                 [19, 21, 4],  # 20: 'L_Wrist'
                                 [20, 22, 3],  # 21: 'R_Wrist'
                                 [19, 21, 4],  # 22: 'L_Hand'
                                 [20, 22, 3]  # 23: 'R_Hand'
                                 ]  #

    knowledge['dp2smpl_mapping'] = [[7, 8, 9, 10, 1, 2],  # 0: 'Pelvis'
                                    [1, 2, 8, 10, 12, 14],  # 1: 'L_Hip'
                                    [1, 2, 7, 9, 11, 13],  # 2: 'R_Hip'
                                    [7, 8, 9, 10, 1, 2],  # 3: 'Spine1'
                                    [1, 2, 8, 10, 12, 14],  # 4:  'L_Knee'
                                    [1, 2, 7, 9, 11, 13],  # 5:  'R_Knee'
                                    [7, 8, 9, 10, 1, 2],  # 6:  'Spine2'
                                    [8, 10, 12, 14, 5, 5],  # 7:  'L_Ankle'
                                    [7, 9, 11, 13, 6, 6],  # 8:  'R_Ankle'
                                    [7, 8, 9, 10, 1, 2],  # 9:  'Spine3'
                                    [8, 10, 12, 14, 5, 5],  # 10: 'L_Foot'
                                    [7, 9, 11, 13, 6, 6],  # 11: 'R_Foot'
                                    [1, 2, 23, 24, 23, 24],  # 12: 'Neck'
                                    [1, 2, 15, 17, 19, 21],  # 13: 'L_Collar'
                                    [1, 2, 16, 18, 20, 22],  # 14: 'R_Collar'
                                    [1, 2, 23, 24, 23, 24],  # 15: 'Head'
                                    [1, 2, 15, 17, 19, 21],  # 16: 'L_Shoulder'
                                    [1, 2, 16, 18, 20, 22],  # 17: 'R_Shoulder'
                                    [1, 2, 15, 17, 19, 21],  # 18: 'L_Elbow'
                                    [1, 2, 16, 18, 20, 22],  # 19: 'R_Elbow'
                                    [15, 17, 19, 21, 4, 4],  # 20: 'L_Wrist'
                                    [16, 18, 20, 22, 3, 3],  # 21: 'R_Wrist'
                                    [15, 17, 19, 21, 4, 4],  # 22: 'L_Hand'
                                    [16, 18, 20, 22, 3, 3]  # 23: 'R_Hand'
                                    ]  #
    knowledge['virtual_nodes_inner_pairs'] = [(24, 25), (24, 26), (24, 27), (27, 28), (27, 29)]
    knowledge['virtual_nodes_outer_pairs'] = [(0, 27), (1, 27), (2, 27), (3, 27),
                                              (5, 28), (8, 28), (11, 28),
                                              (4, 29), (7, 29), (10, 29),
                                              (6, 24), (9, 24), (14, 24), (12, 24), (15, 24), (13, 24),
                                              (17, 25), (19, 25), (21, 25), (23, 25),
                                              (16, 26), (18, 26), (20, 26), (22, 26)]
    # knowledge['dp2smpl_mapping'] = [range(1, 25)] * 24  #

    return knowledge[inquiry_key]


def load_mean_theta():
    mean = np.zeros(total_theta_count, dtype=np.float)

    mean_values = h5py.File('../DensePoseData/SMPL_data/neutral_smpl_mean_params.h5')
    mean_pose = mean_values['pose']
    mean_pose[:3] = 0
    mean_shape = mean_values['shape']
    mean_pose[0] = np.pi

    # init sacle is 0.9
    mean[0] = 0.9

    mean[3:75] = mean_pose[:]
    mean[75:] = mean_shape[:]

    return mean