from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyElement, PlyData
import torch.utils.data as data
import os
import os.path as osp


def write_point_cloud(vertices, filename, colors=None, labels=None):
    # cost much cpu time
    assert (colors is None or labels is None)
    verts_num = vertices.shape[0]
    if labels is not None:
        labels = labels.astype(int)
        num_classes = np.max(labels) + 1
        verts = []
        for i in range(verts_num):
            point_color = colors_dict[color_map[labels[i]]]
            # point_color: (0.8950031625553446, 1.0, 0.07273877292852626, 1.0)
            verts.append((vertices[i, 0], vertices[i, 1], vertices[i, 2], int(point_color[0]), int(point_color[1]), int(point_color[2])))
        verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    elif colors is not None:
        verts = []
        for i in range(verts_num):
            verts.append((vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0]*255, colors[i, 1]*255, colors[i, 2]*255))
        verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        verts = [(vertices[i, 0], vertices[i, 1], vertices[i, 2]) for i in range(verts_num)]
        verts = np.array(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False).write(filename)


class MHAD(data.Dataset):
    def __init__(self, opt):
        self.mhad_path = opt.mhad_path
        self.mhad_keys = os.listdir(self.mhad_path)
        self.mhad_points = []
        for name in self.mhad_keys:
            sequence_path = self.mhad_path + name
            sequence_file = loadmat(sequence_path)
            point_clouds = sequence_file['points']
            self.mhad_points.append(point_clouds)
            # [number_frames, 2500, 3]
        self.mhad_points = np.concatenate(self.mhad_points, axis=0)

    def __getitem__(self, index):
        return self.mhad_points[index]

    def __len__(self):
        return self.mhad_points.shape[0]


if __name__ == '__main__':
    file_name = './s1_a2_r3_red.mat'
    pc_file = loadmat(file_name)
    point_clouds = pc_file['points']
    number_frames = point_clouds.shape[0]
    sample_number = 10
    for index in range(sample_number):
        single_pc = point_clouds[index, :, :]
        pc_name = './' + str(index) + '_.ply'
        write_point_cloud(single_pc, pc_name)
