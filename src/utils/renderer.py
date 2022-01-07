from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
import pickle
import cv2
import matplotlib.pyplot as plt
import os.path as osp
import os
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('AGG')
import torch
import trimesh
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as pyplot

colors_dict = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'black': (0, 0, 0),
    'yellow': (255, 255, 0),
    'purple': (128, 0, 128)
}

color_map = {
    0: 'yellow',
    1: 'green',
    2: 'blue',
    3: 'red',
    4: 'yellow',
    5: 'red',
    6: 'green',
    7: 'blue',
    8: 'blue',
    9: 'purple',
    10: 'red',
    11: 'red',
    12: 'red',
    13: 'yellow',
    14: 'black',
    15: 'yellow',
    16: 'green',
    17: 'green',
    18: 'blue',
    19: 'blue',
    20: 'red',
    21: 'red',
    22: 'black',
    23: 'black'
}

def write_mesh(verts, faces, filename, face_colors=None):
    if face_colors is not None:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, face_colors=face_colors)
    else:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)

def write_point_cloud(vertices, filename, colors=None, labels=None):
    # cost much cpu time
    assert (colors is None or labels is None)
    verts_num = vertices.shape[0]
    cmap = plt.get_cmap('seismic')
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

def write_attention_result(save_dir, point_clouds, attention_value):
    # point_clouds [2500, 3]
    # attention_value [2500, 24]
    key_points_num = attention_value.shape[1]
    points_num = attention_value.shape[0]
    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(key_points_num):
        part_attention_value = attention_value[:, i]
        # [2500]
        part_file_name = save_dir + str(i) + '_body_attention.ply'
        part_color_map = [color_map(part_attention_value[i]) for i in range(points_num)]
        write_point_cloud(point_clouds, part_file_name, colors=part_color_map)


if __name__ == '__main__':
    cmap = plt.get_cmap('seismic')
    gradient = np.linspace(0, 1, 256)
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    ax.set_title('zero to max', fontsize=14)
    pos = list(ax.get_position().bounds)
    x_text = pos[0] - 0.01
    y_text = pos[1] + pos[3] / 2.
    fig.text(x_text, y_text, 'seismic', va='center', ha='right', fontsize=10)
    ax.set_axis_off()
    plt.savefig('./color_map.png')

