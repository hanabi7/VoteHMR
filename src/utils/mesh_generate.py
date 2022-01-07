import numpy as np
import trimesh
import pickle
from scipy.io import loadmat
import cv2
import subprocess as sp
import torch
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight

colors = {
    # colorbline/print/copy safe:
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
}


class MeshGenerator():
    def __init__(self, opt):
        self.opt = opt
        self.original_smpl_male_filename = opt.original_smpl_male_filename
        with open(self.original_smpl_male_filename, 'rb') as f:
            self.params = pickle.load(f, encoding='latin1')
        self.faces = self.params['f']
        self.color = colors['light_blue']

    def _create_renderer(self,
                         w=640,
                         h=480,
                         rt=np.zeros(3),
                         t=np.zeros(3),
                         f=None,
                         c=None,
                         k=None,
                         near=.5,
                         far=10.
                         ):
        f = np.array([w, w]) / 2. if f is None else f
        c = np.array([w, h]) / 2. if c is None else c
        k = np.zeros(5) if k is None else k
        rn = ColoredRenderer()
        rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
        rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
        return rn

    def simple_renderer(self,
                        verts,
                        yrot=np.radians(120),
                        color=colors['light_pink']):
        # Rendered model color
        self.rn = self._create_renderer()
        self.rn.set(v=verts, f=self.faces, vc=color, bgcolor=np.ones(3))
        albedo = self.rn.vc

        # Construct Back Light (on back right corner)
        self.rn.vc = LambertianPointLight(
            f=self.rn.f,
            v=self.rn.v,
            num_verts=len(self.rn.v),
            light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Left Light
        self.rn.vc += LambertianPointLight(
            f=self.rn.f,
            v=self.rn.v,
            num_verts=len(self.rn.v),
            light_pos=_rotateY(np.array([800, 10, 300]), yrot),
            vc=albedo,
            light_color=np.array([1, 1, 1]))

        # Construct Right Light
        self.rn.vc += LambertianPointLight(
            f=self.rn.f,
            v=self.rn.v,
            num_verts=len(self.rn.v),
            light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
            vc=albedo,
            light_color=np.array([.7, .7, .7]))

        return self.rn.r
