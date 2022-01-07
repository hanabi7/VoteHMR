import torch
from src.datasets.smpl import SMPL
import pickle
import numpy as np

def segmentation_visualize(gt_point_clouds, labels, pred_segment, dir):
    """
    Inputs:
        gt_point_clouds: [2500, 3] tensor
        joints: [24, 3] tensor
        labels: [2500] labels
        pr_segment: [2500] labels
        dir:
    """
    fig = plt.figure(figsize=[20, 10])
    labels = labels.detach().cpu().numpy()
    pred_segment = pred_segment.detach().cpu().numpy()
    # joints = joints.detach().cpu().numpy()
    labels = labels / labels.max()
    pred_segment = pred_segment / pred_segment.max()
    point_clouds = gt_point_clouds.detach().cpu().numpy()
    x, y, z = point_clouds[:, 0], point_clouds[:, 1], point_clouds[:, 2]
    y = -y
    ax = fig.add_subplot(131, projection='3d')
    ax.view_init(0, 0)
    ax.scatter(x, z, y, s=0.1, c=labels)
    ax.axis('off')
    plt.title('the ground_truth segmentation')
    ax = fig.add_subplot(132, projection='3d')
    ax.view_init(0, 0)
    ax.scatter(x, z, y, s=0.1, c=pred_segment)
    ax.axis('off')
    plt.title('the predicted segmentation')
    plt.savefig(dir)
    plt.close()

class SMPLModel()
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            params = pickle.load(f)

            self.J_regressor = params['J_regressor']
            self.weights = np.asarray(params['weights'])
            self.posedirs = np.asarray(params['posedirs'])
            self.v_template = np.asarray(params['v_template'])
            self.shapedirs = np.asarray(params['shapedirs'])
            self.faces = np.asarray(params['f'])
            self.kintree_table = np.asarray(params['kintree_table'])
        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]
        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)
        self.verts = None
        self.J = None
        self.R = None
        self.G = None
        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        """
        Called automatically when parameters are updated.

        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0] - 1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
            )
        )
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])
        self.G = G

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.
        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].
        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].
        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
        Save the SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.

        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def segment_visualize(point, label_colors):
    """
    point: [batch_size, 6890, 3] numpy
    label: [batch_size, 6890] numpy
    """
    fig = plt.figure(figsize=[20, 7])
    x, y, z = point[:, 0], point[:, 1], point[:, 2]
    ax = fig.add_subplot(131, projection='3d')
    ax.view_init(0, 0)
    ax.scatter(x, z, y, s=0.1, c=label_colors/255.0)
    plt.title('Points on the SMPL model.')


def get_vertices_bound_to_jnts(skinning_weights, jnts):
    weights_of_interest = skinning_weights[:, jnts]
    return np.where(weights_of_interest > 0.5)

def save_part_of_smpl(smpl, vert_inds, filename):
    with open(filename, 'w') as fp:
        for vi in vert_inds:
            v = smpl.verts[vi]
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

colors_dict = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'black:': (0, 0, 0),
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

def get_vertice_segment(smpl_path, beta, pose, trans=None):
    smpl = SMPLModel(smpl_path)
    if trans is None:
        trans = np.zeros(smpl.trans_shape)
    verts = smpl.set_params(beta=beta, pose=pose, trans=trans)
    # [6890, 3]
    num_verts = verts.shape[0]
    labels = np.zeros(num_verts)
    labels_color = np.zeros(num_verts, 3)
    num_joints = 24
    for i in range(num_joints):
        part_label = get_vertices_bound_to_jnts(smpl.weights, i)
        for vi in part_label:
            labels[vi] = i
            labels_color[vi] = colors_dict[color_map[i]]
    return labels, labels_color
