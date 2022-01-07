import math
import heapq
import numpy as np
import os
import scipy.sparse as sp
from psbody.mesh import Mesh
from opendr.topology import get_vert_connectivity, get_vertices_per_edge
from src.smpl.smpl_webuser.serialization import load_model
import matplotlib
matplotlib.use('AGG')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def vertex_quadrics(mesh):
    """Computes a quadric for each vertex in the Mesh.
    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    v_quadrics = np.zeros((len(mesh.v), 4, 4,))

    # For each face...
    for f_idx in range(len(mesh.f)):
        # Compute normalized plane equation for that face
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(verts)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))
        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)
    return v_quadrics


def setup_deformation_transfer(source, target, use_normals=False):
    rows = np.zeros(3 * target.v.shape[0])
    cols = np.zeros(3 * target.v.shape[0])
    coeffs_v = np.zeros(3 * target.v.shape[0])
    coeffs_n = np.zeros(3 * target.v.shape[0])

    nearest_faces, nearest_parts, nearest_vertices = source.compute_aabb_tree().nearest(target.v, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.ravel()

    for i in range(target.v.shape[0]):
        # Closest triangle index
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source.f[f_id]

        # Closest surface point
        nearest_v = nearest_vertices[3 * i:3 * i + 3]
        # Distance vector to the closest surface point
        dist_vec = target.v[i] - nearest_v

        rows[3 * i:3 * i + 3] = i * np.ones(3)
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        if n_id == 0:
            # Closest surface point in triangle
            A = np.vstack((source.v[nearest_f])).T
            coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v)[0]
        elif n_id > 0 and n_id <= 3:
            # Closest surface point on edge
            A = np.vstack((source.v[nearest_f[n_id - 1]], source.v[nearest_f[n_id % 3]])).T
            tmp_coeffs = np.linalg.lstsq(A, target.v[i])[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest surface point a vertex
            coeffs_v[3 * i + n_id - 4] = 1.0

    #    if use_normals:
    #        A = np.vstack((vn[nearest_f])).T
    #        coeffs_n[3 * i:3 * i + 3] = np.linalg.lstsq(A, dist_vec)[0]

    # coeffs = np.hstack((coeffs_v, coeffs_n))
    # rows = np.hstack((rows, rows))
    # cols = np.hstack((cols, source.v.shape[0] + cols))
    matrix = sp.csc_matrix((coeffs_v, (rows, cols)), shape=(target.v.shape[0], source.v.shape[0]))
    return matrix


def qslim_decimator_transformer(mesh, factor=None, n_verts_desired=None):
    """Return a simplified version of this mesh.
    A Qslim-style approach is used here.
    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.v) * factor)

    Qv = vertex_quadrics(mesh)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    # from psbody.mesh.topology.connectivity import get_vertices_per_edge
    vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
    # vert_adj = sp.lil_matrix((len(mesh.v), len(mesh.v)))
    # for f_idx in range(len(mesh.f)):
    #     vert_adj[mesh.f[f_idx], mesh.f[f_idx]] = 1

    vert_adj = sp.csc_matrix((vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])),
                             shape=(len(mesh.v), len(mesh.v)))
    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate
    collapse_list = []
    nverts_total = len(mesh.v)
    faces = mesh.f.copy()
    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            # print 'found outdated cost, %.2f < %.2f' % (e[0], cost['collapse_cost'])
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            # in our face array, replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    return new_faces, mtx


def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left), num_original_verts))

    return (new_faces, mtx)


def generate_transform_matrices(mesh, factors):
    """Generates len(factors) meshes, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of meshes downsampled from mesh by a factor specified in factors.
       A: Adjacency matrix for each of the meshes
       D: Downsampling transforms between each of the meshes
       U: Upsampling transforms between each of the meshes
    """

    factors = map(lambda x: 1.0 / x, factors)
    M, A, D, U = [], [], [], []
    A.append(get_vert_connectivity(mesh.v, mesh.f))
    M.append(mesh)

    for factor in factors:
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        D.append(ds_D)
        new_mesh_v = ds_D.dot(M[-1].v)
        new_mesh = Mesh(v=new_mesh_v, f=ds_f)
        M.append(new_mesh)
        A.append(get_vert_connectivity(new_mesh.v, new_mesh.f))
        U.append(setup_deformation_transfer(M[-1], M[-2]))

    return M, A, D, U


def mesh_visualize(original_vertices, down_vertices, dir):
    # must mkdir before using this function
    origin_x, origin_y, origin_z = original_vertices[:, 0], original_vertices[:, 1], original_vertices[:, 2]
    origin_y = - origin_y
    down_x, down_y, down_z = down_vertices[:, 0], down_vertices[:, 1], down_vertices[:, 2]
    down_y = - down_y
    # offsets error between [0, 1]
    fig = plt.figure(figsize=[20, 7])
    ax = fig.add_subplot(131, projection='3d')
    ax.view_init(0, 0)
    ax.scatter(origin_x, origin_z, origin_y, s=0.1, c='k')
    ax.axis('off')
    plt.title('Original Vertices:')
    ax = fig.add_subplot(132, projection='3d')
    ax.scatter(down_x, down_z, down_y, s=0.1, c='k')
    ax.view_init(0, 0)
    ax.axis('off')
    plt.title('DownSample Vertices:')
    plt.savefig(dir)
    plt.close()


def sample_operation(filename, n_verts_desired):
    original_mesh = load_model(filename)
    # print('the content of the original mesh:', original_mesh.keys())
    """
    dict_keys(['_dirty_vars', '_itr', '_parents', '_cache', '_make_dense', '_make_sparse',
     '_depends_on_deps', 'a', 'b', 'J_transformed', 'J_regressor_prior', 'f', 'J_regressor', 
     'kintree_table', 'J', 'weights_prior', 'weights', 'vert_sym_idxs', 'posedirs', 
     'pose_training_info', 'bs_style', 'v_template', 'shapedirs', 'bs_type', 'trans', 
     'pose', 'betas', 'v_shaped', 'v_posed'])
    """
    ds_f, ds_D = qslim_decimator_transformer(original_mesh, n_verts_desired=n_verts_desired)
    new_mesh_v = ds_D.dot(original_mesh.v)
    new_mesh = Mesh(v=new_mesh_v, f=ds_f)
    new_A = get_vert_connectivity(new_mesh.v, new_mesh.f)
    new_U = setup_deformation_transfer(new_mesh, original_mesh)
    mesh_visualize(original_mesh.v, new_mesh_v, '/data1/liuguanze/depth_point_cloud/down_sample.png')
    return new_mesh, new_A, ds_D, new_U


if __name__ == '__main__':
    female_filename = '/data1/liuguanze/depth_point_cloud/src/smpl/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    n_verts_desired = 1723
    new_mesh, new_A, new_D, new_U = sample_operation(female_filename, n_verts_desired)
    new_D = new_D.A
    print('Downsample operation:', new_D.shape)
    print('type:', type(new_D))
    np.save('/data1/liuguanze/depth_point_cloud/data/down_sample.npy', new_D)