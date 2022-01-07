import torch
import torch.nn as nn
import torch.utils.data as data
import torch
import torch.utils.data as data
from src.options.train_options import TrainOptions
import h5py
from src.utils.points_sample import PointsSample
from src.datasets.smpl import SMPL
from scipy.io import loadmat
import argparse


class Dfaust(data.Dataset):
    def __init__(self, opt):
        super(Dfaust, self).__init__()
        self.dfaust_save_path = opt.dfaust_save_path
        self.dfaust_data_path = opt.dfaust_data_path
        self.dfaust_depth_path = opt.dfaust_depth_path
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.split = 'train'
        else:
            self.split = 'test'
        self.use_generated_data_file = opt.use_generated_data_file
        if not self.use_generated_data_file:
            self.data_list = []
            self.smpl_model = SMPL(opt.smpl_model_filename, batchSize=batchSize)
            self.point_sampler = PointsSample(opt)
            male_path = self.dfaust_data_path + 'registrations_m.hdf5'
            female_path = self.dfaust_data_path + 'registrations_f.hdf5'
            male_file = h5py.File(male_path, 'r')
            female_file = h5py.File(female_path, 'r')
            cmu_keys = []
            for name in list(male_file.keys()):
                if not name.endswith('faces'):
                    name = 'male' + name
                    cmu_keys.append(name)
            for name in list(female_file.keys()):
                if not name.endswith('faces'):
                    name = 'female' + name
                    cmu_keys.append(name)
            self.generate_datasets(cmu_keys)
            save_filename = self.dfaust_depth_path + self.split + '_annotations.npy'
            np.save(save_filename, self.data_list)
        else:
            save_filename = self.dfaust_depth_path + self.split + '_annotations.npy'
            self.data_list = np.load(save_filename, allow_pickle=True)

    def generate_datasets(self, cmu_keys):
        for name in cmu_keys:
            sequence_name = name + '_c0001_'
            info_file_name = sequence_name + 'info.mat'
            segm_file_name = sequence_name + 'segm.mat'
            depth_file_name = sequence_name + 'depth.mat'
            info_file = loadmat(info_file_name)
            segm_file = loadmat(segm_file_name)
            depth_file = loadmat(depth_file_name)
            number_frames = len(depth_file.keys())
            for index in range(number_frames):
                depth_image = depth_file['depth_%d' % (index + 1)]
                segm_image = segm_file['segm_%d' % (index + 1)]
                camLoc = info_file['camLoc']
                vertices = info_file['vertice'][:, :, index]
                vertices = torch.from_numpy(vertices)
                gt_joints3d = self.smpl_model.get_current_joints(vertices)
                data_file_dir = sequence_save_path + '%d' % index + '_dict.pkl'
                point_clouds, gt_segment = self.point_sampler.point_cloud_generate(depth_image, segm_image, camLoc)
                if point_clouds.shape[0] == 2500:
                    gt_segment = gt_segment - 1
                    num_segment = len(np.unique(gt_segment))
                    if num_segment > 18:
                        single_data = dict(
                            data_dir = data_file_dir,
                            gt_joints3d = gt_joints3d,
                            camLoc=camLoc,
                            name = name
                        )
                        data_file = {
                            'point_cloud': point_clouds,
                            'gt_segment': gt_segment,
                            'gt_vertices': vertices
                        }
                        with open(data_file_dir, 'wb') as f:
                            pickle.dump(data_file, f)
                        self.data_list.append(single_data)

    def __getitem__(self, item):
        single_data = self.data_list[item]
        data_dir = single_data['data_dir']
        with open(data_dir, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        single_data['point_cloud'] = data_dict['point_cloud']
        single_data['gt_segment'] = data_dict['gt_segment']
        single_data['gt_vertices'] = data_dict['gt_vertices']
        return single_data

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dfaust_save_path', type=str,
                             default='/data2/liuguanze/dfaust/out/')
    parser.add_argument('--dfaust_data_path', type=str,
                             default='/data2/liuguanze/dfaust/')
    parser.add_argument('--dfaust_depth_path', type=str,
                             default='/data3/liuguanze/dfaust/')
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--use_generated_data_file', action='store_true')
    opt = parser.parse_args()
    dfaust = Dfaust(opt)
