from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from .surreal_depth_image import SurrealDepth

class CustomDatasetDataLoader(object):
    @property
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        self.dataset = SurrealDepth(opt)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=False,
            drop_last=opt.isTrain)

    def load_data(self):
        return self.dataloader

    def shuffle_data(self):
        self.dataset.shuffle_data()

    def __len__(self):
        return len(self.dataset)


class DistributedDataLoader(object):
    def initialize(self, opt):
        # print("Use distributed dataloader")
        self.dataset = SurrealDepth(opt)

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        self.train_sampler = DistributedSampler(self.dataset, world_size, rank)

        num_workers = opt.nThreads
        assert opt.batchSize % world_size == 0
        batch_size = opt.batchSize // world_size
        shuffle = False
        drop_last = opt.isTrain

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=self.train_sampler,
            drop_last=drop_last,
            pin_memory=False)

    def load_data(self):
        return self.data_loader

    def shuffle_data(self):
        self.dataset.shuffle_data()

    def __len__(self):
        return len(self.dataset)


def CreateDataLoader(opt):
    if opt.dist:
        data_loader = DistributedDataLoader()
    else:
        data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader
