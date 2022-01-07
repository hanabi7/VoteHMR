from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--print_freq', type=int, default=2048,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--total_epoch', type=int, default=20,
                                 help='the number of epoch we need to train the model')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? hset to latest to use latest cached model')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr_e', type=float, default=1e-5, help='initial learning rate for encoder') # In HMR, the paper says the lr is setting to 1e-5, but the code use 1e-3
        self.parser.add_argument('--loss_3d_weight_before', type=int, default=10, help='loss weight for 3d loss before')
        self.parser.add_argument('--loss_3d_weight_after', type=int, default=10, help='loss weight for 3d loss after')
        self.parser.add_argument('--loss_offset_weight', type=float, default=10, help='loss weight for the offset loss')
        self.parser.add_argument('--loss_smpl_weight', type=int, default=10, help='loss weight for the smpl loss')
        self.parser.add_argument('--loss_vertex_weight', type=float, default=10, help='loss weight for the vertex loss.')
        self.parser.add_argument('--loss_segment_weight', type=int, default=10, help='loss weight for the segment loss.')
        self.parser.add_argument('--loss_orthogonal_weight', type=int, default=10, help='loss weight for the orthogonal loss.')
        self.parser.add_argument('--chamfer_loss_weight', type=float, default=10, help='the loss weight for the chamfer loss.')
        self.parser.add_argument('--loss_generate_weight', type=float, default=10.0, help='the loss weight for the generate_loss')
        self.parser.add_argument('--loss_adversarial_weight', type=float, default=10.0, help='the loss weight for the adversarial loss')
        self.parser.add_argument('--loss_dir_weight', type=float, default=0.1,
                                 help='loss weight for the loss direction loss.')
        self.parser.add_argument('--evaluate_epoch', type=int, default=20, help='which epoch to evaluate the result.')
        self.parser.add_argument('--continue_train', action='store_true', help='whether to continue training')
        self.parser.add_argument('--pretrained_weights', type=str, default=None, help='path to pretrained weights')

        self.isTrain = True
