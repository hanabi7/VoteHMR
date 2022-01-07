from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.options.base_options import BaseOptions


class VisualOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--compare_position', type=str, default='/data1/liuguanze/depth_point_cloud')
        self.parser.add_argument('--baseline_position', type=str, default='/data1/liuguanze/3D-CODED-master')
        self.isTrain = False


