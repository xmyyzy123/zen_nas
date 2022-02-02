'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

define SuperResIDWEXKX class with different parameters
'''

import os
import sys
import uuid
from torch import nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import PlainNet
    from PlainNet import _get_right_parentheses_index_
    from PlainNet.super_blocks import PlainNetSuperBlockClass
    #from PlainNet.basic_blocks import GhostModule
    import global_utils
except ImportError:
    print('fail to import zen_nas modules')

# kernal x, Layers x
class SuperVoVKXLX(PlainNetSuperBlockClass):
    """ Ghost bottleneck w/ optional SE"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, kernel_size=None, osa_layers=None, no_create=False, no_reslink=False,
                 no_BN=False, no_relu=False, use_se=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bottleneck_channels = bottleneck_channels
        self.sub_layers = sub_layers
        self.osa_layers = osa_layers
        self.kernel_size = kernel_size
        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.no_relu = no_relu
        self.use_se = use_se
        if self.use_se:
            print('---debug use_se in ' + str(self))
        
        # if stride == 2:
        #     self.bottleneck_channels = self.in_channels * 2
        #     self.out_channels = self.in_channels * 2

        full_str = ''
        last_channels = in_channels
        current_stride = stride
        for i in range(self.sub_layers):
            inner_str = ''

            assert osa_layers != None
            temp_last_channels = last_channels
            for l in range(osa_layers):
                # K * K Conv
                inner_str += f'ConvKX({temp_last_channels},{self.bottleneck_channels},{self.kernel_size},{current_stride})'
                if not self.no_BN:
                    inner_str += f'BN({self.bottleneck_channels})'
                if not self.no_relu:
                    inner_str += f'RELU({self.bottleneck_channels})'
                temp_last_channels = self.bottleneck_channels
                current_stride = 1

            # concat cannels
            next_in_chs = osa_layers * self.bottleneck_channels

            # lower the dimensions
            if not self.no_reslink:
                inner_str += f'ConvKX({next_in_chs},{self.out_channels},{1},{1})'
            else:
                inner_str += f'ConvKX({bottleneck_channels},{self.out_channels},{1},{1})'
            if not self.no_BN:
                inner_str += f'BN({self.out_channels})'
            if not self.no_relu:
                inner_str += f'RELU({self.out_channels})'

            # use se
            if self.use_se:
                inner_str += f'SE({self.out_channels})'

            # reslink
            if not self.no_reslink:
                res_str = f'OSAResBlock({inner_str})'
            else:
                res_str = f'OSABlock({inner_str})'

            full_str += res_str

            last_channels = self.out_channels

        self.block_list = PlainNet.create_netblock_list_from_str(full_str, no_create=no_create,
                                                                 no_reslink=no_reslink, no_BN=no_BN, **kwargs)
        if not no_create:
            self.module_list = nn.ModuleList(self.block_list)
        else:
            self.module_list = None

    def __str__(self):
        return type(self).__name__ + f'({self.in_channels},{self.out_channels},'\
                                        f'{self.stride},{self.bottleneck_channels},{self.sub_layers})'

    def __repr__(self):
        return type(self).__name__ + f'({self.block_name}|in={self.in_channels},out={self.out_channels},'\
                                        f'stride={self.stride},btl_channels={self.bottleneck_channels},'\
                                        f'sub_layers={self.sub_layers},kernel_size={self.kernel_size})'

    def split(self, split_layer_threshold):
        """split the layer when exceeding threshold"""

        if self.sub_layers >= split_layer_threshold:
            new_sublayers_1 = split_layer_threshold // 2
            new_sublayers_2 = self.sub_layers - new_sublayers_1
            new_block_str1 = type(self).__name__ + f'({self.in_channels},{self.out_channels},'\
                                                      f'{self.stride},{self.bottleneck_channels},{new_sublayers_1})'

            new_block_str2 = type(self).__name__ + f'({self.out_channels},{self.out_channels},'\
                                                      f'{1},{self.bottleneck_channels},{new_sublayers_2})'
            return new_block_str1 + new_block_str2
        return str(self)

    def structure_scale(self, scale=1.0, channel_scale=None, sub_layer_scale=None):
        """ adjust the number to a specific multiple or range"""

        if channel_scale is None:
            channel_scale = scale
        if sub_layer_scale is None:
            sub_layer_scale = scale

        new_out_channels = global_utils.smart_round(self.out_channels * channel_scale)
        new_bottleneck_channels = global_utils.smart_round(self.bottleneck_channels * channel_scale)
        new_sub_layers = max(1, round(self.sub_layers * sub_layer_scale))

        return type(self).__name__ + f'({self.in_channels},{new_out_channels},' \
                                        f'{self.stride},{new_bottleneck_channels},{new_sub_layers})'

    @classmethod
    def create_from_str(cls, struct_str, **kwargs):
        assert cls.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len(cls.__name__ + '('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        bottleneck_channels = int(param_str_split[3])
        sub_layers = int(param_str_split[4])
        return cls(in_channels=in_channels, out_channels=out_channels, stride=stride,
                   bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                   block_name=tmp_block_name, **kwargs), struct_str[idx + 1:]

# SuperVoVK3LX
class SuperVoVK3LX(SuperVoVKXLX):
    """ kernel size 3x3"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=3,
                         no_create=no_create, **kwargs)

class SuperVoVK3L1(SuperVoVK3LX):
    """ kernel size 3x3, osa layer 1"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=1,
                         no_create=no_create, **kwargs)

class SuperVoVK3L2(SuperVoVK3LX):
    """ kernel size 3x3, osa layer 2"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=2,
                         no_create=no_create, **kwargs)

class SuperVoVK3L3(SuperVoVK3LX):
    """ kernel size 3x3, osa layer 3"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=3,
                         no_create=no_create, **kwargs)

class SuperVoVK3L4(SuperVoVK3LX):
    """ kernel size 3x3, osa layer 4"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=4,
                         no_create=no_create, **kwargs)

class SuperVoVK3L5(SuperVoVK3LX):
    """ kernel size 3x3, osa layer 5"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=5,
                         no_create=no_create, **kwargs)

# SuperVoVK5LX
class SuperVoVK5LX(SuperVoVKXLX):
    """ kernel size 5x5"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=5,
                         no_create=no_create, **kwargs)

class SuperVoVK5L1(SuperVoVK5LX):
    """ kernel size 5x5, osa layer 1"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=1,
                         no_create=no_create, **kwargs)

class SuperVoVK5L2(SuperVoVK5LX):
    """ kernel size 5x5, osa layer 2"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=2,
                         no_create=no_create, **kwargs)

class SuperVoVK5L3(SuperVoVK5LX):
    """ kernel size 5x5, osa layer 3"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=3,
                         no_create=no_create, **kwargs)

class SuperVoVK5L4(SuperVoVK5LX):
    """ kernel size 5x5, osa layer 4"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=4,
                         no_create=no_create, **kwargs)

class SuperVoVK5L5(SuperVoVK5LX):
    """ kernel size 5x5, osa layer 5"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=5,
                         no_create=no_create, **kwargs)

# SuperVoVK7LX
class SuperVoVK7LX(SuperVoVKXLX):
    """ kernel size 7x7"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         kernel_size=7,
                         no_create=no_create, **kwargs)

class SuperVoVK7L1(SuperVoVK7LX):
    """ kernel size 7x7, osa layer 1"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=1,
                         no_create=no_create, **kwargs)

class SuperVoVK7L2(SuperVoVK7LX):
    """ kernel size 7x7, osa layer 2"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=2,
                         no_create=no_create, **kwargs)

class SuperVoVK7L3(SuperVoVK7LX):
    """ kernel size 7x7, osa layer 3"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=3,
                         no_create=no_create, **kwargs)

class SuperVoVK7L4(SuperVoVK7LX):
    """ kernel size 7x7, osa layer 4"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=4,
                         no_create=no_create, **kwargs)

class SuperVoVK7L5(SuperVoVK7LX):
    """ kernel size 7x7, osa layer 5"""
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None,
                 sub_layers=None, no_create=False, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                         bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                         osa_layers=5,
                         no_create=no_create, **kwargs)


def register_netblocks_dict(netblocks_dict: dict):
    """add different kernel size block to block dict"""

    this_py_file_netblocks_dict = {
        'SuperVoVK3L1': SuperVoVK3L1,
        'SuperVoVK3L2': SuperVoVK3L2,
        'SuperVoVK3L3': SuperVoVK3L3,
        'SuperVoVK3L4': SuperVoVK3L4,
        'SuperVoVK3L5': SuperVoVK3L5,
        'SuperVoVK5L1': SuperVoVK5L1,
        'SuperVoVK5L2': SuperVoVK5L2,
        'SuperVoVK5L3': SuperVoVK5L3,
        'SuperVoVK5L4': SuperVoVK5L4,
        'SuperVoVK5L5': SuperVoVK5L5,
        'SuperVoVK7L1': SuperVoVK7L1,
        'SuperVoVK7L2': SuperVoVK7L2,
        'SuperVoVK7L3': SuperVoVK7L3,
        'SuperVoVK7L4': SuperVoVK7L4,
        'SuperVoVK7L5': SuperVoVK7L5,
    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict