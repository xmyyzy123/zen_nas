'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.

define SuperResIDW Block search space
'''
# pylint: disable=invalid-name
import os
import sys
import itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import global_utils
    from PlainNet import super_blocks, SuperGhostShuffle
except ImportError:
    print('fail to import zen_nas modules')

seach_space_block_type_list_list = [
    [SuperGhostShuffle.SuperGhostShuffleK3, 
     SuperGhostShuffle.SuperGhostShuffleK5, 
     SuperGhostShuffle.SuperGhostShuffleK7],
]

__block_type_round_channels_base_dict__ = {
    SuperGhostShuffle.SuperGhostShuffleK3: 8,
    SuperGhostShuffle.SuperGhostShuffleK5: 8,
    SuperGhostShuffle.SuperGhostShuffleK7: 8,
}

__block_type_min_channels_base_dict__ = {
    SuperGhostShuffle.SuperGhostShuffleK3: 8,
    SuperGhostShuffle.SuperGhostShuffleK5: 8,
    SuperGhostShuffle.SuperGhostShuffleK7: 8,
}


def get_select_student_channels_list(out_channels):
    """generate all possible channels"""
    the_list = [out_channels * 2.5, out_channels * 2, out_channels * 1.5, out_channels * 1.25,
                out_channels,
                out_channels / 1.25, out_channels / 1.5, out_channels / 2, out_channels / 2.5]
    the_list = [max(8, x) for x in the_list]
    the_list = [global_utils.smart_round(x, base=8) for x in the_list]
    the_list = list(set(the_list))
    the_list.sort(reverse=True)
    return the_list


def get_select_student_sublayers_list(sub_layers):
    """generate all possible sublayers"""
    the_list = [sub_layers,
                sub_layers + 1, sub_layers + 2,
                sub_layers - 1, sub_layers - 2, ]
    the_list = [max(0, round(x)) for x in the_list]
    the_list = list(set(the_list))
    the_list.sort(reverse=True)
    return the_list


# pylint: disable=too-many-locals
def gen_search_space(block_list, block_id):
    """ generate all possible blocks

        :param block_list (list): block list
        :param block_id (int): block index
        :return list(list())
    """

    the_block = block_list[block_id]
    student_blocks_list_list = []

    if isinstance(the_block, super_blocks.SuperConvKXBNRELU):
        student_blocks_list = []

        if the_block.kernel_size == 1:  # last fc layer, never change fc
            student_out_channels_list = [the_block.out_channels]
        else:
            student_out_channels_list = get_select_student_channels_list(the_block.out_channels)

        for student_out_channels in student_out_channels_list:
            tmp_block_str = type(the_block).__name__ + f'({the_block.in_channels},'\
                                                          f'{student_out_channels},'\
                                                          f'{the_block.stride},1)'
            student_blocks_list.append(tmp_block_str)
        student_blocks_list = list(set(student_blocks_list))
        assert len(student_blocks_list) >= 1
        student_blocks_list_list.append(student_blocks_list)
    else:
        for student_block_type_list in seach_space_block_type_list_list:
            student_blocks_list = []
            student_out_channels_list = get_select_student_channels_list(the_block.out_channels)
            student_sublayers_list = get_select_student_sublayers_list(sub_layers=the_block.sub_layers)
            student_bottleneck_channels_list = get_select_student_channels_list(the_block.bottleneck_channels)
            for student_block_type in student_block_type_list:
                for student_out_channels, student_sublayers, student_bottleneck_channels in itertools.product(
                        student_out_channels_list, student_sublayers_list, student_bottleneck_channels_list):

                    # filter smallest possible channel for this block type
                    min_possible_channels = __block_type_round_channels_base_dict__[student_block_type]
                    channel_round_base = __block_type_round_channels_base_dict__[student_block_type]
                    student_out_channels = global_utils.smart_round(student_out_channels, channel_round_base)
                    student_bottleneck_channels = global_utils.smart_round(student_bottleneck_channels,
                                                                           channel_round_base)

                    if student_out_channels < min_possible_channels or \
                            student_bottleneck_channels < min_possible_channels:
                        continue
                    if student_sublayers <= 0:  # no empty layer
                        continue
                    tmp_block_str = student_block_type.__name__ + f'({the_block.in_channels},{student_out_channels},'\
                                                                     f'{the_block.stride},{student_bottleneck_channels},'\
                                                                     f'{student_sublayers})'
                    student_blocks_list.append(tmp_block_str)
                student_blocks_list = list(set(student_blocks_list))
                assert len(student_blocks_list) >= 1
                student_blocks_list_list.append(student_blocks_list)
    return student_blocks_list_list
