# -*- coding: utf-8 -*-
# File: config.py

import numpy as np
import os
import six
import pprint

__all__ = ['config', 'finalize_configs']


class AttrDict():

    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed and name not in self.__dict__:
            raise AttributeError(
                "Config was freezed! Unknown config: {}".format(name))
        super().__setattr__(name, value)

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1, width=100, compact=True)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {k: v.to_dict() if isinstance(v, AttrDict) else v
                for k, v in self.__dict__.items() if not k.startswith('_')}

    def update_args(self, args):
        """Update from command line args. """
        print(args)
        for cfg in args:
            keys, v = cfg.split('=', maxsplit=1)
            keylist = keys.split('.')

            dic = self
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(keys)
                dic = getattr(dic, k)
            key = keylist[-1]

            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def freeze(self, freezed=True):
        self._freezed = freezed
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze(freezed)

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


def update_args_from_dict(dic, args):
    """Update from command line args. """
    for key in args:
        v = args[key]
        assert key in dir(dic), "Unknown config key: {}".format(key)
        if type(v) is dict:
            update_args_from_dict(getattr(dic, key), v)
        else:
            setattr(dic, key, v)


config = AttrDict()
_C = config     # short alias to avoid coding

# mode flags ---------------------

# dataset -----------------------
# All TRAIN dataset will be concatenated for training.
# This two config will be populated later by the dataset loader:
_C.DATA.NUM_CATEGORY = 1  # without the background class (e.g., 80 for COCO)

# basemodel ----------------------
_C.BACKBONE.RESNET_NUM_BLOCKS = [3, 4, 23, 3]     # for resnet101

# Use a base model with TF-preferred padding mode,
# which may pad more pixels on right/bottom than top/left.
# See https://github.com/tensorflow/tensorflow/issues/18213
# In tensorpack model zoo, ResNet models with TF_PAD_MODE=False are marked with "-AlignPadding".
# All other models under `ResNet/` in the model zoo are using TF_PAD_MODE=True.
# Using either one should probably give the same performance.
# We use the "AlignPadding" one just to be consistent with caffe2.
_C.BACKBONE.TF_PAD_MODE = False

# preprocessing --------------------
_C.PREPROC.TEST_LONG_EDGE_SIZE = 1600
_C.PREPROC.MAX_SIZE = 1600
# _C.PREPROC.TEST_LONG_EDGE_SIZE = 1056
# _C.PREPROC.MAX_SIZE = 1056
# _C.PREPROC.TEST_LONG_EDGE_SIZE = 2048
# _C.PREPROC.MAX_SIZE = 2048
# _C.PREPROC.TEST_LONG_EDGE_SIZE = 2560
# _C.PREPROC.MAX_SIZE = 2560
# mean and std in RGB order.
# Un-scaled version: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_C.PREPROC.PIXEL_MEAN = [123.675, 116.28, 103.53]
_C.PREPROC.PIXEL_STD = [58.395, 57.12, 57.375]

# anchors -------------------------
_C.RPN.ANCHOR_SIZES = (16, 32, 64, 128, 256)   # sqrtarea of the anchor box
_C.RPN.ANCHOR_RATIOS = (0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2, 5, 10)

# rpn training -------------------------
_C.RPN.PROPOSAL_NMS_THRESH = 0.7
# Anchors which overlap with a crowd box (IOA larger than threshold) will be ignored.
# Setting this to a value larger than 1.0 will disable the feature.
# It is disabled by default because Detectron does not do this.

# RPN proposal selection -------------------------------
# for C4
_C.RPN.TEST_PRE_NMS_TOPK = 6000
_C.RPN.TEST_POST_NMS_TOPK = 1000   # if you encounter OOM in inference, set this to a smaller number
# for FPN, #proposals per-level and #proposals after merging are (for now) the same
# if FPN.PROPOSAL_MODE = 'Joint', these options have no effect
_C.RPN.TEST_PER_LEVEL_NMS_TOPK = 2000

# fastrcnn training ---------------------
_C.FRCNN.BBOX_REG_WEIGHTS = [10., 10., 5., 5.]  # Better but non-standard setting: [20, 20, 10, 10]

# FPN -------------------------
_C.FPN.ANCHOR_STRIDES = (4, 8, 16, 32, 64)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
_C.FPN.PROPOSAL_MODE = 'Joint'  # 'Level', 'Joint'
_C.FPN.NUM_CHANNEL = 256
# The head option is only used in FPN. For C4 models, the head is C5
# choices: fastrcnn_2fc_head, fastrcnn_4conv1fc_{,gn_}head
_C.FPN.FRCNN_CONV_HEAD_DIM = 256
_C.FPN.FRCNN_FC_HEAD_DIM = 1024

# Mask-RCNN
_C.MRCNN.HEAD_DIM = 256

# Cascade-RCNN, only available in FPN mode
_C.FPN.CASCADE = True
_C.CASCADE.BBOX_REG_WEIGHTS = [[10., 10., 5., 5.], [20., 20., 10., 10.], [30., 30., 15., 15.]]

# testing -----------------------
_C.TEST.FRCNN_NMS_THRESH = 0.8

# Smaller threshold value gives significantly better mAP. But we use 0.05 for consistency with Detectron.
_C.TEST.RESULT_SCORE_THRESH = 0.3
_C.TEST.RESULTS_PER_IM = 300

_C.freeze()  # avoid typo / wrong config keys


def finalize_configs():
    """
    Run some sanity checks, and populate some configs from others
    """
    _C.freeze(False)  # populate new keys now
    _C.DATA.NUM_CLASS = _C.DATA.NUM_CATEGORY + 1  # +1 background

    _C.RPN.NUM_ANCHOR = len(_C.RPN.ANCHOR_SIZES) * len(_C.RPN.ANCHOR_RATIOS)
    assert len(_C.FPN.ANCHOR_STRIDES) == len(_C.RPN.ANCHOR_SIZES)
    # image size into the backbone has to be multiple of this number
    _C.FPN.RESOLUTION_REQUIREMENT = _C.FPN.ANCHOR_STRIDES[3]  # [3] because we build FPN with features r2,r3,r4,r5

    size_mult = _C.FPN.RESOLUTION_REQUIREMENT * 1.
    _C.PREPROC.MAX_SIZE = np.ceil(_C.PREPROC.MAX_SIZE / size_mult) * size_mult
    assert _C.FPN.PROPOSAL_MODE in ['Level', 'Joint']

    # autotune is too slow for inference
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

    _C.freeze()
    print("Config: ------------------------------------------\n" + str(_C))


if __name__ == "__main__":
    finalize_configs()