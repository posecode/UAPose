#!/usr/bin/python
# -*- coding:utf8 -*-

import os
from .my_custom import CfgNode

_C = CfgNode()
_C.DETECTOR_NAME = ''
_C.ROOT_DIR = ''
_C.EXPERIMENT_NAME = ''
_C.OUTPUT_DIR = ''
_C.SAVE_HEATMAPS = False
_C.LOAD_HEATMAPS = False
_C.SAVE_PREDS = False
_C.PREDS_SFX = ''
_C.LOAD_PREDS = False
_C.SAVE_OFFSETS = False
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.MODEL_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 8
_C.PRINT_FREQ = 20
_C.PIN_MEMORY = True
_C.RANK = 0
_C.VIS_HTMS = False
#
# _C.DISTANCE_OTHERWISE_MARGIN = True
_C.DISTANCE_WHOLE_OTHERWISE_SEGMENT = True
_C.DISTANCE = 2
_C.PREVIOUS_DISTANCE = 1
_C.NEXT_DISTANCE = 1
_C.CORE_FUNCTION = ""
# _C.SEED = 55593233
_C.SEED = 55069777

_C.EVAL_TRACKING = False
_C.TRACK_PREDS_FILE = ''
_C.TRACKING_THRESHOLD = 0.5

# Cudnn related params
_C.CUDNN = CfgNode()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CfgNode()
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.BACKBONE_NAME = 'pose_hrnet'

_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.FREEZE_WEIGHTS = False
_C.MODEL.VERSION = 'full'

_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.EFFECTIVE_NUM_JOINTS = 15
_C.MODEL.TARGET_TYPE = 'gaussian'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
_C.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
_C.MODEL.SIGMA = 2
_C.MODEL.EXTRA = CfgNode(new_allowed=True)
_C.MODEL.CYCLE_CONSISTENCY_FINETUNE = False
_C.MODEL.DEFORAM_CONV_VERSION = 1
_C.MODEL.DEFORMABLE_CONV = CfgNode(new_allowed=True)
_C.MODEL.USE_RECTIFIER = True
_C.MODEL.USE_MARGIN = True
_C.MODEL.USE_GROUP = True
_C.MODEL.HIGH_RESOLUTION = False
_C.MODEL.FREEZE_BACKBONE_WEIGHTS = False
_C.MODEL.FREEZE_HRNET_WEIGHTS = False

_C.MODEL.MPII_PRETRAINED = False
_C.MODEL.USE_WARPING_TRAIN = True
_C.MODEL.USE_WARPING_TEST = True
_C.MODEL.WARPING_REVERSE = False
_C.MODEL.USE_GT_INPUT_TEST = False
_C.MODEL.USE_GT_INPUT_TRAIN = False
_C.MODEL.ITER = 30000
_C.MODEL.EVALUATE = True
_C.MODEL.DILATION_EXP = 0
_C.MODEL.VISUALIZE_OFFSETS = False
_C.MODEL.USE_SUPP_TARG_NUM = 2
_C.MODEL.USE_SUPP_DIRECT = 'left'
_C.MODEL.DECODER = CfgNode(new_allowed=True)
_C.MODEL.TFF = CfgNode(new_allowed=True)
_C.MODEL.HEAD = CfgNode(new_allowed=True)
_C.MODEL.HTMS_MERG_STYLE= 'all-cat'
_C.MODEL.STREAMS = 1
_C.MODEL.ORDER_TYPE = None
_C.MODEL.FRAME_NUM = 3
_C.MODEL.EXTRA = CfgNode(new_allowed=True)

_C.MODEL.UNET = CfgNode(new_allowed=True)
_C.MODEL.VIT  = CfgNode(new_allowed=True)
_C.MODEL.CADD = CfgNode(new_allowed=True)

_C.DIFFUSION = CfgNode(new_allowed=True)

#### LOSS ####
_C.LOSS = CfgNode()
_C.LOSS.NAME = 'MSELOSS'
_C.LOSS.USE_OHKM = False
_C.LOSS.TOPK = 8
_C.LOSS.USE_TARGET_WEIGHT = True
_C.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False
_C.LOSS.USE_SOFTARGMAX = False
_C.LOSS.HEATMAP_MSE = CfgNode(new_allowed=True)

#### DATASET ####
_C.DATASET = CfgNode()
_C.DATASET.RANDOM_AUX_FRAME = True
_C.DATASET.ROOT = ''
_C.DATASET.NAME = ''
_C.DATASET.DATASET = 'mpii'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'valid'
_C.DATASET.HYBRID_JOINTS_TYPE = ''
_C.DATASET.SELECT_DATA = False
_C.DATASET.TEST_ON_TRAIN = False
_C.DATASET.JSON_FILE = ''
_C.DATASET.JSON_DIR = ''
_C.DATASET.POSETRACK17_JSON_DIR = ''
_C.DATASET.POSETRACK18_JSON_DIR = ''
_C.DATASET.IMG_DIR = ''
_C.DATASET.POSETRACK17_IMG_DIR = ''
_C.DATASET.POSETRACK18_IMG_DIR = ''
_C.DATASET.IS_POSETRACK18 = False
_C.DATASET.COLOR_RGB = False
_C.DATASET.TEST_IMG_DIR = ''
_C.DATASET.POSETRACK17_TEST_IMG_DIR = ''
_C.DATASET.POSETRACK18_TEST_IMG_DIR = ''
_C.DATASET.INPUT_TYPE = ''
_C.DATASET.BBOX_ENLARGE_FACTOR = 1.0
_C.DATASET.TIME_INTERVAL = 7
_C.DATASET.SPLIT_VERSION =  1
_C.DATASET.TRAIN = CfgNode(new_allowed=True)
_C.DATASET.VAL = CfgNode(new_allowed=True)
_C.DATASET.TEST = CfgNode(new_allowed=True)


_C.DATA_PRESET = CfgNode(new_allowed=True)



#### TRAIN ####
_C.TRAIN = CfgNode()  # cfg.Node
_C.TRAIN.SAVE_MODEL_PER_EPOCH = 2
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True
_C.TRAIN.LOSS_ALPHA = 1.0
_C.TRAIN.LOSS_BETA = 1.0
_C.TRAIN.LOSS_GAMA = 1.0
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.MILESTONES = [8, 12, 16]
_C.TRAIN.GAMMA = 0.99
_C.TRAIN.LR = 0.001
_C.TRAIN.STSN_LR = 0.001
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.AUTO_RESUME = False
_C.TRAIN.FLIP = True
_C.TRAIN.SCALE_FACTOR = [0.25, 0.25]
# _C.TRAIN.SCALE_FACTOR = 0.25
_C.TRAIN.ROT_FACTOR = 30
_C.TRAIN.PROB_HALF_BODY = 0.0
_C.TRAIN.NUM_JOINTS_HALF_BODY = 8
_C.TRAIN.LR_SCHEDULER = 'MultiStepLR'
_C.TRAIN.MI_ALPHA = 0.1
_C.TRAIN.MI_BETA  = 0.1
_C.TRAIN.MI_GAMMA  = 0.1


#### VAL ####
_C.VAL = CfgNode()
_C.VAL.BATCH_SIZE_PER_GPU = 1
_C.VAL.MODEL_FILE = ''
_C.VAL.ANNOT_DIR = ''
_C.VAL.COCO_BBOX_FILE = ''
_C.VAL.USE_GT_BBOX = False
_C.VAL.FLIP_VAL = False
_C.VAL.BBOX_THRE = 1.0
# _C.VAL.BBOX_THRE = 1.0
_C.VAL.IMAGE_THRE = 0.1
_C.VAL.IN_VIS_THRE = 0.0
_C.VAL.NMS_THRE = 0.6
_C.VAL.OKS_THRE = 0.5
_C.VAL.SHIFT_HEATMAP = False
_C.VAL.SOFT_NMS = False
_C.VAL.POST_PROCESS = False
_C.VAL.FLIP = False

#### TEST ####
_C.TEST = CfgNode()
_C.TEST.BATCH_SIZE_PER_GPU = 1
_C.TEST.MODEL_FILE = ''
_C.TEST.ANNOT_DIR = ''
_C.TEST.COCO_BBOX_FILE = ''
_C.TEST.USE_GT_BBOX = False
_C.TEST.FLIP_TEST = False
_C.TEST.BBOX_THRE = 1.0
_C.TEST.IMAGE_THRE = 0.1
_C.TEST.IN_VIS_THRE = 0.0
_C.TEST.NMS_THRE = 0.6
_C.TEST.OKS_THRE = 0.5
_C.TEST.SHIFT_HEATMAP = False
_C.TEST.SOFT_NMS = False
_C.TEST.POST_PROCESS = False
### INFERENCE ###
_C.INFERENCE = CfgNode()
_C.INFERENCE.MODEL_FILE = ''

# DEBUG
_C.DEBUG = CfgNode()
_C.DEBUG.VIS_SKELETON = False
_C.DEBUG.VIS_BBOX = False
_C.DEBUG.VIS_TENSORBOARD = False

_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False
