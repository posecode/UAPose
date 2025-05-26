#!/usr/bin/python
# -*- coding:utf8 -*-

# model
from .build import build_model, get_model_hyperparameter

# HRNet
from .backbones.hrnet import HRNet

from .backbones.vit import ViT


# SimpleBaseline
from .backbones.simplebaseline import SimpleBaseline

from .sldpose import SLDPOSE
from .sldpose_mi import SLDPOSE_MI
from .sldpose_mi_V2 import SLDPOSE_MI_V2
from .sldpose_mi_V3 import SLDPOSE_MI_V3
from .sldpose_mi_V2_5_frame_pe import SLDPOSE_MI_V2_5_FRAME_PE
from .sldpose_mi_V3_pe import SLDPOSE_MI_V3_PE

from .multi_scale_represt_encoder.newpose import NEWPOSE_V1
from .multi_scale_represt_encoder.newpose_v2 import NEWPOSE_V2
from .multi_scale_represt_encoder.newpose_v3 import NEWPOSE_V3

from .layers.block_zoo import *
from .layers.rope import *
from .layers.spatiotemporal_attention import *
from .layers.corss_block import *


from .auxpose.AuxposeV6 import AuxposeV6

from .mambapose.mamba_pose import MambaPose

from .Mixpose.mixpose import *

from .DiffusionPose.Diffusion import *

from .UGPose.Uncertainty_Guided_Pose import UGPOSE