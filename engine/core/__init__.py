#!/usr/bin/python
# -*- coding:utf8 -*-

from utils.utils_registry import Registry

CORE_FUNCTION_REGISTRY = Registry("CORE_FUNCTION")

from engine.core.base import BaseFunction, AverageMeter, build_core_function

## function
from .function import CommonFunction
from .function_single_frame import CommonFunctionSingleFrame
from .function_sparsely import Function_Sparsely
from .function_no_supp_targ import Function_No_Supp_Targ
from .function_no_supp_targ_5_frame import Function_No_Supp_Targ_5_Frame
from .function_sparsely_mi import Function_Sparsely_MI
from .function_no_supp_targ_mi import Function_No_Supp_Targ_MI
from .function_sparsely_mi_V2 import Function_Sparsely_MI_V2
from .function_no_supp_targ_mi_5_frame import Function_No_Supp_Targ_MI_5_Frame
from .functions.AuxposeV5 import auxposev5
from .function_no_supp_targ_jhmdb import Function_No_Supp_Targ_JHMDB

from .DiffusionPose.function_diffusionpose_3_frame import Function_DiffusionPose_3_frame
from .DiffusionPose.function_diffusionpose_Subset_5_frame import Function_DiffusionPose_Subset_5_frame
from .DiffusionPose.function_diffusionpose_5_frame import Function_DiffusionPose_5_frame
from .DiffusionPose.function_single_frame_jrdb import CommonFunctionSingleFrameJRDB

from .UGPose.function_no_supp_targ import Function_No_Supp_Targ_UGPose
from .UGPose.function_no_supp_targ_jhmdb_ugpose import Function_No_Supp_Targ_JHMDB_UGPOSE
from .UGPose.function_UAPose_Subset_5_frame import Function_UAPose_Subset_5_frame
