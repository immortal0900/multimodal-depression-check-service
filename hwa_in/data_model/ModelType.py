# hwa_in/data_model/ModelType.py

from enum import Enum

class ModelType(str, Enum):
    RESNET18 = "RESNET18"
    CONVNEXT_SMALL = "CONVNEXT_SMALL"
    CONVNEXT_TINY = "CONVNEXT_TINY"
    CONVNEXT_BASE = "CONVNEXT_BASE"
    CONVNEXT_LARGE = "CONVNEXT_LARGE"
    VIT_B_16 = "VIT_B_16"
    EFFICIENT_NET_B0 =  "EFFICIENT_NET"
