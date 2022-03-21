# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Zekun Li  
# version ='1.0'
# ---------------------------------------------------------------------------
""" Spatial AI Assignment"""  
# ---------------------------------------------------------------------------

import torchvision
from torchvision import models

fcn_model = torchvision.models.segmentation.fcn_resnet50(pretrained = False, num_classes = 2)