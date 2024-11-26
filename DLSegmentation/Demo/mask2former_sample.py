import sys
sys.path.append('../../Detectron2')
sys.path.append('../../Mask2Former')

import numpy as np
import torch, torchvision
import cv2

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from detectron2.structures import BitMasks

from mask2former import add_maskformer2_config

config = "./configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
checkpoint = "../../models/model_final_e5f453.pkl"

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(config)
cfg.MODEL.WEIGHTS = checkpoint

predictor = DefaultPredictor(cfg)

img = cv2.imread("../../Data/000.png")

prediction = predictor(img)
instances = prediction["instances"]
instances = instances.to('cpu')

masks = np.asarray(instances.pred_masks)
masks = 255 * masks
masks = masks.astype(np.uint8)
res_mask = np.bitwise_or.reduce(masks, 0)
print(res_mask.shape)
cv2.imwrite("mask2former.png", res_mask)
