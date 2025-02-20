import sys
sys.path.append('../../detectron2')
import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn
import cv2
import numpy as np
import os

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    STABLE_ONNX_OPSET_VERSION,
    TracingAdapter,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

def detectron2_fn():
    config = "./configs/COCO-InstanceSegmentation\\mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    checkpoint = "../../models/model_final_2d9806.pkl"

    cfg = get_cfg()
    #add_deeplab_config(cfg)
    #add_maskdino_config(cfg)
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = checkpoint

    # create a torch model
    #model = build_model(cfg)
    #DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    #model.eval()
    predictor = DefaultPredictor(cfg)
    numCam = 16

    numCam = 16
    for m in range(13):
        os.makedirs("../../Data/Test/Set{:0d}/seg_detectron".format(m+1), exist_ok=True)
        
        for i in range(numCam):
            inputfilename = "../../Data/Test/Set{:0d}/images/{:02d}".format(m+1, i+1) + ".png"
            outputfilename = "../../Data/Test/Set{:0d}/seg_detectron/{:02d}".format(m+1, i+1) + ".png"
                
            img = cv2.imread(inputfilename)
            height, width = img.shape[:2]

            image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            image.to('cuda')
            inputs = {"image": image, "height": height, "width": width}
            
            #prediction = model([inputs])[0]
            prediction = predictor(img)
            instances = prediction["instances"] 
            instances = instances.to('cpu')
            
            masks = np.asarray(instances.pred_masks)
            masks = 255 * masks
            masks = masks.astype(np.uint8)
            res_mask = np.zeros((height, width), dtype=np.uint8)
            for j in range(masks.shape[0]):
                if instances.pred_classes[j] == 0:
                    res_mask = np.bitwise_or(res_mask, masks[j,...])
                    #res_mask = res_mask | masks[j,...]
            res_mask = cv2.resize(res_mask, dsize=(960, 540))        
            cv2.imwrite(outputfilename, res_mask)

            inputfilename = "../../Data/Test/Set{:0d}/images/{:02d}".format(m+1, i+21) + ".png"
            outputfilename = "../../Data/Test/Set{:0d}/seg_detectron/{:02d}".format(m+1, i+21) + ".png"
                
            img = cv2.imread(inputfilename)
            height, width = img.shape[:2]

            image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
            image.to('cuda')
            inputs = {"image": image, "height": height, "width": width}
            
            #prediction = model([inputs])[0]
            prediction = predictor(img)            
            instances = prediction["instances"] 
            instances = instances.to('cpu')
            
            masks = np.asarray(instances.pred_masks)
            masks = 255 * masks
            masks = masks.astype(np.uint8)
            res_mask = np.zeros((height, width), dtype=np.uint8)
            for j in range(masks.shape[0]):
                if instances.pred_classes[j] == 0:
                    res_mask = np.bitwise_or(res_mask, masks[j,...])
                    #res_mask = res_mask | masks[j,...]
            res_mask = cv2.resize(res_mask, dsize=(960, 540))
            cv2.imwrite(outputfilename, res_mask)
detectron2_fn()