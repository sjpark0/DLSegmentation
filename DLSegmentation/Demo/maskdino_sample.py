import sys
sys.path.append('../../Detectron2')
#sys.path.append('..\Mask2Former')
sys.path.append('../../MaskDINO')

import numpy as np
import torch, torchvision
import cv2
import os

from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from detectron2.structures import BitMasks
from detectron2.checkpoint import DetectionCheckpointer

from maskdino import add_maskdino_config

def maskdino_fn():
    foldername = "../../Data/maskdino"
    os.makedirs(foldername, exist_ok=True)

    config = "./configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"
    checkpoint = "../../models/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"
    img = cv2.imread("../../Data/000.png")

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = checkpoint

    #이거 잘 안됨. maskdino_sample1.py부분 참고하여 수정 필요함.
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    with torch.no_grad():
        height, width = img.shape[:2]        
        image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        image.to('cuda')
        inputs = {"image": image, "height": height, "width": width}

        prediction = model([inputs])[0]

        instances = prediction["instances"]
        instances = instances.to('cpu')

        masks = np.asarray(instances.pred_masks)
        masks = 255 * masks
        masks = masks.astype(np.uint8)

        res_mask = np.bitwise_or.reduce(masks, 0)    
        cv2.imwrite(foldername + "/total.png", res_mask)
        for j in range(masks.shape[0]):
            cv2.imwrite(foldername + "/{:02d}.png".format(j), masks[j,::])

def maskdino_fn1():
    foldername = "../../Data/maskdino1"
    os.makedirs(foldername, exist_ok=True)
    
    config = "./configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"
    checkpoint = "../../models/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"
    img = cv2.imread("../../Data/000.png")

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = checkpoint

    predictor = DefaultPredictor(cfg)

    with torch.no_grad():
        height, width = img.shape[:2]        
        image = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        image.to('cuda')
        inputs = {"image": image, "height": height, "width": width}

        prediction = predictor(img)
        instances = prediction["instances"]
        instances = instances.to('cpu')

        masks = np.asarray(instances.pred_masks)
        masks = 255 * masks
        masks = masks.astype(np.uint8)

        res_mask = np.bitwise_or.reduce(masks, 0)    
        cv2.imwrite(foldername + "/total.png", res_mask)
        for j in range(masks.shape[0]):
            cv2.imwrite(foldername + "/{:02d}.png".format(j), masks[j,::])
            
    #return res_mask
maskdino_fn()
maskdino_fn1()
