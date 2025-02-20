import sys
sys.path.append('../../Detectron2')
sys.path.append('../../Mask2Former')

import numpy as np
import torch, torchvision
import cv2

from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from detectron2.structures import BitMasks
from detectron2.checkpoint import DetectionCheckpointer

from mask2former import add_maskformer2_config

config = "./configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
checkpoint = "../../models/model_final_e5f453.pkl"
img = cv2.imread("../../Data/000.png")
def mask2former():
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
    cv2.imwrite("../../Data/seg_mask2former/total.png", res_mask)

    for j in range(masks.shape[0]):
        cv2.imwrite("../../Data/seg_mask2former/{:02d}.png".format(j), masks[j,::])

def mask2former1():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
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
        cv2.imwrite("../../Data/seg_mask2former1/total.png", res_mask)
        for j in range(masks.shape[0]):
            cv2.imwrite("../../Data/seg_mask2former1/{:02d}.png".format(j), masks[j,::])

mask2former1()