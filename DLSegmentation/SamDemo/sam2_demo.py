import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from misc import *
from pose import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
    
class SCSam2:
    def __init__(self, device):
        sam_checkpoint = "../../models/sam2.1_hiera_large.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam_checkpoint, device=device))
        
        
        self.input_points = []
        self.input_labels = []
        self.images = []
    
    def LoadImage(self, folder, numImage):
        self.folder = folder
        self.numImage = numImage
        
        poses, pts3d, self.perms, self.w2c, self.c2w = load_colmap_data(self.folder)
        cdepth, idepth = computecloseInfinity(poses, pts3d, self.perms)
        self.close_depth = np.min(cdepth) * 0.9
        self.inf_depth = np.max(idepth) * 2.0
        self.focals = poses[2, 4, :]


        for i in range(numImage):
            filename = self.folder + "/images/{:03d}.png".format(i)
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images.append(image)
            self.input_points.append([])
            self.input_labels.append([])

    def AddPoint(self, refCamID, point, label):
        print(point.shape)
        print(label.shape)
        self.predictor.set_image(self.images[refCamID])
        masks, scores, logits = self.predictor.predict(point_coords=point, point_labels=label, multimask_output=False,)
        
        plt.imshow(self.images[refCamID])
        show_points(point, label, plt.gca())
        plt.axis('on')
        plt.show()

        plt.imshow(self.images[refCamID])
        show_mask(masks, plt.gca())
        show_points(point, label, plt.gca())
        #plt.title(f"Score: {scores:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

        print(masks.shape)
        h, w = masks.shape[-2:]
        mask_image = masks.reshape(h, w, 1)

        cv2.imwrite("temp.png", mask_image * 255)

        coords = np.where(mask_image > 0.0)
        boundingBox = int(np.min(coords[1])), int(np.max(coords[1])), int(np.min(coords[0])), int(np.max(coords[0]))
        
        optZ, offsetX, offsetY = computeOffset(self.images, boundingBox, self.c2w, self.w2c, self.focals, refCamID, self.close_depth, self.inf_depth, self.perms)
        for i in range(self.numImage):
            self.input_points[i].append([point[0] + offsetX[i], point[1] + offsetY[i]])
            self.input_labels[i].append([label])
            plt.imshow(self.images[i])
            show_points(self.input_points[i], self.input_labels[i], plt.gca())
            plt.axis('off')
            plt.show()
        #    print(offsetX[i], offsetY[i])
        

def original(image, input_point, input_label):
    sam_checkpoint = "../../models/sam2.1_hiera_large.pt"
    model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
    
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam_checkpoint, device="cpu"))
    predictor.set_image(image)
    start = time.time()
    masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True,)
    end = time.time()
    print(end - start, "sec")

    print(masks.shape)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

#input_point = np.array([[533, 286]])
#input_label = np.array([1])
#image = cv2.imread("../../Data/000.png")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#original(image, input_point, input_label)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)

sam = SCSam2(device)
sam.LoadImage("../../Data/Sample1", 16)
sam.AddPoint(0, np.array([[840, 1356]]), np.array([1]))
