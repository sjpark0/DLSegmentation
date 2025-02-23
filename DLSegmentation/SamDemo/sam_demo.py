import torch
from misc import *
from pose import *
from SCSam import SCSam
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available:
    device = "mps"
else:
    device = "cpu"

sc = SCSam(device)
sc.LoadImage("../../Data/Sample1", 16)
sc.AddPoint(0, [840, 1356], 1)
sc.RunSegmentation()

plt.imshow(sc.images[0])
show_mask(sc.masks[0], plt.gca())
show_points(np.array(sc.input_points[0]), np.array(sc.input_labels[0]), plt.gca())
plt.axis('on')
plt.show()