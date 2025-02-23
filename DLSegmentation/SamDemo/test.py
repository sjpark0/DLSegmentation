import torch
from misc import *
from pose import *
from SCSam2 import SCSam2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

   
sc = SCSam2("cpu")
sc.LoadImage("../../Data/Sample1", 16)
sc.AddPoint(0, [840, 1356], 1)
t1 = time.time()
sc.RunSegmentation()
t2 = time.time()

print(t2 - t1)

plt.imshow(sc.images[0])
show_mask(sc.masks[0], plt.gca())
show_points(np.array(sc.input_points[0]), np.array(sc.input_labels[0]), plt.gca())
plt.axis('on')
plt.show()

   
sc = SCSam2("mps")
sc.LoadImage("../../Data/Sample1", 16)
sc.AddPoint(0, [840, 1356], 1)
t1 = time.time()
sc.RunSegmentation()
t2 = time.time()

print(t2 - t1)

plt.imshow(sc.images[0])
show_mask(sc.masks[0], plt.gca())
show_points(np.array(sc.input_points[0]), np.array(sc.input_labels[0]), plt.gca())
plt.axis('on')
plt.show()


sc = SCSam2("cpu")
sc.LoadImage("../../Data/Sample1", 16)
sc.AddPoint(0, [840, 1356], 1)
t1 = time.time()
sc.RunSegmentation()
t2 = time.time()

print(t2 - t1)

plt.imshow(sc.images[0])
show_mask(sc.masks[0], plt.gca())
show_points(np.array(sc.input_points[0]), np.array(sc.input_labels[0]), plt.gca())
plt.axis('on')
plt.show()