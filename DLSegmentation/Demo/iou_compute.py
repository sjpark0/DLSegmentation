import numpy as np
import cv2

numCam = 16
cnt_and = 0
cnt_or = 0
prefix = "seg"

for m in range(13):
    for i in range(numCam):
        inputfilename1 = "../../Data/Test/Set{:0d}/".format(m+1) + prefix + "/{:02d}".format(i+1) + ".png"
        inputfilename2 = "../../Data/Test/Set{:0d}/masks1/{:02d}".format(m+1, i+1) + "_total.png"
                
        img1 = cv2.imread(inputfilename1)
        img2 = cv2.imread(inputfilename2)

        a = np.bitwise_and(img1, img2)
        b = np.bitwise_or(img1, img2)

        coords1 = np.where(a != 0)
        coords2 = np.where(b != 0)
        
        cnt_and = cnt_and + coords1[0].shape[0]
        cnt_or = cnt_or + coords2[0].shape[0]

print(prefix, cnt_and, cnt_or, cnt_and / cnt_or)

cnt_and = 0
cnt_or = 0
prefix = "seg_detectron"

for m in range(13):
    for i in range(numCam):
        inputfilename1 = "../../Data/Test/Set{:0d}/".format(m+1) + prefix + "/{:02d}".format(i+1) + ".png"
        inputfilename2 = "../../Data/Test/Set{:0d}/masks1/{:02d}".format(m+1, i+1) + "_total.png"
                
        img1 = cv2.imread(inputfilename1)
        img2 = cv2.imread(inputfilename2)

        a = np.bitwise_and(img1, img2)
        b = np.bitwise_or(img1, img2)

        coords1 = np.where(a != 0)
        coords2 = np.where(b != 0)
        
        cnt_and = cnt_and + coords1[0].shape[0]
        cnt_or = cnt_or + coords2[0].shape[0]

print(prefix, cnt_and, cnt_or, cnt_and / cnt_or)

cnt_and = 0
cnt_or = 0
prefix = "seg_maskformer"

for m in range(13):
    for i in range(numCam):
        inputfilename1 = "../../Data/Test/Set{:0d}/".format(m+1) + prefix + "/{:02d}".format(i+1) + ".png"
        inputfilename2 = "../../Data/Test/Set{:0d}/masks1/{:02d}".format(m+1, i+1) + "_total.png"
                
        img1 = cv2.imread(inputfilename1)
        img2 = cv2.imread(inputfilename2)

        a = np.bitwise_and(img1, img2)
        b = np.bitwise_or(img1, img2)

        coords1 = np.where(a != 0)
        coords2 = np.where(b != 0)
        
        cnt_and = cnt_and + coords1[0].shape[0]
        cnt_or = cnt_or + coords2[0].shape[0]

print(prefix, cnt_and, cnt_or, cnt_and / cnt_or)
