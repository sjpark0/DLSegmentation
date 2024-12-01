import numpy as np
import sys

from llff.poses.pose_utils import load_colmap_data

def computeOffset(w, h, refC2W, fltW2C, refFocal, fltFocal, zValue, ptX, ptY):
    centerX = w / 2
    centerY = h / 2
    
    origin = refC2W[:,3]
    print(origin)
    dir = refC2W[:,0:3].dot([(ptX - centerX) / refFocal, (ptY - centerY) / refFocal, 1])
    print(dir)
    t_origin = fltW2C.dot(origin)
    print(t_origin)
    t_dir = fltW2C.dot(dir)
    print(t_dir)
    tr = (zValue - t_origin[2]) / t_dir[2]
    print(tr)
    trans = t_origin + tr * t_dir
    print(trans)
    offsetX = trans[0] / trans[2] * fltFocal + centerX - ptX
    offsetY = trans[1] / trans[2] * fltFocal + centerY - ptY

    print(offsetX, offsetY)

width = 3840
height = 2160
zValue = 67.625877
refCamID = 0
ptX = 702
ptY = 916

poses, pts3d, perms, w2c, c2w = load_colmap_data('../../Data/Sample1')

refC2W = c2w[refCamID,:,:]
refFocal = poses[2, 4, refCamID]

numCam = c2w.shape[0]

for i in range(numCam):
    fltW2C = w2c[perms[i],:,:]
    fltFocal = poses[2, 4, perms[i]]
    print(i)
    computeOffset(width, height, refC2W, fltW2C, refFocal, fltFocal, zValue, ptX, ptY)
