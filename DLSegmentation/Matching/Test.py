import numpy as np
import sys
import cv2

from llff.poses.pose_utils import load_colmap_data

def computecloseInfinity(poses, pts3d, perm):
    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            cams[ind-1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    
    cdepth = []
    idepth = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        cdepth.append(close_depth)
        idepth.append(inf_depth)
    
    return cdepth, idepth

def computeOffsetByZValue(w, h, refC2W, fltW2C, refFocal, fltFocal, zValue, ptX, ptY):
    centerX = w / 2
    centerY = h / 2
    
    origin = refC2W[:,3]
    #print(origin)
    #print(ptX, ptY, centerX, centerY, refFocal, (ptX - centerX) / refFocal, (ptY - centerY) / refFocal)
    dir = refC2W[:,0:3].dot([(ptX - centerX) / refFocal, (ptY - centerY) / refFocal, 1])
    #print(dir)
    t_origin = fltW2C.dot(origin)
    #print(t_origin)
    t_dir = fltW2C.dot(dir)
    #print(t_dir)
    tr = (zValue - t_origin[2]) / t_dir[2]    
    trans = t_origin + tr * t_dir
    #print(tr, trans[0], trans[1], trans[2])

    offsetX = trans[0] / trans[2] * fltFocal + centerX - ptX
    offsetY = trans[1] / trans[2] * fltFocal + centerY - ptY
    #return int(offsetX + 0.5), int(offsetY + 0.5)
    return offsetX, offsetY

def computeSimilarity(img1, img2, boundingBox1, offsetX, offsetY):
    w = img1.shape[1]
    h = img1.shape[0]
    boundingBox2 = [boundingBox1[0] + int(offsetX), boundingBox1[1] + int(offsetY), boundingBox1[2] + int(offsetX), boundingBox1[3] + int(offsetY)]
    #print(boundingBox2)
    boundingBox2[0] = max(0, min(w - 1, boundingBox2[0]))
    boundingBox2[1] = max(0, min(h - 1, boundingBox2[1]))
    boundingBox2[2] = max(0, min(w - 1, boundingBox2[2]))
    boundingBox2[3] = max(0, min(h - 1, boundingBox2[3]))
    #print(boundingBox2)
    num = ((boundingBox2[3] - boundingBox2[1]) * (boundingBox2[2] - boundingBox2[0]))
    if num > 0:
        mean = np.sum((img1[boundingBox2[1] - int(offsetY):boundingBox2[3] - int(offsetY), boundingBox2[0] - int(offsetX):boundingBox2[2] - int(offsetX),:] - img2[boundingBox2[1]:boundingBox2[3],boundingBox2[0]:boundingBox2[2],:])**2) / num
    else:
        mean = -1
    return mean

def ComputeDepth(img, boundingBox, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms):
    zstep = ((1.0 / close_depth) - (1.0 / inf_depth))
    optZ = -1.0
    optMean = 100000000.0
    
    for z in np.linspace(0, 10.0, 101):
        zValueCurrent = 1.0 / (zstep * z + 1.0 / inf_depth)
        mean = 0.0
        numAvailableCam = 0
        for i in range(len(img)):
            if perms[i] == perms[refCamID]:
                continue
            offsetX, offsetY = computeOffsetByZValue(img[perms[i]].shape[1], img[perms[i]].shape[0], c2w[perms[refCamID],:,:], w2c[perms[i],:,:], focals[perms[refCamID]], focals[perms[i]], zValueCurrent, (boundingBox[0] + boundingBox[2]) / 2, (boundingBox[1] + boundingBox[3]) / 2)
            #print(zValueCurrent, offsetX, offsetY)
            similarity = computeSimilarity(img[refCamID], img[i], boundingBox, offsetX, offsetY)
            print(zValueCurrent, similarity)
            if similarity >= 0:
                mean += similarity
                numAvailableCam+= 1
        if numAvailableCam > 0:
            if optMean > mean / numAvailableCam:
                optMean = mean / numAvailableCam
                optZ = zValueCurrent
    return optZ

def computeOffset(img, boundingBox, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms):
    offsetX = []
    offsetY = []
    optZ = ComputeDepth(img, boundingBox, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms)
    for i in range(len(img)):
        off_x, off_y = computeOffsetByZValue(img[i].shape[1], img[i].shape[0], c2w[perms[refCamID],:,:], w2c[perms[i],:,:], focals[perms[refCamID]], focals[perms[i]], optZ, (boundingBox[0] + boundingBox[2]) / 2, (boundingBox[1] + boundingBox[3]) / 2)
        offsetX.append(off_x)
        offsetY.append(off_y)
    
    return optZ, offsetX, offsetY

def Test1():
    refCamID = 0
    boundingBox = [516, 724, 904, 1112]
    poses, pts3d, perms, w2c, c2w = load_colmap_data('../../Data/Sample1')

    cdepth, idepth = computecloseInfinity(poses, pts3d, perms)
    close_depth = np.min(cdepth) * 0.9
    inf_depth = np.max(idepth) * 2.0

    img = cv2.imread('../../Data/Sample1/images/000.png')

    width = img.shape[1]
    height = img.shape[0]


    numCam = c2w.shape[0]
    focals = poses[2, 4, :]
    imgs = []
    for i in range(numCam):
        imgs.append(cv2.imread('../../Data/Sample1/images/{:03d}.png'.format(i)))

    optZ, offsetX, offsetY = computeOffset(imgs, boundingBox, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms)
    print(optZ)
    for i in range(numCam):
        print(offsetX[i], offsetY[i])

def Test2():    
    width = 3840
    height = 2160
    zValue = 67.625877
    #ptX = 702
    #ptY = 916
    boundingBox = [516, 724, 904, 1112]
    numCam = 16
    refCamID = 0
    poses, pts3d, perms, w2c, c2w = load_colmap_data('../../Data/Sample1')

    cdepth, idepth = computecloseInfinity(poses, pts3d, perms)
    close_depth = np.min(cdepth) * 0.9
    inf_depth = np.max(idepth) * 2.0

    refC2W = c2w[perms[refCamID],:,:]
    refFocal = poses[2, 4, perms[refCamID]]
    offsetX = []
    offsetY = []
    for i in range(numCam):
        fltW2C = w2c[perms[i],:,:]
        fltFocal = poses[2, 4, perms[i]]
        off_x, off_y = computeOffsetByZValue(width, height, refC2W, fltW2C, refFocal, fltFocal, zValue, (boundingBox[0] + boundingBox[2]) / 2, (boundingBox[1] + boundingBox[3]) / 2)
        offsetX.append(off_x)
        offsetY.append(off_y)
    for i in range(numCam):
        print(offsetX[i], offsetY[i])

Test1()