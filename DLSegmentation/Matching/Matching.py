import cv2
import numpy as np

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
    dir = refC2W[:,0:3].dot([(ptX - centerX) / refFocal, (ptY - centerY) / refFocal, 1])
    t_origin = fltW2C.dot(origin)
    t_dir = fltW2C.dot(dir)
    tr = (zValue - t_origin[2]) / t_dir[2]    
    trans = t_origin + tr * t_dir
    
    offsetX = trans[0] / trans[2] * fltFocal + centerX - ptX
    offsetY = trans[1] / trans[2] * fltFocal + centerY - ptY
    return offsetX, offsetY

def computeOverlapCount(mask1, mask2, boundingBox1, boundingBox2, w, h, offsetX, offsetY):
    boundingBoxNew2 = [boundingBox1[0] + int(offsetX) - boundingBox2[0], boundingBox1[1] + int(offsetY) - boundingBox2[1], boundingBox1[2] + int(offsetX) - boundingBox2[0], boundingBox1[3] + int(offsetY) - boundingBox2[1]]    
    boundingBoxNew3 = [max(0, boundingBoxNew2[0]), max(0, boundingBoxNew2[1]), min(mask2.shape[1], boundingBoxNew2[2]), min(mask2.shape[0], boundingBoxNew2[3])]

    boundingBoxNew1 = [boundingBoxNew3[0] + boundingBox2[0] - int(offsetX) - boundingBox1[0], boundingBoxNew3[1] + boundingBox2[1] - int(offsetY) - boundingBox1[1], boundingBoxNew3[2] + boundingBox2[0] - int(offsetX) - boundingBox1[0], boundingBoxNew3[3] + boundingBox2[1] - int(offsetY) - boundingBox1[1]]
    #boundingBoxNew1_1 = [boundingBoxNew2[0] + boundingBox2[0] - int(offsetX), boundingBoxNew2[1] + boundingBox2[1] - int(offsetY), boundingBoxNew2[2] + boundingBox2[0] - int(offsetX), boundingBoxNew2[3] + boundingBox2[1] - int(offsetY)]
    
    #print(boundingBox1, boundingBox2, boundingBoxNew1_1)
    #print(mask1.shape, mask2.shape, boundingBoxNew1, boundingBoxNew2, boundingBoxNew3)
    if boundingBoxNew3[3] > boundingBoxNew3[1] and boundingBoxNew3[2] > boundingBoxNew3[0]:
        tmp = np.bitwise_and(mask1[boundingBoxNew1[1]:boundingBoxNew1[3], boundingBoxNew1[0]:boundingBoxNew1[2]], mask2[boundingBoxNew3[1]:boundingBoxNew3[3], boundingBoxNew3[0]:boundingBoxNew3[2]])
        coords = np.where(tmp == 255)
        res = coords[0].shape[0]
    else:
        res = 0
    #print(res)
    
    return res


def ComputeDepth(masks, boundingBoxes, w, h, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms):
    zstep = ((1.0 / close_depth) - (1.0 / inf_depth))
    optZ = -1.0
    optMean = 100000000.0
    objCount = []
    for i in range(len(masks)):
        count = []
        for j in range(len(masks[i])):
            coords = np.where(masks[i][j] == 255)
            count.append(coords[0].shape[0])
        objCount.append(count)

    similarities = []
    depths = []
    for z in range(101):
        s0 = []
        d0 = []
        for i in range(len(masks)):
            s1 = []
            d1 = []
            for j in range(len(masks[refCamID])):
                s2 = []
                d2 = []
                for k in range(len(masks[i])):
                    s2.append(0.0)
                    d2.append(0.0)
                s1.append(s2)
                d1.append(d2)
            s0.append(s1)
            d0.append(d1)
        similarities.append(s0)
        depths.append(d0)

    
    #for z in np.linspace(0, 10.0, 101):
    for z in range(101):
        zValueCurrent = 1.0 / (zstep * (float(z) / 10.0) + 1.0 / inf_depth)
        mean = 0.0
        numAvailableCam = 0
        for i in range(len(masks)):
            if perms[i] == perms[refCamID]:
                continue
            offsetX, offsetY = computeOffsetByZValue(w, h, c2w[perms[refCamID],:,:], w2c[perms[i],:,:], focals[perms[refCamID]], focals[perms[i]], zValueCurrent, 0, 0)
            for j in range(len(masks[refCamID])):
                for k in range(len(masks[i])):
                    overlapCount = computeOverlapCount(masks[refCamID][j], masks[i][k], boundingBoxes[refCamID][j], boundingBoxes[i][k], w, h, offsetX, offsetY)
                    similarities[z][i][j][k] = overlapCount / (objCount[refCamID][j] + objCount[i][k] - overlapCount)
                    #sim = overlapCount / (objCount[refCamID][j] + objCount[i][k] - overlapCount)
                    #if sim > similarities[i][j][k]:
                    #    similarities[i][j][k] = sim
                    #    depths[i][j][k] = zValueCurrent 
    return similarities

def computeOffset(masks, boundingBoxes, w, h, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms):
    offsetX = []
    offsetY = []
    
    opt_sim = 0
    opt_id = -1
    similarities, depths = ComputeDepth(masks, boundingBoxes, w, h, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms)
    for j in range(len(masks[refCamID])):
        for i in range(len(masks)):
            if i != refCamID:
                opt_id = -1
                opt_sim = 0
                for k in range(len(masks[i])):
                    if similarities[i][j][k] > opt_sim:
                        opt_sim = similarities[i][j][k]
                        opt_id = k
                print(i, j, opt_id, depths[i][j][opt_id], similarities[i][j][opt_id], opt_sim)
                cv2.imshow("Image1", masks[refCamID][j])
                cv2.imshow("Image2", masks[i][opt_id])
                cv2.waitKey(0)
            #print(i, j, k, depths[i][j][k], similarities[i][j][k])
    #for i in range(len(masks)):
    #    off_x, off_y = computeOffsetByZValue(w, h, c2w[perms[refCamID],:,:], w2c[perms[i],:,:], focals[perms[refCamID]], focals[perms[i]], optZ, (boundingBoxes[refCamID][0] + boundingBoxes[refCamID][2]) / 2, (boundingBoxes[refCamID][1] + boundingBoxes[refCamID][3]) / 2)
    #    offsetX.append(off_x)
    #    offsetY.append(off_y)
    
    #return optZ, offsetX, offsetY

def maskLoader(folder, prefix, numCam):
    ppimgs = []
    ppboxes = []

    for i in range(numCam):
        j = 0
        pimgs = []
        pboxes = []
        while True:
            img = cv2.imread("{}/{}_{:03d}_{:02d}.png".format(folder, prefix, i, j), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                break
            coords = np.where(img == 255)
            x_min = int(np.min(coords[1]))
            x_max = int(np.max(coords[1]))
            y_min = int(np.min(coords[0]))
            y_max = int(np.max(coords[0]))
            box = x_min, y_min, x_max, y_max
            pboxes.append(box)
            pimgs.append(img[y_min:y_max, x_min:x_max])
            j += 1
        ppimgs.append(pimgs)
        ppboxes.append(pboxes)
    return ppboxes, ppimgs

#numObjs = [57, 53, 48, 46, 54, 49, 43, 39, 

refCamID = 0
poses, pts3d, perms, w2c, c2w = load_colmap_data('../../Data/Sample1')

cdepth, idepth = computecloseInfinity(poses, pts3d, perms)
close_depth = np.min(cdepth) * 0.9
inf_depth = np.max(idepth) * 2.0

img = cv2.imread('../../Data/Sample1/images/000.png')

width = img.shape[1]
height = img.shape[0]
focals = poses[2, 4, :]

boxes, masks = maskLoader("../../Data/Sample1/masks", "detectron2", 2)
computeOffset(masks, boxes, width, height, c2w, w2c, focals, refCamID, close_depth, inf_depth, perms)