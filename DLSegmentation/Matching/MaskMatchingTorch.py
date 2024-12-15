import cv2
import numpy as np
import torch
from torch import Tensor

from llff.poses.pose_utils import load_colmap_data
class MaskMatchingTorch:
    def __init__(self, foldername):
        self.numFocalPoint = 100
        poses, pts3d, self.perms, self.w2c, self.c2w = load_colmap_data(foldername)
        cdepth, idepth = self.ComputeCloseInfinity(poses, pts3d)
        self.close_depth = np.min(cdepth) * 0.9
        self.inf_depth = np.max(idepth) * 2.0

        img = cv2.imread(foldername + '/images/000.png')
        
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.focals = poses[2, 4, :]

    def ComputeCloseInfinity(self, poses, pts3d):
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
        for i in self.perms:
            vis = vis_arr[:, i]
            zs = zvals[:, i]
            zs = zs[vis==1]
            close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
            cdepth.append(close_depth)
            idepth.append(inf_depth)
        
        return cdepth, idepth   
    
    def MaskLoader(self, folder, prefix, numCam):
        self.masks = []
        self.boundingBoxes = []

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
                #pimgs.append(img[y_min:y_max, x_min:x_max])
                pimgs.append(torch.as_tensor(img[y_min:y_max, x_min:x_max]))
                j += 1
            self.masks.append(pimgs)
            self.boundingBoxes.append(pboxes)

    def ComputeOffsetByZValue(self, refC2W, fltW2C, refFocal, fltFocal, zValue, ptX, ptY):
        centerX = self.width / 2
        centerY = self.height / 2
        
        origin = refC2W[:,3]
        dir = refC2W[:,0:3].dot([(ptX - centerX) / refFocal, (ptY - centerY) / refFocal, 1])
        t_origin = fltW2C.dot(origin)
        t_dir = fltW2C.dot(dir)
        tr = (zValue - t_origin[2]) / t_dir[2]    
        trans = t_origin + tr * t_dir
        
        offsetX = trans[0] / trans[2] * fltFocal + centerX - ptX
        offsetY = trans[1] / trans[2] * fltFocal + centerY - ptY
        return offsetX, offsetY

    def ComputeOverlapCount(self, mask1, mask2, boundingBox1, boundingBox2, offsetX, offsetY):
        boundingBoxNew2 = [boundingBox1[0] + int(offsetX) - boundingBox2[0], boundingBox1[1] + int(offsetY) - boundingBox2[1], boundingBox1[2] + int(offsetX) - boundingBox2[0], boundingBox1[3] + int(offsetY) - boundingBox2[1]]    
        boundingBoxNew3 = [max(0, boundingBoxNew2[0]), max(0, boundingBoxNew2[1]), min(mask2.shape[1], boundingBoxNew2[2]), min(mask2.shape[0], boundingBoxNew2[3])]

        boundingBoxNew1 = [boundingBoxNew3[0] + boundingBox2[0] - int(offsetX) - boundingBox1[0], boundingBoxNew3[1] + boundingBox2[1] - int(offsetY) - boundingBox1[1], boundingBoxNew3[2] + boundingBox2[0] - int(offsetX) - boundingBox1[0], boundingBoxNew3[3] + boundingBox2[1] - int(offsetY) - boundingBox1[1]]
        
        if boundingBoxNew3[3] > boundingBoxNew3[1] and boundingBoxNew3[2] > boundingBoxNew3[0]:
            #tmp = np.bitwise_and(mask1[boundingBoxNew1[1]:boundingBoxNew1[3], boundingBoxNew1[0]:boundingBoxNew1[2]], mask2[boundingBoxNew3[1]:boundingBoxNew3[3], boundingBoxNew3[0]:boundingBoxNew3[2]])
            #coords = np.where(tmp == 255)
            tmp = torch.bitwise_and(mask1[boundingBoxNew1[1]:boundingBoxNew1[3], boundingBoxNew1[0]:boundingBoxNew1[2]], mask2[boundingBoxNew3[1]:boundingBoxNew3[3], boundingBoxNew3[0]:boundingBoxNew3[2]])
            coords = torch.where(tmp == 255)
            res = coords[0].shape[0]
        else:
            res = 0
        
        return res
    
    def ComputeDepth(self, refCamID):
        zstep = ((1.0 / self.close_depth) - (1.0 / self.inf_depth))
        optZ = -1.0
        optMean = 100000000.0
        objCount = []
        for i in range(len(self.masks)):
            count = []
            for j in range(len(self.masks[i])):
                coords = torch.where(self.masks[i][j] == 255)
                count.append(coords[0].shape[0])
            objCount.append(count)

        similarities = []
        ids = []
        for z in range(self.numFocalPoint):    
            s0 = []
            id0 = []
            for j in range(len(self.masks[refCamID])):
                s1 = []
                id1 = []
                for i in range(len(self.masks)):
                    s1.append(0.0)
                    id1.append(-1)
                s0.append(s1)
                id0.append(id1)
            similarities.append(s0)
            ids.append(id0)
        
        #for z in np.linspace(0, 10.0, 101):
        for z in range(self.numFocalPoint):
            zValueCurrent = 1.0 / (zstep * (float(z) / 10.0) + 1.0 / self.inf_depth)
            mean = 0.0
            numAvailableCam = 0
            for j in range(len(self.masks[refCamID])):
                for i in range(len(self.masks)):
                    if self.perms[i] == self.perms[refCamID]:
                        continue
                    offsetX, offsetY = self.ComputeOffsetByZValue(self.c2w[self.perms[refCamID],:,:], self.w2c[self.perms[i],:,:], self.focals[self.perms[refCamID]], self.focals[self.perms[i]], zValueCurrent, 0, 0)
                    for k in range(len(self.masks[i])):
                        overlapCount = self.ComputeOverlapCount(self.masks[refCamID][j], self.masks[i][k], self.boundingBoxes[refCamID][j], self.boundingBoxes[i][k], offsetX, offsetY)
                        #similarities[z][j][i][k] = overlapCount / (objCount[refCamID][j] + objCount[i][k] - overlapCount)
                        sim = overlapCount / (objCount[refCamID][j] + objCount[i][k] - overlapCount)
                        if sim > similarities[z][j][i]:
                            similarities[z][j][i] = sim
                            ids[z][j][i] = k
                        #    depths[i][j][k] = zValueCurrent 
        correspondingID = []
        depths = []
        for j in range(len(self.masks[refCamID])):
            c0 = []
            d0 = []
            for i in range(len(self.masks)):
                c0.append(j)
                d0.append(0.0)
            correspondingID.append(c0)
            depths.append(d0)

        for j in range(len(self.masks[refCamID])):
            for i in range(len(self.masks)):
                if self.perms[i] == self.perms[refCamID]:
                    continue
                sim = 0.0
                for z in range(self.numFocalPoint):
                    if similarities[z][j][i] > sim:
                        sim = similarities[z][j][i]
                        correspondingID[j][i] = ids[z][j][i]
                        depths[j][i] = zValueCurrent = 1.0 / (zstep * (float(z) / 10.0) + 1.0 / self.inf_depth)

        return correspondingID, depths
    
    def ComputeOffset(self, refCamID):
        correspondingID, depths = self.ComputeDepth(refCamID)
        for j in range(len(self.masks[refCamID])):
            for i in range(len(self.masks)):
                if i != refCamID:
                    #cv2.imshow("Image1", masks[refCamID][j])
                    #cv2.imshow("Image2", masks[i][correspondingID[j][i]])
                    #cv2.waitKey(0)
                    print(j, i, correspondingID[j][i])