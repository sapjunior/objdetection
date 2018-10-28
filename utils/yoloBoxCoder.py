import torch
import math
import numpy as np
from .box import nms,bboxIOU
class YOLOBoxCoder():
    def __init__(self, anchors, numClass, inputFeatMapSizes, inputImageSize=(412,412)):
        self.anchors = anchors
        self.scaledAnchors = []
        self.numClass = numClass
        self.inputFeatMapSizes = inputFeatMapSizes

        self.numAnchors = [len(anchor) for anchor in self.anchors]
        

        self.inputImageHeight = inputImageSize[0]
        self.inputImageWidth = inputImageSize[1]
        
        self.strideHeight = []
        self.strideWidth = []
        self.gridX = []
        self.gridY = []
        self.anchorW = []
        self.anchorH = []

        self._getDefaultAnchor()

    def _getDefaultAnchor(self):
        #### Anchor Generation ####
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        for featMapIdx, inputFeatMapSize in enumerate(self.inputFeatMapSizes):
            inputFeatMapHeight, inputFeatMapWidth = inputFeatMapSize

            strideHeight = self.inputImageHeight / inputFeatMapHeight
            self.strideHeight.append(strideHeight)
            strideWidth = self.inputImageWidth / inputFeatMapWidth
            self.strideWidth.append(strideWidth)

            scaledAnchors = [(anchorWidth / strideWidth, anchorHeight / strideHeight)
                             for anchorWidth, anchorHeight in self.anchors[featMapIdx]]
            self.scaledAnchors.append(scaledAnchors)
            
            # Calculate offsets for each grid
            gridX = torch.linspace(0, inputFeatMapWidth-1, inputFeatMapWidth).repeat(
                inputFeatMapWidth, 1).repeat(self.numAnchors[featMapIdx], 1, 1)
            
            
            gridY = torch.linspace(0, inputFeatMapHeight-1, inputFeatMapHeight).repeat(
                inputFeatMapHeight, 1).t().repeat(self.numAnchors[featMapIdx], 1, 1)

            # Calculate anchor w, h
            anchorW = torch.tensor(scaledAnchors).index_select(1, LongTensor([0]))
            anchorH = torch.tensor(scaledAnchors).index_select(1, LongTensor([1]))
            anchorW = anchorW.repeat(1, 1, inputFeatMapHeight * inputFeatMapWidth).view(gridX.shape)
            anchorH = anchorH.repeat(1, 1, inputFeatMapHeight * inputFeatMapWidth).view(gridY.shape)
            
            '''
            if torch.cuda.is_available():
                gridX = gridX.cuda()
                gridY = gridY.cuda()
                anchorH = anchorH.cuda()
                anchorW = anchorW.cuda()
            '''

            self.gridX.append(gridX)
            self.gridY.append(gridY)
            self.anchorH.append(anchorH)
            self.anchorW.append(anchorW)
    def encode(self, boxes):
        masks = []
        noobjMasks = []
        txs = []
        tys = []
        tws = []
        ths = []
        tconfs = []
        tclss = []

        for featMapIdx in range(len(self.inputFeatMapSizes)):
            featMapHeight, featMapWidth = self.inputFeatMapSizes[featMapIdx]
            mask = torch.zeros(self.numAnchors[featMapIdx], featMapHeight, featMapWidth, requires_grad=False)
            noobjMask = torch.ones(mask.size(), requires_grad=False)
            tx = torch.zeros(mask.size(), requires_grad=False)
            ty = torch.zeros(mask.size(), requires_grad=False)
            tw = torch.zeros(mask.size(), requires_grad=False)
            th = torch.zeros(mask.size(), requires_grad=False)
            tconf = torch.zeros(mask.size(), requires_grad=False)
            tcls = torch.zeros(self.numAnchors[featMapIdx], featMapHeight, featMapWidth, self.numClass, requires_grad=False)

            for box in boxes:
                gx = box[0] * self.inputFeatMapSizes[featMapIdx][0]
                gy = box[1] * self.inputFeatMapSizes[featMapIdx][1]
                gw = box[2] * self.inputFeatMapSizes[featMapIdx][0]
                gh = box[3] * self.inputFeatMapSizes[featMapIdx][1]
                label = int(box[4])
                # Get anchor indices
                gi = int(gx)
                gj = int(gy)

                # Get shape of gt box
                gtBox = torch.tensor(np.array([0, 0, gw, gh]), dtype=torch.float32).unsqueeze(0)
                anchorShapes = torch.tensor(np.concatenate((np.zeros((self.numAnchors[featMapIdx], 2)),
                                                                  np.array(self.scaledAnchors[featMapIdx])), 1),dtype=torch.float32)
                
                ious = bboxIOU(gtBox, anchorShapes)

                # Set no object mask to false if ious is greater than threshold 
                noobjMask[ious > 0.5, gj, gi] = 0
                # Find the best matching anchor box
                bestBoxIdx = np.argmax(ious)

                # Masks (Match)
                mask[bestBoxIdx, gj, gi] = 1
                # diff from anchor center
                tx[bestBoxIdx, gj, gi] = gx - gi
                ty[bestBoxIdx, gj, gi] = gy - gj
                # width height
                tw[bestBoxIdx, gj, gi] = math.log(gw/self.scaledAnchors[featMapIdx][bestBoxIdx][0] + 1e-16)
                th[bestBoxIdx, gj, gi] = math.log(gh/self.scaledAnchors[featMapIdx][bestBoxIdx][1] + 1e-16)
                # object
                tconf[bestBoxIdx, gj, gi] = 1
                # One-hot encoding of label
                tcls[bestBoxIdx, gj, gi, label] = 1
            masks.append(mask)
            noobjMasks.append(noobjMask)
            txs.append(tx)
            tys.append(ty)
            tws.append(tw)
            ths.append(th)
            tconfs.append(tconf)
            tclss.append(tcls)
        return masks, noobjMasks, txs, tys, tws, ths, tconfs, tclss
    
    def decode(self, inputFeatMaps):
        outputs = []
        for featMapIdx, inputFeatMap in enumerate(inputFeatMaps):
            batchSize, _, inputFeatMapHeight, inputFeatMapWidth = inputFeatMap.shape
            prediction = inputFeatMap.view(
                batchSize, self.numAnchors[featMapIdx], self.numClass+5, inputFeatMapHeight, inputFeatMapWidth).permute(0, 1, 3, 4, 2).contiguous()
            
            # Get outputs
            x = torch.sigmoid(prediction[..., 0])          # Center x
            y = torch.sigmoid(prediction[..., 1])          # Center y
            w = prediction[..., 2]                         # Width
            h = prediction[..., 3]                         # Height
            conf = torch.sigmoid(prediction[..., 4])       # Conf
            predCls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

            predBoxes = torch.empty(prediction[..., :4].shape)
            predBoxes[..., 0] = x.data + self.gridX[featMapIdx]
            predBoxes[..., 1] = y.data +self.gridY[featMapIdx]
            predBoxes[..., 2] = torch.exp(w.data) * self.anchorW[featMapIdx]
            predBoxes[..., 3] = torch.exp(h.data) * self.anchorH[featMapIdx]

            # Results
            _scale = torch.tensor([self.strideWidth[featMapIdx], self.strideHeight[featMapIdx]] * 2)
            
            output = torch.cat((predBoxes.view(batchSize, -1, 4) * _scale, conf.view(
                batchSize, -1, 1), predCls.view(batchSize, -1, self.numClass)), -1)

            outputs.append(output)
        return torch.cat(outputs,1)

def YOLOBoxPostProcess(xyCenterBoxes, numClass, objectThreshold=0.65, nmsThreshold=0.45):
    # xyCenterBoxes ==> [ [[x-center, y-center, width, height, objectnessConf, classScores....]] ]
    # Convert from (x-center, y-center, width, height) to (x1, y1, x2, y2) coordinate
    xyBoxes = torch.empty(xyCenterBoxes.size())
    xyBoxes[:, :, 0] = xyCenterBoxes[:, :, 0] - xyCenterBoxes[:, :, 2] / 2
    xyBoxes[:, :, 1] = xyCenterBoxes[:, :, 1] - xyCenterBoxes[:, :, 3] / 2
    xyBoxes[:, :, 2] = xyCenterBoxes[:, :, 0] + xyCenterBoxes[:, :, 2] / 2
    xyBoxes[:, :, 3] = xyCenterBoxes[:, :, 1] + xyCenterBoxes[:, :, 3] / 2
    xyBoxes[:, :, 4:] = xyCenterBoxes[:, :, 4:]
    boxesOutput = []
    for imageNo in range(xyBoxes.shape[0]):
        imageBoxes = xyBoxes[imageNo,:,:]
        # Filter out boxes with low objectness score
        thresholdMask = (imageBoxes[:, 4] >= objectThreshold).squeeze()
        imageBoxes = imageBoxes[thresholdMask]

        # No remaining boxes
        if not imageBoxes.size(0):
            boxesOutput.append(None)
            continue

        # For each box, grab class with highest confidence
        classConf, classPred = torch.max(imageBoxes[:, 5:5 + numClass], 1,  keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, classConf, classPred)
        detectionResults = torch.cat((imageBoxes[:, :5], classConf.float(), classPred.float()), 1)

        # Back to CPU
        detectionResultsNp = detectionResults.cpu().numpy()
        
        # Grab detected class labels
        detectedClasses = np.unique(detectionResultsNp[:, -1])

        imageOutputBoxes = np.empty((0,6))
        # Iterate over each detected class 
        for detectedClass in detectedClasses:
            boxIdx = np.where(detectionResultsNp[:,-1] == detectedClass)[0]
            classBoxes = detectionResultsNp[boxIdx,:7]
            nmsClassBoxes = (classBoxes[nms(classBoxes, nmsThreshold),:])[:,[0,1,2,3,4,6]] # Remove class conf column
            imageOutputBoxes = np.vstack((imageOutputBoxes, nmsClassBoxes))
        boxesOutput.append(imageOutputBoxes)
    return boxesOutput