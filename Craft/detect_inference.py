
# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
import numpy as np
import json
import zipfile
from collections import OrderedDict

from . import file_utils
from . import craft_utils
from . import imgproc
from .craft import CRAFT


class WordDetector():

    def __init__(self, detect_cfg , device_ids = None ):
        """
        detect_cfg : CraftConfig()
        """

        self.cfg = detect_cfg 

        print('start build ')
        model = CRAFT()
        if isinstance(device_ids, int):
            device_ids = [device_ids]
        model = torch.nn.DataParallel(model.cuda(), device_ids= device_ids)
        print('build ok ')
        model.load_state_dict(torch.load(self.cfg.saved_model))

        self.model = model 
        

    @torch.no_grad()
    def _detect(self,image):
        """
        image : nparray(h,w,3) [RGB]
        return : boxes nparray(n,4,2) 
        """

        self.model.eval()
        cfg = self.cfg 

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, cfg.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=cfg.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        x = x.cuda()
        
        # forward pass
        y, _ = self.model(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, cfg.text_threshold, cfg.link_threshold, cfg.low_text, cfg.poly)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

        return boxes
        
    def detectFile(self, file_name):
        """
        file_name : image file path 
        return : nparray (n,4,2)
        """

        image = imgproc.loadImage(file_name)
        return self._detect(image)

    def detectImage(self, image):

        return self._detect(image)

if __name__ == '__main__':

    from craft_cfg import CraftConfig

    cfg = CraftConfig(saved_model= '/home/menglc/CRAFT/pretrain/pretrain.pth',
                    text_threshold=0.3, 
                    low_text=0.3, 
                    link_threshold=0.3, 
                    canvas_size=1080, 
                    poly= False, 
                    mag_ratio= 1.5)

    detector = WordDetector(cfg)

    print('model load success !')
    result = detector.detectFile('/home/menglc/CRAFT/new_test/test_4.png')

    print(result.shape, result)
