  
import string
import argparse

import sys 
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np 
import cv2 
import math 

from .utils import CTCLabelConverter, AttnLabelConverter
from .dataset import FileAlignCollate, AlignCollate
from .model import Model


class Recognition():

    def __init__(self, recog_config, device_ids = None ):

        """
            device_ids : int 
        """

        self.opt = recog_config 
        c_dict=open(self.opt.character , encoding='utf-8')
        c_dict=c_dict.read().split()
        self.opt.character=''.join(c_dict)
        self.device_ids = device_ids 
        
        self.model= self.buildModel()

        
        self.AlignCollate_demo = FileAlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD, c=self.opt.input_channel)
        
    def _cutRects(self, np_image, np_bboxes):
        """
            np_image: 整张图片 numpy格式, 需要RGB顺序
            np_bboxes: 这张图片的文字区域的bbox, numpy array (n,4,2), 点可以乱序,需要(x,y)格式
        """
        def order_points_old(pts):
            #TODO: parallel code 

            rect = np.zeros_like(pts)
        
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
        
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            return rect

        
        rects = np.zeros_like(np_bboxes)
        for i in range(rects.shape[0]):
            rects[i] = order_points_old(np_bboxes[i])
        
        pil_results=[]
        for i, rect in enumerate(rects):

            w = int(np.linalg.norm(rect[0]-rect[1]))
            h = int(np.linalg.norm(rect[0]-rect[-1]))

            four_points=np.array(((0,0),(w-1,0),(w-1,h-1),(0,h-1)),dtype=np.float32)
            target_points = rect.astype(np.float32)

            try:
                M = cv2.getPerspectiveTransform(target_points, four_points)
                Rotated= cv2.warpPerspective(np_image, M, (w, h))
                pil_pic = Image.fromarray(Rotated)

                if self.opt.input_channel == 1 :
                    pil_pic = pil_pic.convert('L')
                else :
                    pil_pic = pil_pic.convert('RGB')
                pil_results.append(pil_pic)
 
            except:
                print('cut failed ')
                pass 
            
        return pil_results
        
    def bboxPicRecognition(self ,np_image, np_bboxes):
        """
            np_image: 整张图片 numpy格式, 需要RGB顺序
            np_bboxes: 这张图片的文字区域的bbox, numpy array (n,4,2), 点可以乱序,需要(x,y)格式
        """
        if len(np_bboxes) == 0 :
            # print('empty !')
            image_list = []
        else :
            image_list = self._cutRects(np_image, np_bboxes)
        
        # print('preds str len{}-{}'.format(len(np_bboxes),len(image_list)))
        result, probs = self.imageBatchRecognition(image_list)

        return result, probs

    def buildModel(self):
        opt=self.opt

        """ model configuration """

        if len(opt.character) < opt.minC:
            opt.character += ' '*(opt.minC-len(opt.character))
    
        if 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        
        self.converter = converter
        
        opt.num_class = len(converter.character)

        model = Model(opt)
        
        if isinstance(self.device_ids, int) :
            self.device_ids = [self.device_ids]

        model = torch.nn.DataParallel(model, device_ids = self.device_ids)
        if torch.cuda.is_available():
            model = model.cuda()

        # load model
        if opt.saved_model:
            print('start load model')
            print('loading  model from {}'.format(opt.saved_model))
            model.load_state_dict(torch.load(opt.saved_model))
            print('success load model ')
        return model 

    @torch.no_grad()
    def _recognition(self,images):
        model=self.model
        opt=self.opt 
        model.eval()
        str_probs = None 

        with torch.no_grad():
            images = images.cuda()
            length_for_pred = torch.cuda.IntTensor([opt.batch_max_length] * len(images))
            text_for_pred = torch.cuda.LongTensor(len(images), opt.batch_max_length + 1).fill_(0)

        if 'CTC' in opt.Prediction:
            preds=model(images,text_for_pred).softmax(-1)
            preds_size=torch.IntTensor([preds.size(1)]*len(images))
            preds_prob, preds_index = preds.permute(1,0,2).max(2)
            
            preds_index = preds_index.transpose(1,0).contiguous().view(-1)
            preds_prob = preds_prob.transpose(1,0).contiguous().view(-1)
            preds_str, str_probs= self.converter.decode(preds_index.data,preds_size.data, preds_prob)
        
        elif opt.Prediction == 'Transformer':
            preds_index = model(images, None, is_train= False)
            preds_str = self.converter.decode(preds_index)

        else :
            preds = model(images, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
        
        result=[]
        
        for pred in preds_str:
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
            result.append(pred)
        
        return result, str_probs

    def fileRecognition(self,file_name):


        opt=self.opt 
        model=self.model 
        model.eval()
        if opt.input_channel == 1 :
            pil_image = Image.open(file_name).convert('L')
        else :
            pil_image = Image.open(file_name).convert('RGB')

        image_tensors=self.AlignCollate_demo([pil_image])

        result, *_=self._recognition(image_tensors)

        return result[0]    

    def imageBatchRecognition(self, image_list, batch_size=64):
        """
            image_batch : list of PIL Image 
            return : list of labels 
        """ 
        
        index=0
        result=[]
        probs = []
        
        while index < len(image_list):
            # print(index,end='\r')
            
            images=self.AlignCollate_demo(image_list[index:index+batch_size])
            part_result, part_probs=self._recognition(images)
            result+=part_result
            probs+= part_probs
            index+=batch_size

        return result, probs
        
