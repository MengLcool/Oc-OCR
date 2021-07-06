import cv2 
from collections import OrderedDict

import sys 
import json 
import codecs

# import tools.image_tools as image_tools
from .config import g_craft_cfg, g_BERT_CTC_cpx_config
from .Recognition.recog_inference import Recognition
from .Craft.detect_inference import WordDetector
from .Craft.imgproc import loadImage

class TextCraft():

    def __init__(self, craft_config= g_craft_cfg, recog_config= g_BERT_CTC_cpx_config, device_ids= [0]):

        self.detector = WordDetector(craft_config, device_ids= device_ids)
        self.recognizer = Recognition(recog_config, device_ids= device_ids)


    def _processImage(self, image):
        """
            image: nparray(h,w,3) [RGB]
        """
        bboxes = self.detector.detectImage(image)
        texts, probs = self.recognizer.bboxPicRecognition(image, bboxes) 

        return bboxes, texts, probs


    def processImageFile(self, image_path):
        """
        input : 
            file_path 
        return : 
            bboxes: nparray(n,4,2)
            texts: str list, len=n
        """

        image = loadImage(image_path)
        
        return self._processImage(image)


    def processVideoFile(self, video_path, result_file, frame_start, frame_total):

        cap = cv2.VideoCapture(video_path)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)    

        if (frame_start < frame_total):
            i = 0
            while i < frame_start:
                 cap.read()
                 i += 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        
        frame_ori_total = frame_num - frame_start 

        if frame_ori_total < frame_total :
            frame_total = frame_ori_total


        if frame_total < 250:
            sample_interval = 5
        else:
            sample_interval = 15

        pre_texts = []
        video_text_info = []

        print(frame_start , frame_total, sample_interval , type(sample_interval))

        for frame_idx in range(int(frame_start) , int(frame_start + frame_total) , sample_interval):
            if frame_idx != frame_start:
                for tmp in range(sample_interval - 1):
                    ret, frame = cap.read()

            ret, frame = cap.read()
            if not ret :
                continue 

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bboxes, texts, probs = self._processImage(image)

            if not len(texts) > 0 :
                continue 
  
            if set(texts) == set(pre_texts):
                video_text_info[-1]['end'] = frame_idx 
                continue 
            
            bboxes_info = []
            frame_info = OrderedDict()
            keys = ['x0','y0','x1','y1','x2','y2','x3','y3']
            for bbox, text, prob in zip(bboxes, texts, probs):
                if text == '':
                    continue
                bbox_info = OrderedDict()
                bbox = bbox.reshape(-1)
                for i in range(8):
                    bbox_info[keys[i]] = int(bbox[i])
                bbox_info['text'] = text
                bbox_info['probability'] = prob
                bboxes_info.append(bbox_info)
                
            frame_info['start'] = frame_idx
            frame_info['end'] = frame_idx
            frame_info['bbxs'] = bboxes_info
            video_text_info.append(frame_info)
            pre_texts = texts
        
            print('text {}, probs {}'.format(texts,probs))

        result_dict = OrderedDict()
        result_dict['totaltime'] = frame_total / max(frame_rate , 1 )
        result_dict['framerate'] = frame_rate   
        result_dict['trajectories'] = video_text_info
        result_dict['width'] = video_width
        result_dict['height'] = video_height

        with codecs.open(result_file, 'w', 'utf-8') as f:
            json.dump(result_dict, f, ensure_ascii = False,  indent = 4)
        cap.release()
    
    """
    def processImage_from_imgpath(self, image_path):
        return self.processImage_from_cvimg(image_tools.imread(image_path, 'BGR'))
    """

    def processImage_from_cvimg(self, cv_img):
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        bboxes, texts, probs= self._processImage(img)
        result_dict = OrderedDict()

        img_h, img_w = img.shape[:2]

        result_dict['width'] = img_w
        result_dict['height'] = img_h

        bboxes_info = []
        for bbox, text, prob in zip(bboxes, texts, probs):
            if text == '':
                continue 
            bbox_info = OrderedDict()
            bbox = bbox.reshape(-1)
            polygon = []
            for i in range(8):
                polygon.append(int(bbox[i]))
            bbox_info['polygon'] = polygon
            bbox_info['text'] = text
            bbox_info['probability'] = prob
            bboxes_info.append(bbox_info)

        result_dict['analysis_results'] = OrderedDict()
        result_dict['analysis_results']['ocr'] = bboxes_info

        return result_dict

if __name__ == '__main__':

    # import file_utils
    # cool_text = TextCraft(g_craft_cfg, g_BERT_CTC_cpx_config)
    # image_path = 'test_5.png'
    # bboxes, texts = cool_text.processImageFile(image_path)
    
    # image = loadImage(image_path)
    # file_utils.saveResult('test_a.jpg', image[:,:,::-1], bboxes, dirname='new_test/', texts= texts)

    # print(texts)
    # sys.exit(-1)

    import os 

    if len(sys.argv) > 5 :
        video_name = sys.argv[1]
        result_file = sys.argv[2]
        start_frame = int(sys.argv[3])
        video_length = int(sys.argv[4])
        gpu_id = int(sys.argv[5])

    else :
        print("five parameters needed for video: videoName, resultFile, startFrame, videoLength, gpuID")
        sys.exit(-1)

    print(sys.argv[1:6])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    text_craft = TextCraft(g_craft_cfg, g_BERT_CTC_cpx_config)

    text_craft.processVideoFile(video_name, result_file, start_frame, video_length)
