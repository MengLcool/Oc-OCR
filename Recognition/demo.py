import string
import argparse

import sys 
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils import CTCLabelConverter, AttnLabelConverter, BertConverter, TransformerConverter
from dataset import FileAlignCollate, AlignCollate
from model import Model
from new_model import NewModel
from PIL import Image
from BertAttnModel import BertAttnModel, Identity
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import cv2 
import math 

from model_cfg import RecognitionConfig, g_BERT_Attn_config, g_BERT_Attn_cpx_config, g_BERT_CTC_config, g_BERT_CTC_cpx_config

BERT = False

class Recognition():

    def __init__(self, recog_config = None, gpu_id = None ):

        """
            gpu_id : int 
        """
        if recog_config :
            self.opt = recog_config 
        else :
            self.opt = RecognitionConfig(
                saved_model = 'saved_models/BERT_CTC/best_norm_ED.pth',
                batch_max_length = 35,
                imgH = 32,
                imgW = 256,
                character = 'new_dict.txt',
                minC = 5000,
                PAD = True,
                Transformation = 'TPS',
                FeatureExtraction = 'SEResNet',
                SequenceModeling = 'BERT',
                Prediction = 'CTC',
                num_fiducial = 20,
                input_channel = 1,
                output_channel = 512,
                hidden_size = 256
            )


        # tmp 
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--saved_model', type=str,default='saved_models/BERT_Attn_cpx/best_norm_ED.pth' ,help="path to saved_model to evaluation")
        # """ Data processing """
        # parser.add_argument('--batch_max_length', type=int, default=35, help='maximum-label-length')
        # parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
        # parser.add_argument('--imgW', type=int, default=256, help='the width of the input image')
        # # parser.add_argument('--rgb', action='store_true', help='use rgb input')
        # parser.add_argument('--rgb', default=False, help='use rgb input')
        # parser.add_argument('--character', type=str, default='new_dict.txt', help='character label')
        # parser.add_argument('--minC', type=int, default=10000, help='min classes of dict ')
        # # parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
        # parser.add_argument('--PAD', type=bool, default= True, help='whether to keep ratio then pad for image resize')
        # """ Model Architecture """
        # parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
        # parser.add_argument('--FeatureExtraction', type=str, default='SEResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
        # parser.add_argument('--SequenceModeling', type=str, default='BERT', help='SequenceModeling stage. None|BiLSTM')
        # parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
        # parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
        # parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
        # parser.add_argument('--output_channel', type=int, default=512,
        #                     help='the number of output channel of Feature extractor')
        # parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

        c_dict=open(self.opt.character , encoding='utf-8')
        c_dict=c_dict.read().split()
        self.opt.character=''.join(c_dict)
        self.gpu_id = gpu_id 
        
        

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
            print('empty !')
            image_list = []
        else :
            image_list = self._cutRects(np_image, np_bboxes)
        
        print('preds str len{}-{}'.format(len(np_bboxes),len(image_list)))
        result = self.imageBatchRecognition(image_list)

        return result 

    def buildModel(self):
        opt=self.opt

        """ model configuration """

        if len(opt.character) < opt.minC:
            opt.character += ' '*(opt.minC-len(opt.character))
    
        if BERT:
            print('bert converter !')
            converter = BertConverter(opt.character)
        elif 'CTC' in opt.Prediction:
            converter = CTCLabelConverter(opt.character)
        elif opt.Prediction == 'Transformer':
            converter = TransformerConverter(opt.character)
        else:
            converter = AttnLabelConverter(opt.character)
        
        self.converter = converter
        
        opt.num_class = len(converter.character)

        if BERT:
            print(opt.num_class)
            model = BertAttnModel(num_class=opt.num_class)
        else :
            model = Model(opt)
        # model=NewModel(opt)

        if self.gpu_id :
            self.gpu_id = [self.gpu_id]

        model = torch.nn.DataParallel(model, device_ids = self.gpu_id)
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
        with torch.no_grad():
            images = images.cuda()
            length_for_pred = torch.cuda.IntTensor([opt.batch_max_length] * len(images))
            text_for_pred = torch.cuda.LongTensor(len(images), opt.batch_max_length + 1).fill_(0)

        if BERT:
            scores = model(images)
            _, preds = scores.max(-1)
            # preds_str = self.converter.decode(preds,decode_type=0)
            preds_str = self.converter.decode(preds)
            
        elif 'CTC' in opt.Prediction:
            preds=model(images,text_for_pred)
            preds_size=torch.IntTensor([preds.size(1)]*len(images))
            _, preds_index = preds.permute(1,0,2).max(2)
            
            preds_index = preds_index.transpose(1,0).contiguous().view(-1)
            preds_str= self.converter.decode(preds_index.data,preds_size.data)
        
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
        
        return result 

    def fileRecognition(self,file_name):


        opt=self.opt 
        model=self.model 
        model.eval()
        if opt.input_channel == 1 :
            pil_image = Image.open(file_name).convert('L')
        else :
            pil_image = Image.open(file_name).convert('RGB')

        image_tensors=self.AlignCollate_demo([pil_image])

        # predict
        # model.eval()
        # with torch.no_grad():
        #     image = image_tensors.cuda()
        #     length_for_pred = torch.cuda.IntTensor([opt.batch_max_length] * 1)
        #     text_for_pred = torch.cuda.LongTensor(1, opt.batch_max_length + 1).fill_(0)

        # if 'CTC' in opt.Prediction:
        #     # preds=model(image,text_for_pred).log_softmax(2)

        #     preds=model(image,text_for_pred)
        #     preds_size=torch.IntTensor([preds.size(1)])
        #     _, preds_index = preds.permute(1,0,2).max(2)
        #     preds_index = preds_index.transpose(1,0).contiguous().view(-1)
        #     preds_str= self.converter.decode(preds_index.data,preds_size.data)

        # else :
        #     preds = model(image, text_for_pred, is_train=False)
        #     _, preds_index = preds.max(2)
        #     preds_str = self.converter.decode(preds_index, length_for_pred)
        
        result=self._recognition(image_tensors)

        return result[0]    

    def imageBatchRecognition(self, image_list, batch_size=16):
        """
            image_batch : list of PIL Image 
            return : list of labels 
        """ 
        
        index=0
        result=[]
        
        while index < len(image_list):
            print(index,end='\r')
            
            images=self.AlignCollate_demo(image_list[index:index+batch_size])
            part_result=self._recognition(images)
            result+=part_result
            index+=batch_size

        return result
        
g_Recognizer = Recognition(g_BERT_CTC_cpx_config)        


if __name__ =='__main__':
    import time 
    import os 
    import sys 
    #model=Recognition(g_BERT_CTC_cpx_config)
    model = g_Recognizer
    start_time=time.time()
    # g_root_dir='cut_test'
    # g_root_dir = '/ai/local/menglc/v3_dataset/raw_img/validation/ReCTS' 
    # g_root_dir = '/ai/local/menglc/ocrdata/test_cut'
    # g_root_dir = '/home/menglc/ocrdata/cut_img'
    # g_root_dir = '/ai/local/menglc/v3_dataset/raw_img/validation/ctw'
    g_root_dir = '/home/menglc/ocrdata/TextRecognitionDataGenerator-master/TextRecognitionDataGenerator/poster_simple'
    count=1e-8
    correct_count=0
    mode=4

    if mode==0:
        # from collections import OrderedDict
        # model_dict = model.model.module.SequenceModeling.state_dict()

        # from torchvision.models import resnet34 

        # net = resnet34()
        # net.load_state_dict(torch.load('pretrain/resnet34-333f7ec4.pth'))
        # print(net)
        
        # net.avgpool= Identity()
        # net.fc = Identity()
        # new_dict= OrderedDict()
        # for k,v in model_dict.items():
        #     new_dict[k[7:]]=v 

        from transformer_decoder import TransFormerDecoder
        from modules.prediction import Attention

        new_decoder = Attention(256, 256, 10002)
        # new_decoder = TransFormerDecoder(50,5004)
        
        pre_decoder = model.model.module.Prediction 
        

        # new_decoder = torch.nn.Linear(256, 10001)

        # print('pre decoder weight:{}, bias:{} .'.format(pre_decoder.weight.shape, pre_decoder.bias.shape))

        # torch.nn.init.constant_(new_decoder.bias, 0.0)
        # torch.nn.init.kaiming_normal_(new_decoder.weight)

        # new_decoder.weight[:5001,:] = pre_decoder.weight 
        # new_decoder.bias[:5001] = pre_decoder.bias 
        # new_decoder.weight[5001:,:] = 0.0 
        # new_decoder.bias[5001:] = 0.0

        model.model.module.Prediction =  new_decoder

        torch.save(model.model.state_dict(),'pretrain/pretrain_bigger.pth')


        import sys
        sys.exit()


        # torch.save(net.state_dict(),'pretrain/my_resnet34.pth')
        # torch.save(model.model.module.Transformation.state_dict(),'pretrain/HV_CTC_tps.pth')
        # torch.save(model.model.module.FeatureExtraction.state_dict(),'pretrain/HV_CTC_seresnet.pth')

    elif mode ==1:
        log_file=open('Cut_result.txt','w',encoding='utf-8')
        log_file.write('file_name\tresult\tpred\tgt\n')
        dir_list=os.listdir(g_root_dir)
        dir_list.sort()
        pil_images=[]
        gts=[]
        for i,name in enumerate(dir_list):
            if name.endswith('jpg'):
                print(i,end='\r')
                pil_image = Image.open(os.path.join(g_root_dir,name)).convert('L')
                # pil_image = Image.open(os.path.join(g_root_dir,name)).convert('RGB')
                pil_images.append(pil_image)
                # result=model.fileRecognition(os.path.join(g_root_dir,name))
                try:
                    gt = open(os.path.join(g_root_dir,name[:-4]+'.txt')).read()
                    gt=gt.split()[0]
                except:
                    gt ='###'
                gts.append(gt)
        
        results = model.imageBatchRecognition(pil_images)
        print('len results: {}, len gts: {}, '.format(len(results),len(gts)))
        for result, gt in zip(results,gts):

            log_file.write('{}\t{}\t{}\t{}\n'.format(name,(result==gt),result,gt))
            if result==gt :
                correct_count+=1
            # log_file.write('{}\t{}\n'.format(name,result))
            count+=1

    elif mode==2:
        log_file=open('Cut_result.txt','w',encoding='utf-8')
        log_file.write('file_name\tresult\tpred\tgt\n')
        dir_list=os.listdir(g_root_dir)
        dir_list.sort()
        for i,name in enumerate(dir_list):
            if name.endswith('jpg'):
                print(count,end='\r')
                result=model.fileRecognition(os.path.join(g_root_dir,name))
                try:
                    gt = open(os.path.join(g_root_dir,name[:-4]+'.txt')).read()
                    gt=gt.split()[0]
                except:
                    gt ='###'
                
                log_file.write('{}\t{}\t{}\t{}\n'.format(name,(result==gt),result,gt))
                if result==gt :
                    correct_count+=1
                # log_file.write('{}\t{}\n'.format(name,result))
                count+=1

    elif mode==3:
        g_root_dir='normal_test/results_feiweijin'
        dir_list=os.listdir(g_root_dir)
        dir_list.sort()
        for i,name in enumerate(dir_list):
            # if name.endswith('jpg'):
            # print(i,end='\r')
            result=model.fileRecognition(os.path.join(g_root_dir,name))
            # gt = open(os.path.join(g_root_dir,name[:-4]+'.txt')).read()
            # log_file.write('{}\t{}\t\n'.format((result==gt),result))
            # log_file.write('{}\t{}\n'.format(name,result))
            print('{}\t{}'.format(name,result))
            count+=1

    elif mode ==4:
        count=1
        result=model.fileRecognition('test_2.png')
        print('result ',result)

    elif mode==5:
        log_file=open('Cut_result.txt','w',encoding='utf-8')
        log_file.write('file_name\tresult\tpred\tgt\n')
        dir_list=os.listdir(g_root_dir)
        dir_list.sort()
        #gts = open(os.path.join(g_root_dir,'result.txt')).readlines()
        gts = open('/ai/local/menglc/v3_dataset/raw_img/ctwReCTS_validate.txt').readlines()[:200]
        for i, data in enumerate(gts):
            img_path, gt = data.split('\t')
            gt = gt.split('\n')[0]
            print(count,end='\r')
            result=model.fileRecognition(img_path)
            name = img_path.split('/')[-1] 
            log_file.write('{}\t{}\t{}\t{}\n'.format(name,(result==gt),result,gt))
            if result==gt :
                correct_count+=1
            # log_file.write('{}\t{}\n'.format(name,result))
            count+=1        
    elif mode ==6 :
        log_file=open('Cut_result.txt','w',encoding='utf-8')
        log_file.write('file_name\tresult\tpred\tgt\n')
        #gt_file = open('/ai/local/menglc/v3_dataset/raw_img/ctwReCTS_validate.txt').readlines()
        # gt_file = open('/home/menglc/ocrdata/poster_simple/result.txt').readlines()
        gt_file = open('/home/menglc/ocrdata/cut_img/result.txt').readlines()
        # gt_file = open('/ai/local/menglc/LSVT/total_data/cut_validate/result.txt').readlines()
        gts=[]
        pil_images=[]
        for i, data in enumerate(gt_file):
            name, gt = data.split('\t')
            gt = gt.split('\n')[0]
            pil_image = Image.open(os.path.join(g_root_dir,name)).convert('L')
            pil_images.append(pil_image)
            gts.append(gt)
            print(i,end='\r')
        
        count=0      
        results = model.imageBatchRecognition(pil_images)
        print('len results: {}, len gts: {}, '.format(len(results),len(gts)))
        for result, gt in zip(results,gts):

            log_file.write('{}\t{}\t{}\t{}\n'.format('name',(result==gt),result,gt))
            if result==gt :
                correct_count+=1
            # log_file.write('{}\t{}\n'.format(name,result))
            count+=1
        
    elif mode == 7:
        AlignCollate_valid= AlignCollate(imgH=32, imgW=256, keep_ratio_with_pad=True ,c=1 )
        test_dataset = TestDataset('/ai/local/menglc/v3_dataset/raw_img/ctwReCTS_validate.txt')
        valid_loader = DataLoader(
            test_dataset, batch_size=32,
            shuffle=True,  
            collate_fn=AlignCollate_valid, pin_memory=True
        )

        count =0
        for i, (images, labels) in enumerate(valid_loader):
            print(i, end='\r')
            results = model._recognition(images)

            for result, label in zip(results, labels):
                if result == label:
                    correct_count +=1
            
                count+=1

    elif mode == 8 :
        import os 
        import os.path as osp 

        log_file=open('Cut_result.txt','w',encoding='utf-8')
        # image_root = '/home/menglc/ocrdata/SceneTextGenerator/poster_simple'
        image_root = '/ai/local/menglc/LSVT/total_data/validate'
        image_name = os.listdir(image_root)
        for i, name in enumerate(image_name) :
            if name.endswith('.jpg'):
                print(i//2, end='\r')
                image = cv2.imread(osp.join(image_root, name))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ps_str = open(osp.join(image_root,name.split('.')[0]+'.txt')).read().split('\n')
                pps = []
                labels = []
                for line in ps_str :
                    if line == '':
                        continue 
                    line = line.split(',')

                    ps = [int(float(x)) for x in line[:8]]
                    pps.append((ps[0:2],ps[2:4],ps[4:6],ps[6:8]))
                    labels.append(line[8])

                pps = np.array(pps)
                # print(type(pps), pps.shape)

                results = model.bboxPicRecognition(image, pps)

                for result, gt in zip(results, labels):
                    # print('pred\t:{}\ngt:\t{}'.format(result, gt))
                    log_file.write('{}\t{}\t{}\t{}\n'.format(name,(result==gt),result,gt))
                    if result==gt :
                        correct_count+=1
                        # log_file.write('{}\t{}\n'.format(name,result))
                    count+=1



    total_time=time.time()-start_time
    log_file.write('infer time:{}, avg:{}, acc:{}'.format(total_time,total_time/count,correct_count/count)) 
    print('infer time:{}, avg:{}, acc:{}'.format(total_time,total_time/count,correct_count/count))
