import torch
import torch.nn as nn
import torchvision.models as models

from .modules.transformation import TPS_SpatialTransformerNetwork
from .modules.sequence_modeling import BidirectionalLSTM
from .modules.my_resnet import resnet50 


class Identity(nn.Module):
    
    def __init__(self):
        super(Identity,self).__init__()
    
    def forward(self,x):
        return x 



class NewModel(nn.Module):

    def __init__(self,opt, is_pretrain=False):
        super(NewModel,self).__init__()
        self.opt=opt 
        if opt.Transformation=='TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else :
            self.Transformation= Identity()

        resnet=resnet50()
        # resnet.avgpool=Identity()
        # resnet.fc=Identity()

        self.FeatureExtraction=resnet
        self.AdaptiveAvgPool=nn.AdaptiveAvgPool2d((None, 1))
        self.feature_extration_output=opt.output_channel

        if self.feature_extration_output == 2048:
            print('do not need 1x1 conv !')
            self.downsample=Identity()
        else:
            self.downsample = nn.Sequential(
                    nn.Conv2d(2048, self.feature_extration_output,kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.feature_extration_output),
                )

        self.SequenceModeling= nn.Sequential(
            BidirectionalLSTM(self.feature_extration_output,opt.hidden_size,opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size,opt.hidden_size,opt.hidden_size)
        )

        # sequence_output= opt.hidden_size

        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(opt.hidden_size,opt.num_class)

        if is_pretrain:
            print('load origin state dict !')
            self.load_model()


    def forward(self, input_data, text, is_train=True ):

        input_data = self.Transformation(input_data)

        visual_feature= self.FeatureExtraction(input_data)
        visual_feature= self.downsample(visual_feature)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)


        contextual_feature= self.SequenceModeling(visual_feature)

        if self.opt.Prediction == 'CTC':
            prediction= self.Prediction(contextual_feature)

        return prediction

    def load_model(self):
        self.FeatureExtraction.load_state_dict(torch.load('myresnet.pth'))
        if self.opt.Prediction =='CTC':
            print('start init ctc !')
            for n in self.Prediction.parameters():
                nn.init.normal_(n,std=0.1)
        
        # self.SequenceModeling.load_state_dict(torch.load('lstm.pth'))
        # self.Prediction.load_state_dict(torch.load('ctc.pth'))