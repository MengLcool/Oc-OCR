"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
import sys 

from .modules.transformation import TPS_SpatialTransformerNetwork
from .modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor, SEResNet_FeatureExtractor
from .modules.sequence_modeling import BidirectionalLSTM
from .modules.prediction import Attention
# from transformer_decoder import TransFormerDecoder
from .py_trans import BertModel, BertConfig


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'SEResNet' :
            self.FeatureExtraction = SEResNet_FeatureExtractor(opt.input_channel,opt.output_channel)
        elif opt.FeatureExtraction == 'AResNet':
            assert opt.input_channel ==3, 'ACNN input_channel should be 3'
            from antialiased_cnns.models_lpf.resnet import resnet50 as acnn_resnet50 
            self.FeatureExtraction = acnn_resnet50(stride=(2,1),filter_size=3)
            self.FeatureExtraction_output = 2048
        else:
            raise Exception('No FeatureExtraction module specified')
        if opt.FeatureExtraction != 'AResNet':
            self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        elif opt.SequenceModeling == 'BERT':
            
            self.reduction = nn.Sequential(
                nn.Conv2d(self.FeatureExtraction_output, opt.hidden_size,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(opt.hidden_size),
            )

            bert_cfg = BertConfig(
                hidden_size= opt.hidden_size,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=512,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=1024        
            )
            self.SequenceModeling = BertModel(bert_cfg)
            self.SequenceModeling_output = opt.hidden_size
            

        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC' or opt.Prediction =='ENCTC' :
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        # elif opt.Prediction == 'Transformer':
        #     self.Prediction = TransFormerDecoder(opt.batch_max_length, num_classes=opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, text_pos=None, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)
        

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        
        if self.stages['Seq'] == 'BERT' or self.stages['Feat']=='AResNet':
            visual_feature = self.reduction(visual_feature)

        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        elif self.stages['Seq'] == 'BERT':
            contextual_feature = self.SequenceModeling(visual_feature)[0]

        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC' or self.stages['Pred'] == 'ENCTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        # elif self.stages['Pred'] == 'Transformer':
        #     prediction = self.Prediction(text, text_pos, contextual_feature.contiguous(), is_train)
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
        

        return prediction

