import torch 
import torch.nn as nn
import torchvision.models as models 
# from torch.nn.Identity import Identity
from modules.my_resnet import resnet50
from modules.my_transformer import * 
import math, copy 
import torch.nn.functional as F
import numpy as np 

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Identity(nn.Module):
    
    def __init__(self):
        super(Identity,self).__init__()
    
    def forward(self,x):
        return x 

class TransformerModel(nn.Module):

    def __init__(self,opt=None):
        super(TransformerModel,self).__init__()
        
        self.FeatureExtraction= resnet50(w_stride=2)
        self.feature_output_channel = opt.output_channel

        if self.feature_output_channel == 2048 :
            self.downsample= Identity()
        else :
            self.downsample= nn.Sequential(
                nn.Conv2d(2048, self.feature_output_channel,kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.feature_output_channel),
            )

        self.Decoder= self._buildDecoder(self.feature_output_channel)
        self.emdedding= nn.Sequential(
                            Embeddings(self.feature_output_channel, opt.num_class),
                            PositionalEncoding(opt.output_channel,0.1)
                        )

        self.generator= Generator(self.feature_output_channel, opt.num_class)

    
    def _buildDecoder(self, d_model=512, nhead=8, dim_feedforward=2048, num_decoder_layers=6, dropout=0.1, activation='relu'):

        # decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        # decoder_norm = nn.LayerNorm(d_model)
        # decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        attn = MultiHeadedAttention(nhead, d_model)
        ff = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        decoder_layer = DecoderLayer(d_model,attn,attn, ff, dropout )
        decoder= Decoder(decoder_layer, num_decoder_layers)

        for p in decoder.parameters():
            if p.dim() >1 :
                nn.init.xavier_uniform_(p)

        return decoder


    @staticmethod    
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)

        # m1 = torch.Tensor(subsequent_mask(tgt.size(-1)), dtype=torch.uint8)
        m1 = subsequent_mask(tgt.size(-1))
        print('tgt_mask type {}-{}'.format(tgt_mask.dtype,m1.dtype))
        
        tgt_mask = tgt_mask & m1  
        tgt_mask = (tgt != pad).unsqueeze(-2)
        return tgt_mask


    def forward(self, input_data, tgt, is_train=True):
        """
            remind : Batch First !!!!
            input_data : image (B,C,H,W)
            tgt : LongIntTensor (B,L), L is max_length with pad 
        """

        visual_feature = self.FeatureExtraction(input_data)
        visual_feature = self.downsample(visual_feature)
        b,c,_,_ = visual_feature.shape
        print(visual_feature.shape)
        # visual_feature = visual_feature.permute(2,3,0,1) # permute [b,c,h,w] -> [h,w,b,c]
        visual_feature = visual_feature.contiguous().view(b,-1,c)
    
        text_feature = self.emdedding(tgt) 

        print('features', visual_feature.shape, text_feature.shape)

        result = self.Decoder(text_feature, visual_feature, None, self.make_std_mask(tgt, 0))
        
        if is_train :
            result = self.generator(result)
        else:
            _, result = result.max(-1)

        return result 


class Option():
    output_channel=512 
    num_class= 4002

    def __init__(self):
        pass 


if __name__ =='__main__':
    opt= Option()
    model=TransformerModel(opt)
    # print(model)

    # # a=torch.randn(10,32,opt.output_channel)
    image = torch.randn(1,3,128,1024)
    tgt= torch.randint(0,opt.num_class,(1,25))


    # a= torch.randn(32,10,512)
    # b= torch.randn(32,5,512)
    # result = model.Decoder(b, a, None, None)
    result=model(image, tgt)
    print(result.shape)

    # a = torch.randn(32,10,512)
    # embeddings=nn.Sequential(PositionalEncoding(opt.output_channel,0.1))

    # result=embeddings(a)
    # print(result.shape)

