import torch 
import torch.nn as nn 

# TODO: exactly import 
from py_trans import * 
from modules.my_resnet import resnet50 

class Identity(nn.Module):
    
    def __init__(self):
        super(Identity,self).__init__()
    
    def forward(self,x):
        return x 


class TransfoXLModel(nn.Module):
    
    def __init__(self, opt, is_pretrain=False):

        super(TransfoXLModel, self).__init__()
        self.opt = opt 
        
        self.PAD=0 

        # transfoxl_cfg= BertConfig(
        #     d_embed=opt.output_channel,
        #     d_model=512,
        #     n_head=6,
        #     d_head=32,
        #     d_inner=1024,
        #     tgt_len=opt.max_batch_length,
        #     n_layer=6,
        #     cutoffs=[500,1000,3000],
        #     num_labels = opt.num_class
        
        # )


        transfoxl_cfg= BertConfig(
            vocab_size_or_config_json_file=4000,
            hidden_size=opt.output_channel,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=1024,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            max_position_embeddings=512,
            num_labels = opt.num_class
        
        )
        
        resnet=resnet50()

        self.featureExtraction=resnet 

        # self.feature_extration_output = opt.output_channel 


        if opt.output_channel  != 2048 :
            self.reduction = nn.Sequential(
                    nn.Conv2d(2048, opt.output_channel ,kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(opt.output_channel ),
                )
        else :
            self.reduction = Identity()

        self.transformer= BertForTokenClassification(transfoxl_cfg)

    def forward(self,x, labels=None):
        
        visual_feature= self.featureExtraction(x)
        visual_feature = self.reduction(visual_feature)

        b,c,h,w= visual_feature.shape
        # visual_feature = visual_feature.permute(3,2,0,1).view(-1,b,c)
        visual_feature = visual_feature.permute(0,3,2,1).view(b,-1,c)

        output= self.transformer(visual_feature, labels=labels)

        return output 

    def loadModel(self):
        pass 


class Option():
    output_channel= 512
    max_batch_length = 35 
    num_class= 4003 

    def __init__(self):
        pass 

if __name__ == '__main__':
    
    opt=Option()
    model = TransfoXLModel(opt)
    model.train()
    model= model.cuda()


    images= torch.randn(128,3,16,256).cuda()

    labels = torch.randint(0,4000,(128,64)).cuda()

    # outputs = model(images, labels=labels)[0]
    
    import time 

    start = time.time()
    # for _ in range(100):
    #     result = model(images)
    
    result = model(images,labels)[0]

    print(result)


    print('avg {}'.format((time.time()- start)/100 ))

    # print(outputs.shape, outputs)

    # print(result.shape)
    # print(result)


