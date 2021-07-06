import torch 
import torch.nn as nn 
import torch.nn.functional as F
from transformer.Models import Transformer
import transformer.Constants as Constants
from modules.my_resnet import resnet50 

class Identity(nn.Module):
    
    def __init__(self):
        super(Identity,self).__init__()
    
    def forward(self,x):
        return x 



class TransformerModel(nn.Module):

    def __init__(self, opt, is_pretrain=False):
        super(TransformerModel, self).__init__()
        self.opt = opt 
        
        self.PAD=0 
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.PAD).cuda()
        resnet=resnet50()

        self.FeatureExtraction=resnet 
        
        self.feature_extration_output = opt.output_channel 



        if self.feature_extration_output != 2048 :
            self.reduction = nn.Sequential(
                    nn.Conv2d(2048, self.feature_extration_output,kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.feature_extration_output),
                )
        else :
            self.reduction = Identity()

        h_out,w_out = opt.imgH//4, opt.imgW//32 

        self.Prediction = Transformer(
            len_encoder= h_out * w_out,
            n_tgt_vocab= opt.num_class,
            len_max_seq= opt.batch_max_length + 2 
        )
        
        if is_pretrain:
            self.load_pretrain()

    # TODO: modelify  input_data pos 
    def extractVisualFeature(self, image_data, image_pos=None):
        visual_feature = self.FeatureExtraction(image_data) # b,c,h,w
        visual_feature = self.reduction(visual_feature)
        b,c,h,w = visual_feature.shape
        visual_feature = visual_feature.permute(0,3,2,1) # b,w,h,c
        visual_feature = visual_feature.contiguous().view(b, -1, c)
           
        if image_pos is None :
            visual_pos = [list(range(1,h*w+1))]*b
            visual_pos = torch.cuda.LongTensor(visual_pos)
        
        return visual_feature, visual_pos


    def forward(self ,input_data, tgt, len_labels):

        visual_feature = self.FeatureExtraction(input_data) # b,c,h,w
        visual_feature = self.reduction(visual_feature)
        b,c,h,w = visual_feature.shape
        visual_feature = visual_feature.permute(0,2,3,1) # b,h,w,c
        visual_feature = visual_feature.contiguous().view(b, -1, c)
        
        # TODO: fix len_features 
        # len_features = [h*w] * b
        len_features = [list(range(1,h*w+1))] * b 
        len_features = torch.cuda.LongTensor(len_features)

        # print('visual {}, len {}, gtg {},len label {}'.format(visual_feature.dtype, len_features.dtype, tgt.dtype, len_labels.dtype))
        prediction = self.Prediction.forward(visual_feature,len_features, tgt, len_labels)

        return prediction 

    def load_pretrain(self):
        self.FeatureExtraction.load_state_dict(torch.load('myresnet.pth'))
        print('success load pretrain model !')

# def cal_loss(model, criterion, input_data, labels, len_labels, is_label_smoothing=True):
#     # labels= labels.cuda()
#     result= model(input_data, labels, len_labels)
#     # labels = labels[:,1:]
    
#     if is_label_smoothing:
#         eps = 0.1 
#         num_class= result.size(1)
#         one_hot = torch.zeros_like(result).scatter(1, labels.view(-1,1), 1)
#         one_hot = one_hot *(1- eps) + (1- one_hot)* eps / (num_class-1)

#         log_prob= F.log_softmax(result, dim=1)
#         non_pad_mask = labels.view(-1).ne(0)
#         loss = -(one_hot * log_prob).sum(dim=1)
#         loss = loss.masked_select(non_pad_mask).sum()
#         loss /= non_pad_mask.sum()
    
#     else:
#         loss = criterion(result, labels.view(-1))
    
#     return loss 

def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''


    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='mean')

    return loss




class Option():
    imgH=32
    imgW=256
    num_class=4001
    batch_max_length=35

    def __init__(self):
        pass 

if __name__ == '__main__':
    opt=Option()
    model = TransformerModel(opt)
    print(model)