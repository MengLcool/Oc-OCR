import torch 
import torch.nn as nn 
from transformer.Models import Decoder 
import transformer.Constants as Constants
import torch.nn.functional as F 

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
    pred = pred.contiguous().view(-1, pred.size(-1))

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss



class TransFormerDecoder(nn.Module):

    def __init__(self, max_batch_length, num_classes, d_vec=256, d_inner=512):
        """
        max_batch_length : should less than 128 
        """
        super(TransFormerDecoder, self).__init__()
        
        self.max_batch_length = max_batch_length

        self.decoder= Decoder(
            n_tgt_vocab= num_classes,
            len_max_seq= 128,
            d_word_vec=d_vec ,
            n_layers= 4,
            n_head= 16,
            d_k= 64,
            d_v= 64,
            d_model= d_vec,
            d_inner= d_inner
        )

        self.tgt_word_prj = nn.Linear(d_vec, num_classes, bias=False)



    def forward(self, tgt_seq, tgt_pos, enc_output, is_train =True):
    
        fake_src_seq = torch.ones(enc_output.shape[0],enc_output.shape[1] ,device=enc_output.device)

        if is_train:
            dec_output, *_ = self.decoder(tgt_seq, tgt_pos, fake_src_seq, enc_output)
            seq_logit = self.tgt_word_prj(dec_output)

            return seq_logit
        else :
            
            device = enc_output.device 
            b = enc_output.size(0)
            # ys = torch.ones(b, 1, dtype=torch.long, device=device).fill_(Constants.BOS)
            ys = torch.cuda.LongTensor(b, self.max_batch_length+1).fill_(Constants.BOS)
            ys_pos = torch.arange(1, self.max_batch_length+1, dtype=torch.long, device=device).unsqueeze(0).expand(b,-1)
            for i in range(1, self.max_batch_length+1):
                
                # ys_pos = ys_pos.unsqueeze(0).expand(b,-1)
                dec_output, *_ = self.decoder(ys[:,:i], ys_pos[:,:i], fake_src_seq, enc_output)
                _, next_words = self.tgt_word_prj(dec_output[:,-1,:]).max(-1)

                ys[:,i]= next_words
                # ys= torch.cat([ys, next_wrods.type_as(ys).unsqueeze(1)], dim=1)
                # ys_pos = torch.cat([ys_pos, torch.tensor(i+1, device=ys_pos.device).expand()])
                
            return ys[:,1:] 

if __name__ =='__main__':
    decoder = TransFormerDecoder(30, 5000).cuda()
    batch_size = 2 
    enc_output = torch.Tensor(batch_size,64,256).cuda()

    import time
    print('start pred')
    time_start = time.time()


    for _ in range(100):
        tgt_seq = torch.randint(1,512,(batch_size,30)).long().cuda()
        tgt_pos = torch.arange(1,31).unsqueeze(0).expand(batch_size,-1).long().cuda()
        preds = decoder(tgt_seq, tgt_pos, enc_output, False)
    
    print('end pred , preds size: {}, time:{}'.format(preds.size(),(time.time() - time_start)/100))
