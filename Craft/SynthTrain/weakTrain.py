import torch 
from torch.utils.data import DataLoader 
import torch.nn.functional as F 
from synthDataLoader import SynthData
from synthLoss import cal_loss
import numpy as np 


import sys 
sys.path.append('..')
from craft import CRAFT 

THRESHOLD_POSITIVE=0.001
THRESHOLD_NEGATIVE=0

device = torch.device('cuda')


# TODO: 
# 1. distributeDataParallel 
# 2. weak train 

def hard_negative_mining(pred, target):
    all_loss = F.mse_loss(pred, target, reduction='none')
    
    positive = all_loss[target>THRESHOLD_NEGATIVE]
    negative = all_loss[target<=THRESHOLD_NEGATIVE]

    neg_len = min(3 * len(positive), len(negative))

    idx = negative.argsort(descending=True)
    negative = negative[idx[:neg_len]]

    # print('all loss ', all_loss.sum(), positive.shape, negative.shape)
    loss = (positive.sum() + negative.sum()) / (pred.shape[0])
    return loss 


def valid (model, valid_loader, criterion, with_sigmoid=False):
    
    loss_list = []
    for (images, labels) in valid_loader :
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)

            pred, _ = model(images)

            # pred = pred.new_zeros(pred.size())

            loss = cal_loss(pred, labels, criterion, with_sigmoid)

            # print('valid loss ',loss, 'shape', pred.shape)
            loss_list.append(loss)

    loss = torch.Tensor(loss_list).mean()
    print('loss avg ={}'.format(loss))

    return loss 

    


def train(train_set_path, validate_set_path, batch_size , with_sigmoid = False):
    train_set = SynthData(train_set_path)
    valid_set = SynthData(validate_set_path)


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)

    model = CRAFT()
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load('../pretrain/pretrain.pth'))

    model = model.to(device)
    # model.load_state_dict(torch.load('/home/menglc/CRAFT/pretrain/pretrain.pth'))


    print('success load ')

    # criterion = torch.nn.MSELoss(reduction='sum').to(device)
    criterion = hard_negative_mining
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epoch = 1000 

    best_loss = 1e10 


    for epoch in range(num_epoch):
        
        

        model.train()
        for i, (images, labels) in enumerate(train_loader):

            # validation
            if i % 1000 == 0 :
                model.eval()
                loss = valid(model, valid_loader, criterion, with_sigmoid)
                print('pre loss {} new loss {}'.format(best_loss, loss))
                if loss < best_loss :
                    best_loss = loss 
                    if epoch or i :
                        torch.save(model.state_dict(), 'best_v2.pth')


            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred, _ = model(images)
            loss = cal_loss(pred, labels, criterion, with_sigmoid = with_sigmoid)
            print('iter: {} | loss: {}'.format(i, loss), end='\r')

            loss.backward()
            optimizer.step()


def weaktrain()   


    


if __name__ == '__main__':
    
    train('/ai/local/menglc/CRAFT_dataset/train','/ai/local/menglc/CRAFT_dataset/valid',2, with_sigmoid=False)


    
