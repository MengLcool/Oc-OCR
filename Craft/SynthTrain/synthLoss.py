import torch 

def cal_loss(pred, gt , mse_criterion, with_sigmoid=False):
    """
    pred : tensor (b,h,w,2)
    gt : tensor (b,h,w,2)
    mse_criterion : MSELoss [ cuda() ]

    TODO: use sigmoid ? 
    """

    # pred[pred >1] = 1
    # pred[pred <0] = 0 

    if with_sigmoid:
        pred = torch.sigmoid(pred) 
    else :
        pred = pred.clamp(0,1)

    loss = mse_criterion(pred, gt)

    return loss 

