import math
from functools import partial
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F


def CE_Loss(inputs, target):
    CE_loss  = nn.BCEWithLogitsLoss()(inputs.float(), target.float())
    return CE_loss

def Dice_loss(inputs, targets, smooth=0.00001): 
    inputs = torch.sigmoid(inputs)
    #inputs, targets = inputs.float(), targets.float()
    #print(inputs, targets)

    a,b,c,d = inputs.size()
    sums = []
    
    for i in range(a):
        for j in range(b):
            img = inputs[i,j,:,:]
            label = targets[i,j,:,:]
            intersection = (img * label).sum()                            
            sums.append(1 - (2.*intersection + smooth)/(img.sum() + label.sum() + smooth))
            #sum = dice + sum
    #mean = sum / (a*b)
    return sum(sums) / len(sums)

def Boudaryloss(inputs, targets):
    #a,b,c,d = inputs.size()
    probs = torch.sigmoid(inputs)

    pc = probs[:, :, ...].type(torch.float32)
    dc = targets[:, :, ...].type(torch.float32)

    multipled = torch.einsum("bkwh,bkwh->bkwh", pc, dc)

    loss = multipled.mean()

    return loss

    
'''
def Dice_loss(inputs, targets):
    smooth = 1
    inputs = F.sigmoid(inputs)    
    inputs, targets = inputs.float(), targets.float()
    a,b,c,d = inputs.size()
    m1 = inputs.view(a*b, -1)
    m2 = targets.view(a*b, -1)
    intersection = (m1 * m2).sum()
    return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

'''
'''
def Dice_loss(inputs, targets):
    smooth = 0.00001
    inputs = F.sigmoid(inputs)  
    inputs, targets = inputs.float(), targets.float()
    a,b,c,d = inputs.size()
    num = a * b
    m1 = inputs.view(a, b, -1)
    m2 = targets.view(a, b, -1)
    m1 = m1.view(-1, m1.shape[2])
    m2 = m2.view(-1, m2.shape[2])
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    score = 1 - score.sum() / num 
    return score
'''      
class focal_loss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, size_average=True):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        #if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        #if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        pred_sigmoid = inputs.sigmoid()
        
        target = target.type_as(inputs)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) *
                        (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
        inputs, target, reduction='none') * focal_weight
        #print(loss)
        #loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        
        return loss.mean()


'''
def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    inputs, target = inputs.float(), target.float()    
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss
'''

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
