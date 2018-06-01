import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8

def dice_loss(preds, trues):
    batch = preds.size(0)
    classes = preds.size(1)
    preds = preds.view(batch, classes, -1)
    trues = trues.view(batch, classes, -1)
    weights = torch.clamp(trues.sum(-1), 0., 1.)
    intersection = (preds * trues).sum(2)
    scores = 2. * (intersection + eps) / (preds.sum(2) + trues.sum(2) + eps)
    scores = scores * weights
    score = scores.sum() / weights.sum()
    return torch.clamp(score, 0., 1.)

smooth = 1e-4
def fb_loss(preds, trues, beta):
    beta2 = beta*beta
    batch = preds.size(0)
    classes = preds.size(1)
    preds = preds.view(batch, classes, -1)
    trues = trues.view(batch, classes, -1)
    weights = torch.clamp(trues.sum(-1), 0., 1.)
    TP = (preds * trues).sum(2)
    FP = (preds * (1-trues)).sum(2)
    FN = ((1-preds) * trues).sum(2)
    Fb = ((1+beta2) * TP + smooth)/((1+beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = Fb.sum() / weights.sum()
    return torch.clamp(score, 0., 1.)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return 1 - dice_loss(input, target)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_coef):
        super().__init__()
        self.dice = DiceLoss()
        self.bce_coef = bce_coef

    def forward(self, input, target):
        bce = nn.BCELoss()(input, target)
        dice = self.dice(input, target)
        return self.bce_coef * bce + dice

class LyftLoss2(nn.Module):
    def __init__(self, bce_w=1, car_w=1):
        super().__init__()
        self.bce_w = bce_w
        self.car_w = car_w
        self.bce = nn.BCELoss()
        self.pad = nn.ReflectionPad2d((0, 0, 4, 4))

    def forward(self, input, target):
        target = self.pad(target)
        #bce_loss = self.bce(input, target)
        other = 1-fb_loss(input[:,:2,:,:], target[:,:2,:,:], 0.5)  # F0.5 road and bg
        car = 1-fb_loss(input[:,2,:,:].unsqueeze(1),
                        target[:,2,:,:].unsqueeze(1), 2)  # F2 car
        return self.car_w * car + other# + self.bce_w * bce_loss


class LyftLoss(nn.Module):
    def __init__(self, bce_w=1, car_w=1, other_w=1):
        super().__init__()
        self.bce_w = bce_w
        self.car_w = car_w
        self.other_w = other_w
        self.bce = nn.BCELoss()
        self.pad = nn.ReflectionPad2d((0, 0, 4, 4))

    def forward(self, input, target):
        target = self.pad(target)
        if self.bce_w > 0:
            bce_loss = self.bce(input, target)
        else:
            bce_loss = 0
        if self.other_w > 0:
            other = 1-fb_loss(input[:,0,:,:].unsqueeze(1), target[:,0,:,:].unsqueeze(1), 0.5)  # F0.5 bg
        else:
            other = 0
        road = 1-fb_loss(input[:,1,:,:].unsqueeze(1), target[:,1,:,:].unsqueeze(1), 0.5)  # F0.5 road
        if self.car_w > 0:
            car = 1-fb_loss(input[:,2,:,:].unsqueeze(1),
                            target[:,2,:,:].unsqueeze(1), 2)  # F2 car
        else:
            car = 0
        return self.car_w * car + self.other_w * other + road\
               + self.bce_w * bce_loss
