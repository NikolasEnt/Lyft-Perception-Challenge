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

class LyftLoss(nn.Module):
    def __init__(self, bce_coef, car_coef):
        super().__init__()
        self.car_coef = car_coef
        self.bcedice = BCEDiceLoss(bce_coef)
        self.pad = nn.ReflectionPad2d((0, 0, 4, 4))

    def forward(self, input, target):
        target = self.pad(target)
        other = self.bcedice(input[:,:2,:,:], target[:,:2,:,:])
        car = self.bcedice(input[:,2,:,:].unsqueeze(1),
                      target[:,2,:,:].unsqueeze(1))
        return self.car_coef * car + other
