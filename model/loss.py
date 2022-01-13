import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tm
import numpy as np
from torch.autograd import Variable
from skimage.morphology import medial_axis, skeletonize


# ========================================
# nll_loss
# ========================================
def nll_loss(output, target):
    return F.nll_loss(output, target)


# ========================================
# bce_loss
# ========================================
def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)


# ========================================
# focal_loss
# ========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def focal_loss(input, target, weight=1, gamma=5, ignore_index=None):
    if ignore_index is None:
        target[target == 2] = 0
    return weight * FocalLoss(gamma=gamma)(input, target.squeeze(1).long())


# ========================================
# bce_dice_loss
# ========================================
class BCEDiceLoss(nn.Module):
    def __init__(self, penalty_weight=None, size_average=True):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE Loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        if self.penalty_weight:
            dice_loss = self.penalty_weight * (1 - dice_coef)
        else:
            dice_loss = (1 - dice_coef)

        loss = bce_loss + dice_loss
        return loss, bce_loss, dice_loss


def bce_dice_loss(input, target, penalty_weight=1, weight=1):
    return weight * BCEDiceLoss(penalty_weight=penalty_weight)(input, target)


# ========================================
# bce_dice_centerline_loss
# for CasNet
# ========================================
def bce_dice_centerline_loss(input, target, penalty_weight=1, weight=1):
    target = target.numpy()
    skel, dist = medial_axis(target, return_distance=True)
    target = skel.astype(np.int32)
    return weight * BCEDiceLoss(penalty_weight=penalty_weight)(input, target)


# ========================================
# bce_dice_logits_loss
# ========================================
class BCEDiceLogitsLoss(nn.Module):
    def __init__(self, penalty_weight=None, size_average=True):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE Loss
        bce_loss = nn.BCEWithLogitsLoss()(pred, truth).double()

        # Dice Loss
        pred = nn.Sigmoid()(pred)
        dice_coef = (2. * (pred * truth).double().sum() + 1) / (pred.double().sum() + truth.double().sum() + 1)

        if self.penalty_weight:
            dice_loss = self.penalty_weight * (1 - dice_coef)
        else:
            dice_loss = (1 - dice_coef)

        loss = bce_loss + dice_loss
        return loss, bce_loss, dice_loss


def bce_dice_logits_loss(input, target, penalty_weight=1, weight=1):
    return weight * BCEDiceLogitsLoss(penalty_weight=penalty_weight)(input, target)


# ========================================
# lsgan_loss
# ========================================
def lsgan_loss(input, target):
    target = target.repeat(1, input.size(1))
    return F.binary_cross_entropy_with_logits(input, target)


# ========================================
# vgg_loss
# ========================================
def vgg_loss(input, target, vgg=None):
    input_1 = F.interpolate(input, scale_factor=0.5, mode='bilinear')
    target_1 = F.interpolate(target, scale_factor=0.5, mode='bilinear')

    inputs = [input_1]
    targets = [target_1]

    # layers: [14, 25, 32]
    block1 = vgg.features[: 15]
    block2 = vgg.features[15: 26]
    block3 = vgg.features[26: 33]
    blocks = [block1, block2, block3]
    for bl in blocks:
        for idx in range(len(bl)):
            p = bl[idx]
            p.requires_grad = False
            if 'ReLU' in str(p):
                bl[idx] = nn.ReLU(inplace=False)
    total_loss = 0.
    for idx in range(len(inputs)):
        x = inputs[idx].repeat(1, 3, 1, 1)
        y = targets[idx].repeat(1, 3, 1, 1)
        loss = 0.
        for block in blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        total_loss += loss
    return total_loss


# ========================================
# mp_loss
# - missing part loss
# ========================================
def mp_loss(maps, partials, completes):
    # missing part
    mp = (completes - partials)
    loss = nn.BCELoss(reduction='none')(maps, mp) * mp
    loss = loss.mean()
    return loss


if __name__ == '__main__':
    maps = torch.randint(0, 2, (4, 1, 4, 4)).float().requires_grad_()
    partials = torch.randint(0, 2, (4, 1, 4, 4))
    completes = torch.ones((4, 1, 4, 4))
    # loss = mp_loss(maps, partials, completes)
    loss = nn.BCELoss(reduction='none')(maps, (completes - partials))
    print(loss)
    loss_mask = loss * (completes - partials)
    print(loss_mask)
    # loss = loss.mean()
    # loss.backward()
