import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from utils import mixup_data, mixup_criterion
from train_detail import train_detail
train_opt = train_detail().parse()


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class CB_loss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, loss_type, beta, gamma, device):
        super(CB_loss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.device = device

    def forward(self, preds, truth):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(truth, self.no_of_classes).float()

        weights = torch.tensor(weights, device=self.device).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, preds, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=preds, target=labels_one_hot, weight=weights)
        elif self.loss_type == "softmax":
            pred = preds.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss


def get_negative_mask(batch_size):
    negative_mask = t.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = t.cat((negative_mask, negative_mask), 0)
    return negative_mask



def cdd(output_t1, output_t2, sim_weight, sim_labels, label_batch_size, targets_a, targets_b, lam, Is_Mix, target_or):
    one_hot_label = torch.zeros(label_batch_size * 20, 5, device=sim_labels.device)
    idx = sim_labels.view(-1, 1).long() - 1
    one_hot_label = one_hot_label.scatter(dim=-1, index=idx, value=1.0)
    min_value, max_value = sim_weight.min(), sim_weight.max()
    decide_map = (sim_weight - min_value) / (max_value - min_value)

    pred_scores = torch.sum(one_hot_label.view(label_batch_size, -1, 5) * decide_map.unsqueeze(dim=-1), dim=1)
    criterion = nn.CrossEntropyLoss().cuda()
    if Is_Mix:
        cdd_loss1 = 0.001 * mixup_criterion(criterion, pred_scores, targets_a, targets_b, lam)
    else:
        cdd_loss1 = 0.001 * criterion(pred_scores, target_or)
    mul = output_t1.transpose(0, 1).mm(output_t2)
    cdd_loss = 0.01 * (torch.sum(mul) - torch.trace(mul))
    if torch.isnan(cdd_loss1):
        return cdd_loss
    else:
        return cdd_loss + cdd_loss1.cuda()


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.num_matri = num
        self.positi = nn.ReLU()
    def forward(self, *x):
        loss_sum = 0
        loss_indentify = torch.ones(self.num_matri)
        for i, loss in enumerate(x):
            loss_sum += self.positi(torch.exp(- 2 * self.params[i]) * loss + self.params[i])
            loss_indentify[i] = self.positi(torch.exp(- 2 * self.params[i]) * loss + self.params[i])
        return loss_sum, loss_indentify.data.cpu().numpy()

def multi_smooth_loss(input, target, smooth_ratio=0.85, loss_weight=None,
                      weight=None, size_average=True, ignore_index=-100, reduce=True):
    assert isinstance(input, tuple), 'input is less than 2'
    criterion = CB_loss(
                        samples_per_cls=[1, 3, 1, 2, 3], no_of_classes=5, loss_type='focal', beta=0.75,
                        gamma=2.0,
                        device=torch.device('cuda'))

    weight_loss = torch.ones(len(input)).to(input[0].device)
    if loss_weight is not None:
        for item in loss_weight.items():
            weight_loss[int(item[0])] = item[1]
    loss = 0
    for i in range(0, len(input)):
        if i in [1, len(input) - 1]:
            prob = F.log_softmax(input[i], dim=1)
            ymask = prob.data.new(prob.size()).zero_()
            ymask = ymask.scatter_(1, target.view(-1, 1), 1)
            ymask = smooth_ratio * ymask + (1 - smooth_ratio) * (1 - ymask) / (input[i].shape[1] - 1)
            loss_tmp = - weight_loss[i] * ((prob * ymask).sum(1).mean())
        else:
            loss_tmp = weight_loss[i] * criterion(input[i], target)
        loss += loss_tmp
    return loss
def cal_unsupervised(output_u_w, output_u_s, criterion_u):
    pseudo_label = torch.softmax(output_u_w.detach(), dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(train_opt.Distrib_Threshold).float()
    max_probs = max_probs.max()
    Loss_u = (criterion_u(output_u_s, targets_u) * mask).mean()
    return Loss_u, max_probs

def smooth_unsupervise_loss(input, target, mask, smooth_ratio=0.85, loss_weight=None,
                      weight=None, size_average=True, ignore_index=-100, reduce=True):
    assert isinstance(input, tuple), 'input is less than 2'
    criterion = CB_loss(
                        samples_per_cls=[1, 3, 1, 2, 3], no_of_classes=5, loss_type='focal', beta=0.75,
                        gamma=2.0,
                        device=torch.device('cuda'))

    weight_loss = torch.ones(len(input)).to(input[0].device)
    if loss_weight is not None:
        for item in loss_weight.items():
            weight_loss[int(item[0])] = item[1]

    loss = 0
    for i in range(0, len(input)):
        if i in [1, len(input) - 1]:
            prob = F.log_softmax(input[i], dim=1)
            ymask = prob.data.new(prob.size()).zero_()
            ymask = ymask.scatter_(1, target[i].view(-1, 1), 1)
            ymask = smooth_ratio * ymask + (1 - smooth_ratio) * (1 - ymask) / (input[i].shape[1] - 1)
            loss_tmp = - weight_loss[i] * ((prob * ymask).sum(1).mean()) * mask[i]
        else:
            loss_tmp = weight_loss[i] * criterion(input[i], target[i]) * mask[i]
        loss += loss_tmp

    return loss
