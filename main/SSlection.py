import copy
import random
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function
#from model.resnest import *
from train_detail import train_detail
from model.resnet import ResNet34, ResNet34_addition, smooth_leaky_relu
from model.resnet_m import ResNet_with_BotStack
from model.efficientNet import EfficientNet
from model.RepVGG import create_RepVGG_A0, create_RepVGG_A1
#from efficientnet_pytorch import EfficientNet
from model.transformer import BottleStack
from utils import weights_init
from common.hessian import Hessian
from model.conv_transformer import MHConvAttention
import timm

train_opt = train_detail().parse()
if train_opt.IF_GPU:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class ScaleLayer(nn.Module):
    def __init__(self, init_value=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
    def forward(self, input):
        return input.to(device) * self.scale
class Weightfactor(nn.Module):
    def __init__(self, num=2):
        super(Weightfactor, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.num_matri = num
        self.positi = nn.ReLU()
    def forward(self, *x):
        loss_indentify = []
        for i, loss in enumerate(x):
            loss_indentify.append(self.positi(torch.exp(- 2 * self.params[i]) * loss + self.params[i]))
        return loss_indentify[0], loss_indentify[1]
def makeGaussian(size, fwhm=3, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
class KernelGenerator(nn.Module):
    def __init__(self, size, offset=None):
        super(KernelGenerator, self).__init__()

        self.size = self._pair(size)
        xx, yy = np.meshgrid(np.arange(0, size), np.arange(0, size))
        if offset is None:
            offset_x = offset_y = size // 2
        else:
            offset_x, offset_y = self._pair(offset)
        self.factor = torch.from_numpy(-(np.absolute(xx - offset_x) + np.absolute(yy - offset_y)) / 2).float()
    @staticmethod
    def _pair(x):
        return (x, x) if isinstance(x, int) else x
    def forward(self, theta):
        pow2 = torch.pow(theta * self.size[0], 2)
        kernel = 1.0 / (2 * np.pi * pow2) * torch.exp(self.factor.to(theta.device) / pow2)
        return kernel / kernel.max()
def kernel_generate(theta, size, offset=None):
    return KernelGenerator(size, offset)(theta)
def _mean_filter(input):
    batch_size, num_channels, h, w = input.size()
    threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
    return threshold.contiguous().view(batch_size, num_channels, 1, 1)


class PeakStimulation(Function):

    @staticmethod
    def forward(ctx, input, return_aggregation, win_size, peak_filter):
        ctx.num_flags = 4
        offset = (win_size - 1) // 2
        padding = torch.nn.ConstantPad2d(offset, float('-inf'))
        padded_maps = padding(input)
        batch_size, num_channels, h, w = padded_maps.size()
        element_map = torch.arange(0, h * w).long().view(1, 1, h, w)[:, :, offset: -offset, offset: -offset]
        element_map = element_map.to(input.device)
        _, indices = F.max_pool2d(
            padded_maps,
            kernel_size=win_size,
            stride=1,
            return_indices=True)
        peak_map = (indices == element_map)

        if peak_filter:
            mask = input >= peak_filter(input)
            peak_map = (peak_map & mask)
        peak_list = torch.nonzero(peak_map)
        ctx.mark_non_differentiable(peak_list)

        if return_aggregation:
            peak_map = peak_map.float()
            ctx.save_for_backward(input, peak_map)
            return peak_list, (input * peak_map).view(batch_size, num_channels, -1).sum(2) / \
                   peak_map.view(batch_size, num_channels, -1).sum(2)
        else:
            return peak_list

    @staticmethod
    def backward(ctx, grad_peak_list, grad_output):
        input, peak_map, = ctx.saved_tensors
        batch_size, num_channels, _, _ = input.size()
        grad_input = peak_map * grad_output.view(batch_size, num_channels, 1, 1) / \
                     (peak_map.view(batch_size, num_channels, -1).sum(2).view(batch_size, num_channels, 1, 1) + 1e-4)   #1e-6
        return (grad_input,) + (None,) * ctx.num_flags


def peak_stimulation(input, return_aggregation=True, win_size=3, peak_filter=None):
    return PeakStimulation.apply(input, return_aggregation, win_size, peak_filter)

def weights_init_kaiming(m,  scale=1):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()

class CMS2(nn.Module):

    def __init__(self, num_classes, task_input_size, base_ratio, radius, radius_inv):
        super(CMS2, self).__init__()
        self.grid_size = 25
        self.padding_size = 24
        self.global_size = self.grid_size + 2 * self.padding_size
        self.input_size_net = task_input_size
        gaussian_weights = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1, fwhm=19))
        self.base_ratio = base_ratio
        self.radius = ScaleLayer(radius)
        self.radius_inv = ScaleLayer(radius_inv)
        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1, 2 * self.padding_size + 1), bias=False)
        self.filter.weight[0].data[:, :, :] = gaussian_weights.to(device)
        self.P_basis = torch.zeros(2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        for k in range(2):
            for i in range(self.global_size):
                for j in range(self.global_size):
                    self.P_basis[k, i, j] = k * (i - self.padding_size) / (self.grid_size - 1.0) + (1.0 - k) * (
                            j - self.padding_size) / (self.grid_size - 1.0)
        self.extraction_features = 2048
        self.num_features = 1024
        self.classifier_feature = 1024
        self.first_feature = 512
        self.middle_feature = 256
        self.weight_factor = Weightfactor(2)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.features = timm.create_model('seresnext26d_32x4d', num_classes=0, global_pool='', pretrained=True, in_chans=15)
        self.sampler_buffer = MHConvAttention(embedding_dim=self.extraction_features, out_dim=self.num_features)
        self.sampler_buffer0 = MHConvAttention(embedding_dim=self.extraction_features, out_dim=self.num_features)
        self.sampler_buffer_raw = MHConvAttention(embedding_dim=self.extraction_features, out_dim=self.num_features)
        self.map_origin = nn.Conv2d(self.num_features, num_classes, 1, 1, 0)
        self.original_state = nn.Sequential(nn.Conv2d(self.extraction_features, self.middle_feature, kernel_size=1, stride=1, padding=0, bias=False),
                                            nn.BatchNorm2d(self.middle_feature),
                                            nn.ReLU(inplace=True)
                                            )
        self.map_origin.apply(weights_init_kaiming)
        self.original_state.apply(weights_init_kaiming)
        self.raw_classifier = nn.Linear(self.num_features, num_classes)
        self.con_linear = nn.Linear(self.num_features * 3, num_classes)
        self.bi_classifier1_dense = nn.Linear(self.num_features, self.classifier_feature)
        self.bi_classifier2_dense = nn.Linear(self.num_features, self.classifier_feature)

        self.raw_classifier.apply(weights_init_classifier)
        self.con_linear.apply(weights_init_classifier)
        self.bi_classifier1_dense.apply(weights_init_classifier)
        self.bi_classifier2_dense.apply(weights_init_classifier)
        self.Bi_classifier1 = nn.Sequential(nn.Dropout(p=0.1),
                                            nn.Linear(self.classifier_feature, self.first_feature),
                                            nn.BatchNorm1d(self.first_feature, affine=True),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=0.1),
                                            nn.Linear(self.first_feature, self.middle_feature),
                                            nn.BatchNorm1d(self.middle_feature, affine=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.middle_feature, num_classes)
                                            )
        self.Bi_classifier2 = nn.Sequential(nn.Dropout(p=0.1),
                                            nn.Linear(self.classifier_feature, self.first_feature),
                                            nn.BatchNorm1d(self.first_feature, affine=True),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=0.1),
                                            nn.Linear(self.first_feature, self.middle_feature),
                                            nn.BatchNorm1d(self.middle_feature, affine=True),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.middle_feature, num_classes)
                                            )
        self.Bi_classifier1.apply(weights_init_classifier)
        self.Bi_classifier2.apply(weights_init_classifier)
    def create_grid(self, x):
        P = torch.autograd.Variable(
            torch.zeros(1, 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size).to(device),
            requires_grad=False)
        P[0, :, :, :] = self.P_basis
        P = P.expand(x.size(0), 2, self.grid_size + 2 * self.padding_size, self.grid_size + 2 * self.padding_size)
        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x)
        x_mul = torch.mul(P, x_cat).view(-1, 1, self.global_size, self.global_size)
        all_filter = self.filter(x_mul).view(-1, 2, self.grid_size, self.grid_size)
        x_filter = all_filter[:, 0, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        y_filter = all_filter[:, 1, :, :].contiguous().view(-1, 1, self.grid_size, self.grid_size)
        x_filter = x_filter / p_filter
        y_filter = y_filter / p_filter
        xgrids = x_filter * 2 - 1
        ygrids = y_filter * 2 - 1
        xgrids = torch.clamp(xgrids, min=-1, max=1)
        ygrids = torch.clamp(ygrids, min=-1, max=1)
        xgrids = xgrids.view(-1, 1, self.grid_size, self.grid_size)
        ygrids = ygrids.view(-1, 1, self.grid_size, self.grid_size)
        grid = torch.cat((xgrids, ygrids), 1)
        grid = F.interpolate(grid, size=(self.input_size_net, self.input_size_net), mode='bilinear', align_corners=True)
        grid = torch.transpose(grid, 1, 2)
        grid = torch.transpose(grid, 2, 3)
        return grid

    def generate_map(self, input_x, class_response_maps, hess_g1, if_val=False):
        N, C, H, W = class_response_maps.size()
        score_pred, sort_number = torch.sort(F.softmax(F.adaptive_avg_pool2d(class_response_maps, 1),
                                                       dim=1), dim=1, descending=True)
        gate_score = (score_pred[:, 0:3] * torch.log(score_pred[:, 0:3])).sum(1)
        gate_score = gate_score * hess_g1
        xs = []
        xs_inv = []
        xs_soft = []
        for idx_i in range(N):
            if not if_val:
                if gate_score[idx_i] < -365:
                    decide_map = class_response_maps[idx_i, sort_number[idx_i, 0:2], :, :].mean(0)
                else:
                    decide_map = class_response_maps[idx_i, sort_number[idx_i, 0:5], :, :].mean(0)
            else:
                if gate_score[idx_i] < -1.6:
                    decide_map = class_response_maps[idx_i, sort_number[idx_i, 0:2], :, :].mean(0)
                else:
                    decide_map = class_response_maps[idx_i, sort_number[idx_i, 0:5], :, :].mean(0)

            min_value, max_value = decide_map.min(), decide_map.max()
            decide_map = (decide_map - min_value) / (max_value - min_value)

            peak_list, aggregation = peak_stimulation(decide_map, win_size=3, peak_filter=_mean_filter)

            decide_map = decide_map.squeeze(0).squeeze(0)

            score = [decide_map[item[2], item[3]] for item in peak_list]
            x = [item[3] for item in peak_list]
            y = [item[2] for item in peak_list]
            if score == []:
                temp = torch.zeros(1, 1, self.grid_size, self.grid_size).to(device)
                temp += self.base_ratio
                xs.append(temp)
                xs_soft.append(temp)
                continue
            peak_num = torch.arange(len(score))
            temp = self.base_ratio
            temp_w = self.base_ratio
            for i in peak_num:
                temp += score[i] * kernel_generate(self.radius(torch.sqrt(score[i])), H,
                                                   (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).to(device)
                temp_w += 100 * score[i] * \
                          kernel_generate(self.radius_inv(torch.sqrt(score[i])), H,
                                          (x[i].item(), y[i].item())).unsqueeze(0).unsqueeze(0).to(device)
                temp, temp_w = self.weight_factor(temp, temp_w)
            if type(temp) == float:
                temp += torch.zeros(1, 1, self.grid_size, self.grid_size).to(device)
            xs.append(temp)
            if type(temp_w) == float:
                temp_w += torch.zeros(1, 1, self.grid_size, self.grid_size).to(device)
            xs_inv.append(temp_w)
        xs = torch.cat(xs, 0)
        xs_hm = nn.ReplicationPad2d(self.padding_size)(xs)
        grid = self.create_grid(xs_hm).to(input_x.device)
        x_sampled_zoom = F.grid_sample(input_x, grid)
        xs_inv = torch.cat(xs_inv, 0)
        xs_hm_inv = nn.ReplicationPad2d(self.padding_size)(xs_inv)
        grid_inv = self.create_grid(xs_hm_inv).to(input_x.device)
        x_sampled_inv = F.grid_sample(input_x, grid_inv)
        return x_sampled_zoom, x_sampled_inv

    def forward(self, input_x, **kwargs):
            feature_raw = self.features(input_x)
            self.map_origin.weight.data.copy_(self.raw_classifier.weight.data.unsqueeze(-1).unsqueeze(-1))
            self.map_origin.bias.data.copy_(self.raw_classifier.bias.data)
            with torch.no_grad():
                class_response_maps = F.interpolate(self.map_origin(self.sampler_buffer_raw(feature_raw)), size=self.grid_size, mode='bilinear',
                                                    align_corners=True)
            x_sampled_zoom, x_sampled_inv = self.generate_map(input_x, class_response_maps, p, 1, if_val=True)
            feature_zoom = self.sampler_buffer(self.features(x_sampled_zoom))
            feature_inv = self.sampler_buffer0(self.features(x_sampled_inv))
            feature_raw = self.sampler_buffer_raw(feature_raw)
            agg_origin = self.raw_classifier(self.avg(feature_raw).view(-1, self.num_features))
            aggregation = self.con_linear(self.avg(torch.cat([feature_raw, feature_zoom, feature_inv], 1)).view(-1, self.num_features * 3))
            att_classify_zoom = self.avg(feature_zoom)
            att_classify_inv = self.avg(feature_inv)
            att_classify_1 = att_classify_zoom
            att_classify_2 = att_classify_inv
            att_classify_1 = att_classify_1.view(att_classify_1.size(0), -1)
            att_classify_1 = self.bi_classifier1_dense(att_classify_1)
            att_classify_1 = att_classify_1.view(att_classify_1.size(0), self.classifier_feature)
            agg_sampler_1 = self.Bi_classifier1(att_classify_1)
            att_classify_2 = att_classify_2.view(att_classify_2.size(0), -1)
            att_classify_2 = self.bi_classifier2_dense(att_classify_2)
            att_classify_2 = att_classify_2.view(att_classify_2.size(0), self.classifier_feature)
            agg_sampler_2 = self.Bi_classifier2(att_classify_2)
            return aggregation, agg_origin, agg_sampler_1, agg_sampler_2
        self.map_origin.weight.data.copy_(self.raw_classifier.weight.data.unsqueeze(-1).unsqueeze(-1))
        self.map_origin.bias.data.copy_(self.raw_classifier.bias.data)
        feature_raw = self.features(input_x)
        if "feature_or" in kwargs:
            feature_origi = self.original_state(feature_raw)
            feature_origi = self.avg(feature_origi)
            feature_or = torch.flatten(feature_origi, start_dim=1)
            return feature_or
        with torch.no_grad():
            class_response_maps = F.interpolate(self.map_origin(self.sampler_buffer_raw(feature_raw)), size=self.grid_size, mode='bilinear',
                                                align_corners=True)
        hess_f = F.adaptive_avg_pool2d(self.original_state(feature_raw), 1)
        hess_prob = F.softmax(hess_f, dim=1)
        temp_map = self.original_state
        Hess_score = Hessian(hess_f=hess_f, hess_prob=hess_prob,
                                map_origin=temp_map).compute_G_decomp()
        hess_g1 = torch.from_numpy(np.asarray(Hess_score['dist']))
        hess_g1 = hess_g1[:5, :5]
        hess_g1 = hess_g1.mean()
        lamda_g1 = 1 / hess_g1
        x_sampled_zoom, x_sampled_inv = self.generate_map(input_x, class_response_maps, p, lamda_g1)
        feature_zoom = self.sampler_buffer(self.features(x_sampled_zoom))
        feature_inv = self.sampler_buffer0(self.features(x_sampled_inv))
        att_classify_zoom = self.avg(feature_zoom)
        att_classify_inv = self.avg(feature_inv)
        att_classify_1 = att_classify_zoom
        att_classify_2 = att_classify_inv
        att_classify_1 = att_classify_1.view(att_classify_1.size(0), -1)
        att_classify_1 = self.bi_classifier1_dense(att_classify_1)
        att_classify_1 = att_classify_1.view(att_classify_1.size(0), self.classifier_feature)
        agg_sampler_1 = self.Bi_classifier1(att_classify_1)
        att_classify_2 = att_classify_2.view(att_classify_2.size(0), -1)
        att_classify_2 = self.bi_classifier2_dense(att_classify_2)
        att_classify_2 = att_classify_2.view(att_classify_2.size(0), self.classifier_feature)
        agg_sampler_2 = self.Bi_classifier2(att_classify_2)
        feature_raw = self.sampler_buffer_raw(feature_raw)
        agg_origin = self.raw_classifier(self.avg(feature_raw).view(-1, self.num_features))
        aggregation = self.con_linear(self.avg(torch.cat([feature_raw, feature_zoom, feature_inv], 1)).view(-1, self.num_features * 3))
        return aggregation, agg_origin, agg_sampler_1, agg_sampler_2
def cms2():
    model = CMS2(5, 100, 0.09, 0.0001, 0.0001)
    return model
