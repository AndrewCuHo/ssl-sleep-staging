import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from train_detail import train_detail
from numpy import linalg as LA
from torch.autograd import Variable

train_opt = train_detail().parse()
Hess_label = torch.zeros(train_opt.batch_size)
Hess_batch = 0
Hess_epoch = 0

def get_target(target):
    global Hess_label
    Hess_label = target

def get_batch(batch):
    global Hess_batch
    Hess_batch = batch

def get_epoch(epoch):
    global Hess_epoch
    Hess_epoch = epoch

def give_value(means_t, counters_t):
    return means_t, counters_t


class Hessian:
    def __init__(self,
                 num_classes=5,
                 class_list=None,
                 hess_f=None,
                 hess_prob=None,
                 map_origin=None,
                 if_val=False
                 ):
        self.if_val = if_val
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.class_list = class_list
        self.f = hess_f
        self.prob = hess_prob
        self.model = map_origin
        self.target = Hess_label
        self.batch = Hess_batch
        self.epoch = Hess_epoch
    def my_randn(self):
        v_0_l = []
        for param in self.model.parameters():
            Z = torch.randn(param.shape)

            Z = Z.to(self.device)

            v_0_l.append(Z)

        return v_0_l
    def my_zero(self):
        return [0 for x in self.my_randn()]

    def my_sub(self, X, Y):
        return [x - y for x, y in zip(X, Y)]

    def my_sum(self, X, Y):
        return [(x + y).cuda() for x, y in zip(X, Y)]

    def my_inner(self, X, Y):
        return sum([torch.dot(x.view(-1), y.view(-1)) for x, y in zip(X, Y)])

    def my_mult(self, X, Y):
        return [x * y for x, y in zip(X, Y)]

    def my_norm(self, X):
        return torch.sqrt(self.my_inner(X, X))

    def my_mult_const(self, X, c):
        return [x * c for x in X]

    def my_div_const(self, X, c):
        return [x / c for x in X]

    def my_data(self, X):
        return [x.data for x in X]

    def my_cpu(self, X):
        return [x.cpu() for x in X]

    def my_device(self, X):
        return [x.to(self.device) for x in X]

    def compute_delta_c_cp(self):
        f = self.f
        prob = self.prob
        target = self.target
        means = []
        counters = []
        class_list = [i for i in range(5)]
        for c in class_list:
            means.append([])
            counters.append([])
            for cp in class_list:
                means[-1].append(None)
                counters[-1].append(0)
        for idx_c, c in enumerate(class_list):
            idxs = (target == c).nonzero()
            if len(idxs) == 0:
                continue

            fc = f[idxs.squeeze(-1),]
            probc = prob[idxs.squeeze(-1),]
            for idx_cp, cp in enumerate(class_list):
                w = -probc
                w[:, cp] = w[:, cp] + 1
                w = w * torch.sqrt(probc[:, [cp]])
                J = torch.autograd.grad(fc, self.model.parameters(),
                                            grad_outputs=w,
                                            retain_graph=True)
                J = self.my_cpu(self.my_data(J))
                if means[idx_c][idx_cp] is None:
                    means[idx_c][idx_cp] = self.my_zero()

                means[idx_c][idx_cp] = self.my_sum(means[idx_c][idx_cp], J)
                counters[idx_c][idx_cp] += fc.shape[0]
        for idx_c in range(len(class_list)):
            for idx_cp in range(len(class_list)):
                means[idx_c][idx_cp] = [x / counters[idx_c][idx_cp] for x in means[idx_c][idx_cp]]

        return means
    def compute_G_decomp(self):
        mu_ccp = self.compute_delta_c_cp()

        C = len(mu_ccp)

        mu_ccp_flat = []
        for c in range(C):
            for c_ in range(C):
                mu_ccp_flat.append(mu_ccp[c][c_])
        mu = []
        for c in range(C):
            s = self.my_zero()
            for c_ in range(C):
                if c != c_:
                    s = self.my_sum(s, mu_ccp[c][c_])
            avg = self.my_div_const(s, C - 1)
            mu.append(avg)
        V = []
        labels = []
        for c in range(C):
            V.append(mu[c])
            labels.append([c])
        for c in range(C):
            for c_ in range(C):
                V.append(mu_ccp[c][c_])
                labels.append([c, c_])

        N = C + C ** 2
        dist = np.zeros([N, N])
        for c in range(N):
            for c_ in range(N):
                dist[c, c_] = self.my_norm(self.my_sub(V[c], V[c_])) ** 2
        res = {'mu_ccp': mu_ccp,
               'mu_ccp_flat': mu_ccp_flat,
               'mu': mu,
               'dist': dist,
               'labels': labels,
               }

        return res
