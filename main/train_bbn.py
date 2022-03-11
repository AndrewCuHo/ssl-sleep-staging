import os
import random
import time
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import pickle
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt
from common.misc import mkdir_p
from sklearn.preprocessing import OneHotEncoder
from loss import *
from model.utils import select_network, freeze_param
from train_detail import train_detail
from utils import accuracy, select_sehcduler, split_data, opcounter, split_data_addition, plot_roc, \
    plt_roc_curve, plot_visial, GaussianBlur, weights_init, BalancedBatchSampler, calweight, accuracy_combine, \
    mixup_data, mixup_criterion, select_data
from torch.nn import functional as F
from common.cca import CCAHook
from tqdm import tqdm
import warnings
from common.hessian import get_target
from collections import Counter
from common.ema import ModelEMA
from torch.autograd import Variable
from dataset import read_dataset, random_dataset
import logging

warnings.filterwarnings("ignore")
train_opt = train_detail().parse()
train_path = train_opt.train_path
Num_classes = train_opt.num_classes
Size_height = train_opt.input_height
Size_weight = train_opt.input_weight
Model = train_opt.model
Checkpoint = train_opt.checkpoints
Resume = train_opt.resume
Loss = train_opt.loss
Num_epochs = train_opt.num_epochs
Batch_size = train_opt.batch_size
Init_lr = train_opt.init_lr
Lr_scheduler = train_opt.lr_scheduler
Step_size = train_opt.step_size
Multiplier = train_opt.multiplier
Total_epoch = train_opt.total_epoch
Alpha = train_opt.alpha
Gamma = train_opt.gamma
Re = train_opt.re
ManualSeed = train_opt.manualSeed
torch.manual_seed(ManualSeed)
torch.cuda.manual_seed_all(ManualSeed)
np.random.seed(ManualSeed)
random.seed(ManualSeed)
torch.backends.cudnn.deterministic = True
OutpuDir = train_opt.out
UnP = train_opt.UnlabeledPercent / 100
PThreshold = train_opt.Distrib_Threshold
Un_lamda = train_opt.Balance_loss
IF_GPU = train_opt.IF_GPU
IF_TRAIN = train_opt.IF_TRAIN
Is_EMA = train_opt.use_ema
Is_Mix = train_opt.use_mixup
Is_public = train_opt.is_public
Iteration_max = train_opt.iteration
best_acc = 0
best_kappa = 0
def train():
    global feature_bank
    global best_acc
    global best_kappa
    begin_time = time.time()
    model = select_network(Model, Num_classes)
    if Is_EMA:
        ema_model = ModelEMA(model)

    if IF_GPU:
        model.cuda()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        model.cpu()
    Normal_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((train_opt.crop_size, train_opt.crop_size)),
        transforms.ToTensor()])
    Strong_transform = transforms.Compose([
        GaussianBlur(15),
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.01, 0.01, 0.01),
        transforms.Resize((train_opt.crop_size, train_opt.crop_size)),
        transforms.ToTensor(),
    ])
    Iteration_max = len(x_unlabel_strong) // 10
    torch.device('cuda')
    criterion = CB_loss(samples_per_cls=[1, 20, 3, 4, 10], no_of_classes=5, loss_type='focal', beta=0.75,
                            gamma=2.0,device=torch.device('cuda'))
    awl = AutomaticWeightedLoss(3)

    optimizer = t.optim.Adam([
        {'params': model.parameters(), 'weight_decay': 3e-5, 'lr': Init_lr},
        {'params': awl.parameters(), 'weight_decay': 0, 'lr': 1e-3}
    ])
    sehcduler = select_sehcduler(Lr_scheduler, optimizer, Step_size, Multiplier, Num_epochs, Batch_size)
    cca_flag = 0

    if IF_TRAIN:
        print('Start training')
        # -------- train --------------------#
        for epoch in range(start_epoch + 1, Num_epochs):
            feature_bank = []
            ep_start = time.time()
            ema_to_save = ema_model.ema.module if hasattr(
                        ema_model.ema, "module") else ema_model.ema
                checkpoint = {
                    "net": model.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if Is_EMA else None,
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    'lr_schedule': sehcduler.state_dict()
                }
                for data_memory, _ in tqdm(meta_data):
                    test_model = ema_model.ema
                    test_model.eval()
                    if IF_GPU:
                        data_memory = data_memory.cuda()
                    else:
                        data_memory = data_memory.cpu()
                    with torch.no_grad():
                        feature_cdd_t = test_model(data_memory, feature_or=True)
                    feature_bank.append(feature_cdd_t)
                feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
                feature_bank = F.normalize(feature_bank)
                feature_weight_labels = y_cal_debiased.clone().detach().cuda()
                unlabeled_s_iter = iter(strong_unlabeldataloader_train)
                unlabeled_w_iter = iter(weak_unlabeldataloader_train)
                labeled_iter = iter(dataloader_train)
                sehcduler.step()
                for batch_idx in tqdm(range(Iteration_max)):
                    model.train()
                    try:
                        ims, labels = labeled_iter.__next__()
                    except:
                        labeled_iter = iter(dataloader_train)
                        ims, labels = labeled_iter.__next__()
                    try:
                        ims_strong, _ = unlabeled_s_iter.__next__()
                    except:
                        unlabeled_s_iter = iter(strong_unlabeldataloader_train)
                        ims_strong, _ = unlabeled_s_iter.__next__()

                    try:
                        ims_weak, _ = unlabeled_w_iter.__next__()
                    except:
                        unlabeled_w_iter = iter(weak_unlabeldataloader_train)
                        ims_weak, _ = unlabeled_w_iter.__next__()
                    if IF_GPU:
                        target = labels.cuda().long()
                        target = target - 1
                        get_target(target)
                        target = labels.cpu().long()
                        target = target - 1
                    if IF_GPU:
                        input_ub_w = ims_weak.cuda()
                        input_ub_s = ims_strong.cuda()
                        input_ob = ims.cuda()
                    else:
                        input_ub_w = ims_weak.cpu()
                        input_ub_s = ims_strong.cpu()
                        input_ob = ims.cpu()
                    label_batch_size = ims.shape[0]
                    if cca_flag == 0:
                        cca_flag += 1
                        hook1 = CCAHook(model, "sampler_buffer.out.0", svd_device="cuda")
                        hook2 = CCAHook(model, "sampler_buffer0.out.0", svd_device="cuda")
                    input_ob, targets_a, targets_b, lam = mixup_data(input_ob, target)
                    input_ob, targets_a, targets_b = map(Variable, (input_ob, targets_a, targets_b))
                    if IF_GPU:
                        inputs = torch.cat((input_ob, input_ub_w, input_ub_s)).cuda()
                    else:
                        inputs = torch.cat((input_ob, input_ub_w, input_ub_s)).cpu()
                    outputs, outputs_1, outputs_2, outputs_3 = model(inputs)
                    sim_matrix = torch.mm(model(input_ob, feature_or=True), feature_bank)

                    sim_matrix = torch.where(torch.isinf(sim_matrix), torch.full_like(sim_matrix, 1e-6),
                                             sim_matrix)
                    sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                    sim_labels = torch.gather(feature_weight_labels.expand(input_ob.size(0), -1), dim=-1,
                                              index=sim_indices)
                    sim_weight = (sim_weight / 0.5).exp()
                    if epoch < 6 and cca_flag == 0:
                        with torch.no_grad():
                            coeff1 = abs(hook1.distance(hook2))
                    output = outputs[:label_batch_size]
                    output_u_w, output_u_s = outputs[label_batch_size:].chunk(2)
                    output_1 = outputs_1[:label_batch_size]
                    output_u_w_1, output_u_s_1 = outputs_1[label_batch_size:].chunk(2)
                    output_2 = outputs_2[:label_batch_size]
                    output_u_w_2, output_u_s_2 = outputs_2[label_batch_size:].chunk(2)
                    output_3 = outputs_3[:label_batch_size]
                    output_u_w_3, output_u_s_3 = outputs_3[label_batch_size:].chunk(2)
                    loss_dis = cdd(F.softmax(output_2, dim=1), F.softmax(output_3, dim=1), sim_weight,
                                   sim_labels,
                                   input_ob.size(0), targets_a, targets_b, lam, Is_Mix=Is_Mix, target_or=target)
                    criterion_s = multi_smooth_loss
                    loss = mixup_criterion(criterion_s, (output, output_1, output_2, output_3), targets_a, targets_b, lam).mean()
                    max_probs1, targets_u_1 = torch.max(torch.softmax(output_u_w.detach(), dim=-1), dim=-1)
                    mask1 = max_probs1.ge(train_opt.Distrib_Threshold).float()
                    max_probs2, targets_u_2 = torch.max(torch.softmax(output_u_w_1.detach(), dim=-1), dim=-1)
                    mask2 = max_probs2.ge(train_opt.Distrib_Threshold).float()
                    max_probs3, targets_u_3 = torch.max(torch.softmax(output_u_w_2.detach(), dim=-1), dim=-1)
                    mask3 = max_probs3.ge(train_opt.Distrib_Threshold).float()
                    max_probs4, targets_u_4 = torch.max(torch.softmax(output_u_w_3.detach(), dim=-1), dim=-1)
                    mask4 = max_probs4.ge(train_opt.Distrib_Threshold).float()
                    mask = (mask1, mask2, mask3, mask4)
                    output_u_all = (output_u_s, output_u_s_1, output_u_s_2, output_u_s_3)
                    target_u_all = (targets_u_1, targets_u_2, targets_u_3, targets_u_4)
                    Loss_u = smooth_unsupervise_loss(output_u_all,target_u_all, mask).mean()
                    max_probs = ((max_probs4 + max_probs1 + max_probs2 + max_probs3) / 4).mean()
                    loss_sum1 = loss
                    loss_sum2 = Loss_u
                    loss_sum3 = loss_dis
                    loss, loss_identify = awl(loss_sum1, loss_sum2, loss_sum3)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    sehcduler.batch_step()
                    ema_model.update(model)

if __name__ == '__main__':
    if not os.path.isdir(OutpuDir):
        mkdir_p(OutpuDir)
    train()
