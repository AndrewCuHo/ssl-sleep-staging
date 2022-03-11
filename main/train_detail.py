import argparse
import os
import utils as util


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class train_detail():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):

        self._parser = argparse.ArgumentParser()
        self._parser.add_argument('--train_path', default='',
                                  type=str, help='train dataset path')
        self._parser.add_argument('--IF_TRAIN', type=str2bool, default=True, help='if test')
        self._parser.add_argument('--IF_GPU', type=str2bool, default=True, help='use of GPU')
        self._parser.add_argument('--model', default='cms2', type=str)
        self._parser.add_argument('--checkpoints', type=str, default='result_model', help='your checkpoints model name')
        self._parser.add_argument('--loss', type=str, default='loss')
        self._parser.add_argument('--iteration', type=int, default='0', help='Iteration per epoch')
        self._parser.add_argument('--resume', type=str2bool, default=False, help='if use checkpoints or not')
        self._parser.add_argument('--num_classes', type=int, default='5', help='number of classes')
        self._parser.add_argument('--input_height', type=int, default='50', help='size of input image')
        self._parser.add_argument('--input_weight', type=int, default='75', help='size of input image')
        self._parser.add_argument('--crop_size', type=int, default='100', help='size of resize image')
        self._parser.add_argument('--batch_size', type=int, default='10', help='batch to put in')
        self._parser.add_argument('--num_epochs', type=int, default='200', help='total epochs to train')
        self._parser.add_argument('--init_lr', type=float, default='0.001')
        self._parser.add_argument('--lr_scheduler', type=str, default='adamw')
        self._parser.add_argument('--use_ema', type=str2bool, default=True, help='use EMA')
        self._parser.add_argument('--use_mixup', type=str2bool, default=True, help='use Mixup')
        self._parser.add_argument('--step_size', type=int, default='20',
                                  help='step size betweeen next adjustion')
        self._parser.add_argument('--multiplier', type=float, default='10')
        self._parser.add_argument('--total_epoch', type=int, default='50')
        self._parser.add_argument('--alpha', type=float, default='0.8')
        self._parser.add_argument('--gamma', type=int, default='2')
        self._parser.add_argument('--manualSeed', type=int, default=110, help='manual seed')
        self._parser.add_argument('--out', default='./log')
        self._parser.add_argument('--UnlabeledPercent', type=int, default=50, help='Unlabeled img percent')
        self._parser.add_argument('--Distrib_Threshold', type=float, default=0.96, help='pseudo label')
        self._parser.add_argument('--Balance_loss', type=float, default=1, help='balance supervised and unsupervised')
        self._parser.add_argument('--epoch_idx', type=int, default=0)
        self._parser.add_argument('--Is_Visual', type=str2bool, default=False, help='Visual for feature map')
        self._parser.add_argument('--is_public', type=str2bool, default=False)

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()
        args = vars(self._opt)

        return self._opt

    def _myprint(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')