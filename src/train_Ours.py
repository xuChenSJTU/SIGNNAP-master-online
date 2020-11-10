from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from earlystopping import EarlyStopping
from sample import Sampler
from metric import accuracy, roc_auc_compute_fn
# from deepgcn.utils import load_data, accuracy
# from deepgcn.models import GCN

from metric import accuracy
from utils import load_citation
from model_Ours import *
from earlystopping import EarlyStopping
from sample import Sampler

from NCE_utils import NCEAverage, NCECriterion, NCESoftmaxLoss

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
method_name = 'Ours'  # default is Ours which means Ours(GCN), if type==resgcn, it is Ours(ResGCN)

# seed = np.random.randint(100)

# Training settings
parser = argparse.ArgumentParser()
# Training parameter 
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Disable validation during training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed. default=42')
parser.add_argument('--freq', type=int, default=500, help='save frequency')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--lradjust', action='store_true',
                    default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
parser.add_argument('--weight_decay', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument("--mixmode", action="store_true",
                    default=False, help="Enable CPU GPU mixing mode.")
parser.add_argument("--warm_start", default="",
                    help="The model name to be loaded for warm start.")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--dataset', default="pubmed", help="The data set. pubmed, facebook_page, coauthor_cs, coauthor_phy")
parser.add_argument('--datapath', default="../data/", help="The data path.")
parser.add_argument("--early_stopping", type=int,
                    default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
parser.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")

# Model parameter
parser.add_argument('--type', default='multigcn',
                    help="Choose the model to be trained.(multigcn, resgcn, densegcn, inceptiongcn)")
parser.add_argument('--inputlayer', default='gcn',
                    help="The input layer of the model.")
parser.add_argument('--outputlayer', default='gcn',
                    help="The output layer of the model.")
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--withbn', default=False,
                    help='Enable Bath Norm GCN')
parser.add_argument('--withloop', default=True,
                    help="Enable loop layer GCN")
parser.add_argument('--nhiddenlayer', type=int, default=1,
                    help='The number of hidden layers.')
parser.add_argument("--normalization", default="AugRWalk",
                    help="AugRWalk, AugNormAdj, BingGeNormAdj, The normalization on the adj matrix.")
parser.add_argument("--sampling_percent", type=float, default=0.7,
                    help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
# parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, multigcn, inceptiongcn).")
parser.add_argument("--nbaseblocklayer", type=int, default=0,
                    help="The number of layers in each baseblock")
parser.add_argument("--aggrmethod", default="default",
                    help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn")
parser.add_argument("--task_type", default="self-supervised", help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

parser.add_argument('--softmax', default=False, help='using softmax contrastive loss rather than NCE')
parser.add_argument('--nce_k', type=int, default=1024)
parser.add_argument('--nce_t', type=float, default=0.1, help='temperature')
parser.add_argument('--nce_m', type=float, default=0.5)

args = parser.parse_args()
# if args.debug:
print(args)

if args.type=='resgcn':
    method_name = 'Ours_resgcn' 
    # args.nbaseblocklayer = 1  # for GCN backbone and ResGCN backbone
elif args.type=='densegcn':
    # args.nbaseblocklayer = 1
    method_name = 'Ours_jknet' # nbaseblocklayer is set according to the setting, usually args.nbaseblocklayer = 1 has best performance
else:
    # args.nbaseblocklayer = 0  # for GCN backbone and ResGCN backbone
    print('The default method is Ours(GCN)')
    pass

# save log path
log_pth = os.path.join(os.getcwd(), 'logs', method_name, args.dataset, 'train', 'latent_d_{}'.format(args.hidden),
'type_{}_sampling_{}_softmax_{}_nce_k_{}_nce_t_{}_nce_m_{}_n_layer_{}'.format(args.type, args.sampling_percent,
                                args.softmax, args.nce_k, args.nce_t, args.nce_m, args.nbaseblocklayer*args.nhiddenlayer+2))

if os.path.exists(log_pth):
    os.system('rm -r {}'.format(log_pth))  # delete old log, the dir will be automatically built later

# set model save path
model_save_pth = os.path.join(os.getcwd(), 'saved_models', method_name, args.dataset, 'train', 'latent_d_{}'.format(args.hidden),
'type_{}_sampling_{}_softmax_{}_nce_k_{}_nce_t_{}_nce_m_{}_n_layer_{}'.format(args.type, args.sampling_percent, args.softmax, 
                                                                              args.nce_k, args.nce_t, args.nce_m, args.nbaseblocklayer*args.nhiddenlayer+2))
if not os.path.exists(model_save_pth):
    os.makedirs(model_save_pth) # make save model path, if exists, the models will be overwritten

training_save_path = os.path.join(os.getcwd(), 'saved_training', method_name, args.dataset)
if not os.path.exists(training_save_path):
    os.makedirs(training_save_path)

# pre setting
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.mixmode = args.no_cuda and args.mixmode and torch.cuda.is_available()
if args.aggrmethod == "default":
    if args.type == "resgcn":
        args.aggrmethod = "add"
    else:
        args.aggrmethod = "concat"
if args.fastmode and args.early_stopping > 0:
    args.early_stopping = 0
    print("In the fast mode, early_stopping is not valid option. Setting early_stopping = 0.")
if args.type == "multigcn":
    print("For the multi-layer gcn model, the aggrmethod is fixed to nores and nhiddenlayers = 1.")
    # args.nhiddenlayer = 1
    args.aggrmethod = "nores"

# random seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda or args.mixmode:
    torch.cuda.manual_seed(args.seed)

# should we need fix random seed here?
sampler = Sampler(args.dataset, args.datapath, args.task_type)

# get labels and indexes
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

# The model
model = GCNModel(nfeat=nfeat,
                 nhid=args.hidden,
                 nclass=nclass,
                 nhidlayer=args.nhiddenlayer,
                 dropout=args.dropout,
                 baseblock=args.type,
                 inputlayer=args.inputlayer,
                 outputlayer=args.outputlayer,
                 nbaselayer=args.nbaseblocklayer,
                 activation=F.relu,
                 withbn=args.withbn,
                 withloop=args.withloop,
                 aggrmethod=args.aggrmethod,
                 mixmode=args.mixmode)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# define contrastive model
contrast = NCEAverage(args.hidden, sampler.n_nodes, args.nce_k, args.nce_t, args.nce_m, args.softmax, args.dataset)
criterion_v1 = NCESoftmaxLoss() if args.softmax else NCECriterion(sampler.n_nodes)
criterion_v2 = NCESoftmaxLoss() if args.softmax else NCECriterion(sampler.n_nodes)

# # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
# convert to cuda
if args.cuda:
    model.cuda()
    # contrast.cuda()
    # criterion_v1.cuda()
    # criterion_v2.cuda()

if args.warm_start is not None and args.warm_start != "":
    early_stopping = EarlyStopping(fname=args.warm_start, verbose=False)
    print("Restore checkpoint from %s" % (early_stopping.fname))
    model.load_state_dict(early_stopping.load_checkpoint())

# set early_stopping
if args.early_stopping > 0:
    early_stopping = EarlyStopping(patience=args.early_stopping, verbose=False)
    print("Model is saving to: %s" % (early_stopping.fname))

if args.no_tensorboard is False:
    tb_writer = SummaryWriter(log_dir=log_pth, comment="-dataset_{}-type_{}".format(args.dataset, args.type))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Train model
t_total = time.time()
loss_train = []

sampling_t = 0
index = torch.LongTensor(range(sampler.n_nodes))
# if torch.cuda.is_available():
#     index = index.cuda()
for epoch in range(args.epochs):

    sampling_t = time.time()
    # no sampling
    # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.
    # train_fea_v1 and train_fea_v2 are the same
    (train_adj_v1, train_fea_v1) = sampler.randomedge_sampler(percent=args.sampling_percent, normalization=args.normalization,
                                                        cuda=args.cuda)
    (train_adj_v2, train_fea_v2) = sampler.randomedge_sampler(percent=args.sampling_percent, normalization=args.normalization,
                                                        cuda=args.cuda)
    # if args.mixmode:
    #     train_adj_v1 = train_adj_v1.cuda()
    #     train_adj_v2 = train_adj_v2.cuda()

    sampling_t = time.time() - sampling_t

    feat_v1 = model(train_fea_v1, train_adj_v1)
    feat_v2 = model(train_fea_v2, train_adj_v2)

    if args.cuda:
        feat_v1 = feat_v1.cpu()
        feat_v2 = feat_v2.cpu()
    out_v1, out_v2 = contrast(feat_v1, feat_v2, index)
    v1_loss = criterion_v1(out_v1).cuda()
    v2_loss = criterion_v2(out_v2).cuda()
    loss = v1_loss+v2_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_train.append([loss.item()])
    
    print('Epoch: {}, loss: {:.4f}, loss_v1: {:.4f}, loss_v2: {:.4f}'.format(epoch+1, loss.item(), v1_loss.item(), v2_loss.item()))

    if args.no_tensorboard is False:
        tb_writer.add_scalars('Loss', {'train v1 loss': v1_loss.item(), 'train v2 loss': v1_loss.item()}, epoch)

    if (epoch+1)%args.freq==0:
        torch.save(model.state_dict(), os.path.join(model_save_pth, 'model_{}.ckpt'.format(epoch+1)))

# np.save(os.path.join(training_save_path, 'unsupervised_model_train_loss_sampling_{}_n4.npy'.format(args.sampling_percent)), np.array(loss_train))
print('Training Over........')
print(args)