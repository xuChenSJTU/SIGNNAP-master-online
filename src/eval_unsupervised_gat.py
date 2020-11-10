from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

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

from evaluation_utils import LinearClassifier

import networkx as nx
import scipy.sparse as sp
from dgl import DGLGraph
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs, g):
        # inputs are the node features and g is the adj of graph in networkx form
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        h = self.gat_layers[-1](g, h).flatten(1)
        return h

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
method_name = 'Ours_GAT'

seed = np.random.randint(2020)

# Training settings
parser = argparse.ArgumentParser()
# Training parameter
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Disable validation during training.')
parser.add_argument('--seed', type=int, default=seed, help='Random seed. default=42')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--lradjust', action='store_true',
                    default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument("--mixmode", action="store_true",
                    default=False, help="Enable CPU GPU mixing mode.")
parser.add_argument("--warm_start", default="",
                    help="The model name to be loaded for warm start.")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--dataset', default="facebook_page", help="The data set, pubmed, facebook_page, coauthor_cs, coauthor_phy")
parser.add_argument('--datapath', default="../data/", help="The data path.")
parser.add_argument("--early_stopping", type=int,
                    default=400, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
parser.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")

# Model parameter
parser.add_argument('--num_heads', type=int, default=8,
                    help='Number of attention heads.')
parser.add_argument('--num_out_heads', type=int, default=8,
                    help='Number of out attention heads')
parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")

parser.add_argument('--type', default='multigcn',
                    help="Choose the model to be trained.(multigcn, resgcn, densegcn, inceptiongcn)")
parser.add_argument('--inputlayer', default='gcn',
                    help="The input layer of the model.")
parser.add_argument('--outputlayer', default='gcn',
                    help="The output layer of the model.")
parser.add_argument('--hidden', type=int, default=8,
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
parser.add_argument("--task_type", default="semi",
                    help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

# hyper-parameters for loading unsupervised model
parser.add_argument('--test_epoch', type=int, default=5000)
parser.add_argument('--softmax', default=False, help='using softmax contrastive loss rather than NCE')
parser.add_argument('--nce_k', type=int, default=1024)
parser.add_argument('--nce_t', type=float, default=0.1)
parser.add_argument('--nce_m', type=float, default=0.5)

args = parser.parse_args()
if args.debug:
    print(args)

# save log path
log_pth = os.path.join(os.getcwd(), 'logs', method_name, args.dataset, 'eval', 'latent_d_{}'.format(args.num_heads*args.hidden),
                       'type_{}_sampling_{}_softmax_{}_nce_k_{}_nce_t_{}_nce_m_{}_n_layer_{}'.format(args.type,
                                                                                          args.sampling_percent,
                                                                                          args.softmax, args.nce_k,
                                                                                          args.nce_t, args.nce_m, args.nhiddenlayer+1))
if os.path.exists(log_pth):
    os.system('rm -r {}'.format(log_pth))  # delete old log, the dir will be automatically built later

# set model save path
model_load_path = os.path.join(os.getcwd(), 'saved_models', method_name, args.dataset, 'train', 'latent_d_{}'.format(args.num_heads*args.hidden),
                               'type_{}_sampling_{}_softmax_{}_nce_k_{}_nce_t_{}_nce_m_{}_n_layer_{}'.format(args.type,
                                                                                                  args.sampling_percent,
                                                                                                  args.softmax,
                                                                                                  args.nce_k,
                                                                                                  args.nce_t,
                                                                                                  args.nce_m, args.nhiddenlayer+1))
model_save_pth = os.path.join(os.getcwd(), 'saved_models', method_name, args.dataset, 'eval', 'latent_d_{}'.format(args.num_heads*args.hidden),
                              'type_{}_sampling_{}_softmax_{}_nce_k_{}_nce_t_{}_nce_m_{}_n_layer_{}'.format(args.type,
                                                                                                 args.sampling_percent,
                                                                                                 args.softmax,
                                                                                                 args.nce_k, args.nce_t,
                                                                                                 args.nce_m, args.nhiddenlayer+1))
if not os.path.exists(model_save_pth):
    os.makedirs(model_save_pth)
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
    args.nhiddenlayer = 1
    args.aggrmethod = "nores"

# random seed setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda or args.mixmode:
    torch.cuda.manual_seed(args.seed)

# should we need fix random seed here?
sampler = Sampler(args.dataset, args.datapath, args.task_type)

# get labels and indexes
labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

# The model
classifier_model = LinearClassifier(input_dim=args.num_heads*args.hidden, output_dim=nclass)
heads = ([args.num_heads] * args.nhiddenlayer) + [args.num_out_heads]
unsupervised_model = GAT(args.nhiddenlayer,
                nfeat,
                args.hidden,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)


optimizer = optim.Adam(classifier_model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

early_stopping = EarlyStopping(patience=args.early_stopping, fname='best_classifier.model',
                               save_model_pth=model_save_pth)

# # define contrastive model
# contrast = NCEAverage(args.hidden, n_nodes, args.nce_k, args.nce_t, args.nce_m, args.softmax)
# criterion_v1 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_nodes)
# criterion_v2 = NCESoftmaxLoss() if args.softmax else NCECriterion(n_nodes)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
# convert to cuda
if args.cuda:
    classifier_model.cuda()
    unsupervised_model.cuda()
    # contrast.cuda()
    # criterion_v1.cuda()
    # criterion_v2.cuda()

# For the mix mode, lables and indexes are in cuda.
if args.cuda or args.mixmode:
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

if args.no_tensorboard is False:
    tb_writer = SummaryWriter(log_dir=log_pth, comment="-dataset_{}-type_{}".format(args.dataset, args.type))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# define the training function.
def train(epoch, train_adj, train_fea, idx_train, idx_val, val_adj=None, val_fea=None, labels=None):
    unsupervised_model.eval()

    if val_adj is None:
        val_adj = train_adj
        val_fea = train_fea

    t = time.time()
    classifier_model.train()

    optimizer.zero_grad()

    # construct g from train adj
    train_edges = train_adj._indices().data.cpu().numpy()
    train_edges = sp.coo_matrix((np.ones(train_edges.shape[1]),
                             (train_edges[0], train_edges[1])),
                            shape=(train_adj.shape[0], train_adj.shape[0]),
                            dtype=np.float32)

    train_g = nx.from_scipy_sparse_matrix(train_edges, create_using=nx.DiGraph())
    train_g = DGLGraph(train_g)

    feats = unsupervised_model(train_fea, train_g)

    output = classifier_model(feats)
    # special for inductive
    if sampler.learning_type == "inductive":
        loss_train = F.nll_loss(output, labels[idx_train])
        acc_train = accuracy(output, labels[idx_train])
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()
    train_t = time.time() - t
    val_t = time.time()
    # We can not apply the fastmode for the coauthor_phy dataset.
    # if sampler.learning_type == "inductive" or not args.fastmode:

    classifier_model.eval()
    if sampler.learning_type=='inductive':
        unsupervised_model.cpu()
        classifier_model.cpu()
        labels = labels.cpu()

    # construct g from val adj
    if sampler.learning_type=='inductive':
        val_edges = val_adj._indices().data.numpy()
    else:
        val_edges = val_adj._indices().data.cpu().numpy()

    val_edges = sp.coo_matrix((np.ones(val_edges.shape[1]),
                             (val_edges[0], val_edges[1])),
                            shape=(val_adj.shape[0], val_adj.shape[0]),
                            dtype=np.float32)

    val_g = nx.from_scipy_sparse_matrix(val_edges, create_using=nx.DiGraph())
    val_g = DGLGraph(val_g)
    
    feats = unsupervised_model(val_fea, val_g)

    output = classifier_model(feats)
    if args.early_stopping > 0 and sampler.learning_type != "inductive":
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        early_stopping(loss_val, classifier_model)

    if not args.fastmode:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        if sampler.learning_type == "inductive":
            early_stopping(loss_val, classifier_model)
    else:
        loss_val = 0
        acc_val = 0

    if sampler.learning_type=='inductive':
        unsupervised_model.cuda()
        classifier_model.cuda()
        labels = labels.cuda()

    if args.lradjust:
        scheduler.step()

    val_t = time.time() - val_t
    return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)


def test(test_adj, test_fea, idx_test, labels):
    unsupervised_model.eval()
    classifier_model.eval()

    if sampler.learning_type=='inductive':
        unsupervised_model.cpu()
        classifier_model.cpu()
        labels = labels.cpu()

    # construct g from test adj
    if sampler.learning_type=='inductive':
        test_edges = test_adj._indices().data.numpy()
    else:
        test_edges = test_adj._indices().data.cpu().numpy()

    test_edges = sp.coo_matrix((np.ones(test_edges.shape[1]),
                             (test_edges[0], test_edges[1])),
                            shape=(test_adj.shape[0], test_adj.shape[0]),
                            dtype=np.float32)

    test_g = nx.from_scipy_sparse_matrix(test_edges, create_using=nx.DiGraph())
    test_g = DGLGraph(test_g)
    feats = unsupervised_model(test_fea, test_g)

    output = classifier_model(feats)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    auc_test = roc_auc_compute_fn(output[idx_test], labels[idx_test])
    if args.debug:
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "auc= {:.4f}".format(auc_test),
              "accuracy= {:.4f}".format(acc_test.item()))
        print("accuracy=%.5f" % (acc_test.item()))
    return (loss_test.item(), acc_test.item())


# Train model
t_total = time.time()
loss_train = []
acc_train = []
loss_val = []
acc_val = []

sampling_t = 0

# load trained unsupervised model
unsupervised_model.load_state_dict(torch.load(os.path.join(model_load_path, 'model_{}.ckpt'.format(args.test_epoch))))

for epoch in range(args.epochs):
    input_idx_train = idx_train
    sampling_t = time.time()
    # no sampling
    # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.
    # train_fea_v1 and train_fea_v2 are the same
    (train_adj, train_fea) = sampler.randomedge_sampler(percent=args.sampling_percent, normalization=args.normalization,
                                                        cuda=args.cuda)
    if args.mixmode:
        train_adj = train_adj.cuda()

    sampling_t = time.time() - sampling_t

    if sampler.learning_type=='inductive':
        (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=False)
        idx_val = idx_val.cpu()
    else:
        (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
    
    outputs = train(epoch, train_adj, train_fea, input_idx_train, idx_val, val_adj, val_fea, labels)

    if (epoch + 1) % 1 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(outputs[0]),
              'acc_train: {:.4f}'.format(outputs[1]),
              'loss_val: {:.4f}'.format(outputs[2]),
              'acc_val: {:.4f}'.format(outputs[3]),
              'cur_lr: {:.5f}'.format(outputs[4]),
              's_time: {:.4f}s'.format(sampling_t),
              't_time: {:.4f}s'.format(outputs[5]),
              'v_time: {:.4f}s'.format(outputs[6]))

    if args.no_tensorboard is False:
        tb_writer.add_scalars('Loss', {'train': outputs[0], 'val': outputs[2]}, epoch)
        tb_writer.add_scalars('Accuracy', {'train': outputs[1], 'val': outputs[3]}, epoch)
        tb_writer.add_scalar('lr', outputs[4], epoch)
        tb_writer.add_scalars('Time', {'train': outputs[5], 'val': outputs[6]}, epoch)

    loss_train.append(outputs[0])
    acc_train.append(outputs[1])
    loss_val.append(outputs[2])
    acc_val.append(outputs[3])

    if args.early_stopping > 0 and early_stopping.early_stop:
        print("Early stopping.")
        classifier_model.load_state_dict(early_stopping.load_checkpoint())
        break

if args.early_stopping > 0:
    classifier_model.load_state_dict(early_stopping.load_checkpoint())

if args.debug:
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
if sampler.learning_type=='inductive':
    (test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=False)
    idx_test = idx_test.cpu()
else:
    (test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)

(loss_test, acc_test) = test(test_adj, test_fea, idx_test, labels)

print("best epoch: {}\t best val loss: {:.6f}\t test loss: {:.6f}\t test_acc: {:.10f}".format(np.argmin(loss_val),
                                                                                              -early_stopping.best_score,
                                                                                              loss_test, acc_test))
print(args)
