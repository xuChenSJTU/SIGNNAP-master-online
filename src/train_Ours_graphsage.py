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

from NCE_utils import NCEAverage, NCECriterion, NCESoftmaxLoss

#import for graphsage
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
import tqdm
from dgl.nn.pytorch.conv import SAGEConv

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.

    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])

def load_subtensor(g, labels, seeds, input_nodes):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].cuda()
    batch_labels = labels[seeds].cuda()
    return batch_inputs, batch_labels

#### Neighbor sampler

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

class SAGE_Sampling(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y

class SAGE_Full(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        super(SAGE_Full, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features, g):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h


# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
method_name = 'Ours_GraphSage'

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
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument("--mixmode", action="store_true",
                    default=False, help="Enable CPU GPU mixing mode.")
parser.add_argument("--warm_start", default="",
                    help="The model name to be loaded for warm start.")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--dataset', default="coauthor_phy", help="The data set. pubmed, facebook_page, coauthor_cs, coauthor_phy.")
parser.add_argument('--datapath', default="../data/", help="The data path.")
parser.add_argument("--early_stopping", type=int,
                    default=0, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
parser.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")

# Model parameter
parser.add_argument('--num_workers', type=int, default=1,
                    help='The number of workers.')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Number of batch size to train.')
parser.add_argument('--fan-out', type=str, default='10,25')

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
parser.add_argument('--nce_t', type=float, default=0.1)
parser.add_argument('--nce_m', type=float, default=0.5)

args = parser.parse_args()

# if args.debug:
print(args)

# save log path
log_pth = os.path.join(os.getcwd(), 'logs', method_name, args.dataset, 'train', 'latent_d_{}'.format(args.hidden), 
                    'type_{}_sampling_{}_softmax_{}_nce_k_{}_nce_t_{}_nce_m_{}_n_layer_{}'.format(args.type, args.sampling_percent,
                                                                                            args.softmax, args.nce_k, args.nce_t, args.nce_m, args.nhiddenlayer))

if os.path.exists(log_pth):
    os.system('rm -r {}'.format(log_pth))  # delete old log, the dir will be automatically built later
    
# set model save path
model_save_pth = os.path.join(os.getcwd(), 'saved_models', method_name, args.dataset, 'train', 'latent_d_{}'.format(args.hidden),
'type_{}_sampling_{}_softmax_{}_nce_k_{}_nce_t_{}_nce_m_{}_n_layer_{}'.format(args.type, args.sampling_percent, args.softmax,
                                                                              args.nce_k, args.nce_t, args.nce_m, args.nhiddenlayer))
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
labels, _, _, _ = sampler.get_label_and_idxes(args.cuda)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

# The model

model = SAGE_Full(nfeat,
                      args.hidden,
                      args.nhiddenlayer,
                      F.relu,
                      args.dropout,
                      'gcn')

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
all_node_index = torch.LongTensor(range(sampler.n_nodes))
# if torch.cuda.is_available():
#     index = index.cuda()
for epoch in range(args.epochs):
    model.train()
    
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

    # Construct feed data g
    if torch.cuda.is_available():
        train_edges_v1 = train_adj_v1._indices().cpu().data
    else:
        train_edges_v1 = train_adj_v1._indices().data
    train_edges_v1 = (train_edges_v1[0], train_edges_v1[1])
    train_g_v1 = dgl.graph(train_edges_v1)
    train_g_v1.ndata['features'] = train_fea_v1
    prepare_mp(train_g_v1)

    # Construct feed data g
    if torch.cuda.is_available():
        train_edges_v2 = train_adj_v2._indices().cpu().data
    else:
        train_edges_v2 = train_adj_v2._indices().data
    train_edges_v2 = (train_edges_v2[0], train_edges_v2[1])
    train_g_v2 = dgl.graph(train_edges_v2)
    train_g_v2.ndata['features'] = train_fea_v2
    prepare_mp(train_g_v2)


    feat_v1 = model(train_g_v1.ndata['features'], train_g_v1).cpu()
    feat_v2 = model(train_g_v2.ndata['features'], train_g_v2).cpu()
    out_v1, out_v2 = contrast(feat_v1, feat_v2, all_node_index)
    v1_loss = criterion_v1(out_v1).cuda()
    v2_loss = criterion_v2(out_v2).cuda()
    loss = v1_loss+v2_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss_v1_train = v1_loss.item()
    epoch_loss_v2_train = v2_loss.item()
    epoch_loss_train = epoch_loss_v1_train+epoch_loss_v2_train

    print('Epoch: {}, loss: {:.4f}, loss_v1: {:.4f}, loss_v2: {:.4f}'.format(epoch+1, epoch_loss_train, epoch_loss_v1_train, epoch_loss_v2_train))

    if args.no_tensorboard is False:
        tb_writer.add_scalars('Loss', {'train v1 loss': epoch_loss_v1_train, 'train v2 loss': epoch_loss_v2_train}, epoch)

    if (epoch+1)%args.freq==0:
        torch.save(model.state_dict(), os.path.join(model_save_pth, 'model_{}.ckpt'.format(epoch+1)))

print('Training Over........')