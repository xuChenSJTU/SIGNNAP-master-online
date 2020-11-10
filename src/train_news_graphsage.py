from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os
import torch as th

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
from models import *
from earlystopping import EarlyStopping
from sample import Sampler

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
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
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
        return F.log_softmax(h, dim=1)

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

                h = x[input_nodes].cuda()
                h_dst = h[:block.number_of_dst_nodes()]
                h = F.log_softmax(layer(block, (h, h_dst)), dim=1)
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
                 n_classes,
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
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None)) # activation None

    def forward(self, features, g):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return F.log_softmax(h, dim=1)

# os.environ['CUDA_VISIBLE_DEVICES'] = '9'
method_name = 'DropEdge_GraphSage'

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
parser.add_argument('--lr', type=float, default=0.01,
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
parser.add_argument('--dataset', default="coauthor_phy", help="The data set. pubmed, facebook_page, coauthor_cs, coauthor_phy")
parser.add_argument('--datapath', default="../data/", help="The data path.")
parser.add_argument("--early_stopping", type=int,
                    default=400, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
parser.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")

# Model parameter
parser.add_argument('--num_workers', type=int, default=0,
                    help='The number of workers.')
parser.add_argument('--batch_size', type=int, default=512,
                    help='Number of batch size to train. Only works when dataset=coauthor_phy')
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
parser.add_argument('--nhiddenlayer', type=int, default=2,
                    help='The number of hidden layers.')
parser.add_argument("--normalization", default="AugRWalk",
                    help="AugRWalk, AugNormAdj, BingGeNormAdj, The normalization on the adj matrix.")
parser.add_argument("--sampling_percent", type=float, default=1.0,
                    help="The percent of the preserve edges. If it equals 1, no sampling is done on adj matrix.")
# parser.add_argument("--baseblock", default="res", help="The base building block (resgcn, densegcn, multigcn, inceptiongcn).")
parser.add_argument("--nbaseblocklayer", type=int, default=0,
                    help="The number of layers in each baseblock")
parser.add_argument("--aggrmethod", default="default",
                    help="The aggrmethod for the layer aggreation. The options includes add and concat. Only valid in resgcn, densegcn and inecptiongcn")
parser.add_argument("--task_type", default="semi",
                    help="The node classification task type (full and semi). Only valid for cora, citeseer and pubmed dataset.")

args = parser.parse_args()

# if args.debug:
print(args)

# log path
# save log path
log_pth = os.path.join(os.getcwd(), 'logs', method_name, args.dataset, str(args.hidden), 
                        'sampling_percent_{}_n_layer_{}'.format(args.sampling_percent, args.nhiddenlayer))
if os.path.exists(log_pth):
    os.system('rm -r {}'.format(log_pth))  # delete old log, the dir will be automatically built later

# model save path
model_save_path = os.path.join(os.getcwd(), 'saved_models', method_name, args.dataset, str(args.hidden), 
                        'sampling_percent_{}_n_layer_{}'.format(args.sampling_percent, args.nhiddenlayer))
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

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
labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

# The model
model = SAGE_Full(nfeat,
                      args.hidden,
                      nclass,
                      args.nhiddenlayer,
                      F.relu,
                      args.dropout,
                      'gcn')

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
# convert to cuda
if args.cuda:
    model.cuda()

# For the mix mode, lables and indexes are in cuda.
# if args.cuda or args.mixmode:
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()

if args.warm_start is not None and args.warm_start != "":
    early_stopping = EarlyStopping(patience=args.early_stopping, fname='best_classifier.model',
                                   save_model_pth=model_save_path)
    print("Restore checkpoint from %s" % (early_stopping.fname))
    model.load_state_dict(early_stopping.load_checkpoint())

# set early_stopping
if args.early_stopping > 0:
    early_stopping = EarlyStopping(patience=args.early_stopping, fname='best_classifier.model',
                                   save_model_pth=model_save_path)
    print("Model is saving to: %s" % (early_stopping.fname))

if args.no_tensorboard is False:
    tb_writer = SummaryWriter(log_dir=log_pth, comment="-dataset_{}-type_{}".format(args.dataset, args.type))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# define the training function.
def train_sampling(epoch, dataloader, train_g, val_g, val_batch_size):
    # if val_adj is None:
    #     val_adj = train_adj
    #     val_fea = train_fea

    t = time.time()


    batch_loss_train = []
    batch_acc_train = []
    
    model.train()
    batch_loss_train_list = []
    batch_acc_train_list = []
    for step, blocks in enumerate(dataloader):
        tic_step = time.time()

        # The nodes for input lies at the LHS side of the first block.
        # The nodes for output lies at the RHS side of the last block.
        input_nodes = blocks[0].srcdata[dgl.NID]
        seeds = blocks[-1].dstdata[dgl.NID]

        # Load the input features as well as output labels
        batch_inputs, batch_labels = load_subtensor(train_g, labels, seeds, input_nodes)
        output = model(blocks, batch_inputs)
        
        # Compute loss and prediction
        # special for coauthor_phy, GraphSage is an inductive model and the training proess is conducted in inductive manner
        loss_train = F.nll_loss(output, labels[seeds])
        acc_train = accuracy(output, labels[seeds])
        # if sampler.learning_type == "inductive":
        #     loss_train = F.nll_loss(output, labels[idx_train].cpu()).cuda()
        #     acc_train = accuracy(output, labels[idx_train].cpu())
        # else:
        #     loss_train = F.nll_loss(output[idx_train.cpu()], labels[idx_train].cpu()).cuda()
        #     acc_train = accuracy(output[idx_train.cpu()], labels[idx_train].cpu())

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        batch_loss_train_list.append(loss_train.item())
        batch_acc_train_list.append(acc_train.item())
    loss_train = np.mean(batch_loss_train_list)
    acc_train = np.mean(batch_acc_train_list)

    train_t = time.time() - t
    val_t = time.time()
    # We can not apply the fastmode for the coauthor_phy dataset.
    # if sampler.learning_type == "inductive" or not args.fastmode:
    model.eval()
    output = model.inference(val_g, val_g.ndata['features'], val_batch_size, 'cpu')

    if args.early_stopping > 0 and sampler.dataset != "coauthor_phy":
        loss_val = F.nll_loss(output[idx_val.cpu()], labels[idx_val].cpu()).item()
        acc_val = accuracy(output[idx_val.cpu()], labels[idx_val].cpu()).item()
        early_stopping(loss_val, model)

    if not args.fastmode:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        loss_val = F.nll_loss(output[idx_val.cpu()], labels[idx_val].cpu()).item()
        acc_val = accuracy(output[idx_val.cpu()], labels[idx_val].cpu()).item()
        if sampler.dataset == "coauthor_phy":
            early_stopping(loss_val, model)
    else:
        loss_val = 0
        acc_val = 0

    if args.lradjust:
        scheduler.step()

    val_t = time.time() - val_t
    return (loss_train, acc_train, loss_val, acc_val, get_lr(optimizer), train_t, val_t)

# define the training function.
def train_full(epoch, train_g, val_g, idx_val, labels):
    # if val_adj is None:
    #     val_adj = train_adj
    #     val_fea = train_fea

    t = time.time()

    batch_loss_train = []
    batch_acc_train = []
    
    model.train()
    output = model(train_g.ndata['features'], train_g)

    if sampler.learning_type=='inductive':
        loss_train = F.nll_loss(output, labels[idx_train])
        acc_train = accuracy(output, labels[idx_train])
    else:
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    train_t = time.time() - t
    val_t = time.time()
    # We can not apply the fastmode for the coauthor_phy dataset.
    # if sampler.learning_type == "inductive" or not args.fastmode:
    model.eval()
    if sampler.dataset in ['coauthor_phy']:
        model.cpu()
        labels = labels.cpu()

    output = model(val_g.ndata['features'], val_g)

    if args.early_stopping > 0 and sampler.dataset != "coauthor_phy":
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        early_stopping(loss_val, model)
    
    if not args.fastmode:
        #    # Evaluate validation set performance separately,
        #    # deactivates dropout during validation run.
        loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
        acc_val = accuracy(output[idx_val], labels[idx_val]).item()
        if sampler.dataset in ['coauthor_phy']:
            early_stopping(loss_val, model)
    else:
        loss_val = 0
        acc_val = 0

    if sampler.dataset in ['coauthor_phy']:
        model.cuda()
        labels = labels.cuda()

    if args.lradjust:
        scheduler.step()

    val_t = time.time() - val_t
    return (loss_train.item(), acc_train.item(), loss_val, acc_val, get_lr(optimizer), train_t, val_t)


def test_sampling(model, test_g, val_batch_size):
    model.eval()
    output = model.inference(test_g, test_g.ndata['features'], val_batch_size, 'cpu')

    loss_test = F.nll_loss(output[idx_test.cpu()], labels[idx_test].cpu())
    acc_test = accuracy(output[idx_test.cpu()], labels[idx_test].cpu())
    auc_test = roc_auc_compute_fn(output[idx_test.cpu()], labels[idx_test].cpu())
    if args.debug:
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "auc= {:.4f}".format(auc_test),
              "accuracy= {:.4f}".format(acc_test.item()))
        print("accuracy=%.5f" % (acc_test.item()))
    return (loss_test.item(), acc_test.item())

def test_full(model, test_g, idx_test, labels):
    model.eval()
    if sampler.dataset in ['coauthor_phy']:
        model.cpu()
        labels = labels.cpu()

    output = model(test_g.ndata['features'], test_g)

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

for epoch in range(args.epochs):
    # input_idx_train = idx_train
    sampling_t = time.time()
    # no sampling
    # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.
    (train_adj, train_fea) = sampler.randomedge_sampler(percent=args.sampling_percent, normalization=args.normalization,
                                                        cuda=args.cuda)

    (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)


    sampling_t = time.time() - sampling_t

    '''
    ####################### Core part for training GraphSage #################################
    '''
    # Construct feed data g
    if torch.cuda.is_available():
        train_edges = train_adj._indices().cpu().data
    else:
        train_edges = train_adj._indices().data
    train_edges = (train_edges[0], train_edges[1])
    train_g = dgl.graph(train_edges)
    train_g.ndata['features'] = train_fea
    prepare_mp(train_g)
    if torch.cuda.is_available():
        val_edges = val_adj._indices().cpu().data
    else:
        val_edges = val_adj._indices().data

    val_edges = (val_edges[0], val_edges[1])
    val_g = dgl.graph(val_edges)
    if sampler.dataset=='coauthor_phy':
        val_g.ndata['features'] = val_fea.cpu()
        idx_val = idx_val.cpu()
    else:
        val_g.ndata['features'] = val_fea
    prepare_mp(val_g)

    
    outputs = train_full(epoch, train_g, val_g, idx_val, labels)


    '''
    ####################### End core part for training GraphSage #################################
    '''
    # # The validation set is controlled by idx_val
    # # if sampler.learning_type == "transductive":
    # if False:
    #     outputs = train(epoch, train_adj, train_fea, input_idx_train)
    # else:
    #     (val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
    #     if args.mixmode:
    #         val_adj = val_adj.cuda()
    #     outputs = train(epoch, train_adj, train_fea, input_idx_train, val_adj, val_fea)

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
        break

if args.early_stopping > 0:
    model.load_state_dict(early_stopping.load_checkpoint())

if args.debug:
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
if torch.cuda.is_available():
    test_edges = test_adj._indices().cpu().data
else:
    test_edges = test_adj._indices().data
test_edges = (test_edges[0], test_edges[1])
test_g = dgl.graph(test_edges)

if sampler.dataset=='coauthor_phy':
    test_g.ndata['features'] = test_fea.cpu()
    idx_test = idx_test.cpu()
else:
    test_g.ndata['features'] = test_fea
prepare_mp(test_g)


(loss_test, acc_test) = test_full(model, test_g, idx_test, labels)

print("best epoch: {}\t best val loss: {:.6f}\t test loss: {:.6f}\t test_acc: {:.6f}".format(np.argmin(loss_val),
                                                                                             -early_stopping.best_score,
                                                                                             loss_test, acc_test))

print(args)
