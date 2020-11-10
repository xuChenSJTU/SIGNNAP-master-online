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
from models import *
from earlystopping import EarlyStopping
from sample import Sampler

import networkx as nx
import scipy.sparse as sp
from dgl import DGLGraph
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv

import torch
import torch.nn as nn
import math
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # ltrue ayers = n_layers+1
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, g):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.conv = GCN(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features, g, corrupt=False):
        if corrupt:
            perm = torch.randperm(g.number_of_nodes())
            features = features[perm]
        features = self.conv(features, g)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class DGI(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout):
        super(DGI, self).__init__()
        self.encoder = Encoder(in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, features, g):
        positive = self.encoder(features, g, corrupt=False)
        negative = self.encoder(features, g, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2


class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        for m in self.modules():
                self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
method_name = 'DGI'

seed = np.random.randint(2020)

train_flag = True

# Training settings
parser = argparse.ArgumentParser()
# Training parameter
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Disable validation during training.')
parser.add_argument('--seed', type=int, default=seed, help='Random seed. default=42')
parser.add_argument('--dgi-epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--classifier-epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--dgi_lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--classifier_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--lradjust', action='store_true',
                    default=False, help='Enable leraning rate adjust.(ReduceLROnPlateau or Linear Reduce)')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument("--mixmode", action="store_true",
                    default=False, help="Enable CPU GPU mixing mode.")
parser.add_argument("--warm_start", default="",
                    help="The model name to be loaded for warm start.")
parser.add_argument('--debug', action='store_true',
                    default=False, help="Enable the detialed training output.")
parser.add_argument('--dataset', default="facebook_page", help="The data set. pubmed, facebook_page, coauthor_cs")
parser.add_argument('--datapath', default="../data/", help="The data path.")
parser.add_argument("--early_stopping", type=int,
                    default=400, help="The patience of earlystopping. Do not adopt the earlystopping when it equals 0.")
parser.add_argument("--no_tensorboard", default=False, help="Disable writing logs to tensorboard")
parser.add_argument("--run_num", type=int,
                    default=0, help="The num th of run.")

# Model parameter

parser.add_argument('--type', default='multigcn',
                    help="[does not matter] Choose the model to be trained.(multigcn, resgcn, densegcn, inceptiongcn)")
parser.add_argument('--inputlayer', default='gcn',
                    help="The input layer of the model.")
parser.add_argument('--outputlayer', default='gcn',
                    help="The output layer of the model.")
parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--withbn', default=False,
                    help='Enable Bath Norm GCN')
parser.add_argument('--withloop', default=True,
                    help="Enable loop layer GCN")
parser.add_argument('--nhiddenlayer', type=int, default=1,
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
log_pth = os.path.join(os.getcwd(), 'logs', method_name, args.dataset, str(args.hidden), 'n_layer_{}'.format(args.nhiddenlayer+1))
if os.path.exists(log_pth):
    os.system('rm -r {}'.format(log_pth))  # delete old log, the dir will be automatically built later

# model save path
model_save_path = os.path.join(os.getcwd(), 'saved_models', method_name, args.dataset, str(args.hidden), 'n_layer_{}'.format(args.nhiddenlayer+1))
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# path for plot data
embed_save_path = os.path.join(os.getcwd(), 'saved_embeds', method_name, args.dataset)
if not os.path.exists(embed_save_path):
    os.makedirs(embed_save_path)

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
labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(args.cuda)
nfeat = sampler.nfeat
nclass = sampler.nclass
print("nclass: %d\tnfea:%d" % (nclass, nfeat))

# The model
# create model
# create DGI model
model = DGI(nfeat,
              args.hidden,
              args.nhiddenlayer,
              nn.PReLU(args.hidden),
              args.dropout)

optimizer = optim.Adam(model.parameters(),
                       lr=args.dgi_lr, weight_decay=args.weight_decay)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.618)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 400, 500, 600, 700], gamma=0.5)
# convert to cuda
if args.cuda:
    model.cuda()

# For the mix mode, lables and indexes are in cuda.
if args.cuda or args.mixmode:
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

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
def train(epoch, train_adj, train_fea):

    t = time.time()
    model.train()
    optimizer.zero_grad()

    # construct g from train adj
    train_edges = train_adj._indices().data.cpu().numpy()
    train_edges = sp.coo_matrix((np.ones(train_edges.shape[1]),
                                 (train_edges[0], train_edges[1])),
                                shape=(train_adj.shape[0], train_adj.shape[0]),
                                dtype=np.float32)

    train_g = nx.from_scipy_sparse_matrix(train_edges, create_using=nx.DiGraph())
    train_g = DGLGraph(train_g)

    loss_train = model(train_fea, train_g)
   

    loss_train.backward()
    optimizer.step()
    train_t = time.time() - t
    
    if args.lradjust:
        scheduler.step()
        
    return (loss_train.item(), get_lr(optimizer), train_t)

if train_flag:
    # Train model
    t_total = time.time()

    sampling_t = 0

    cnt_wait = 0
    best = 1e9
    best_t = 0
    for epoch in range(args.dgi_epochs):
        input_idx_train = idx_train
        sampling_t = time.time()
        # no sampling
        # randomedge sampling if args.sampling_percent >= 1.0, it behaves the same as stub_sampler.
        (train_adj, train_fea) = sampler.randomedge_sampler(percent=1.0, normalization=args.normalization,
                                                            cuda=args.cuda)

        sampling_t = time.time() - sampling_t

        outputs = train(epoch, train_adj, train_fea)

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
                'cur_lr: {:.5f}'.format(outputs[1]),
                't_time: {:.4f}s'.format(outputs[2]))

        if args.no_tensorboard is False:
            tb_writer.add_scalars('Loss', {'train': outputs[0]}, epoch)
            tb_writer.add_scalar('lr', outputs[1], epoch)
            tb_writer.add_scalars('Time', {'train': outputs[2]}, epoch)

        loss_train = outputs[0]

        # early stop
        if loss_train < best:
            best = loss_train
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_dgi.pkl'))
        else:
            cnt_wait += 1

        if cnt_wait == 20:
            print('Early stopping!')
            break
    print('Loading {}th epoch'.format(best_t))


#################################  Test Process ###################################
# create classifier model
classifier = Classifier(args.hidden, nclass)
classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)
if args.cuda:
    classifier.cuda()


# construct g from train adj
(train_adj, train_fea) = sampler.randomedge_sampler(percent=1.0, normalization=args.normalization,
                                                        cuda=args.cuda)
train_edges = train_adj._indices().data.cpu().numpy()
train_edges = sp.coo_matrix((np.ones(train_edges.shape[1]),
                                 (train_edges[0], train_edges[1])),
                                shape=(train_adj.shape[0], train_adj.shape[0]),
                                dtype=np.float32)

train_g = nx.from_scipy_sparse_matrix(train_edges, create_using=nx.DiGraph())
train_g = DGLGraph(train_g)

(val_adj, val_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
val_edges = val_adj._indices().data.cpu().numpy()
val_edges = sp.coo_matrix((np.ones(val_edges.shape[1]),
                                 (val_edges[0], val_edges[1])),
                                shape=(val_adj.shape[0], val_adj.shape[0]),
                                dtype=np.float32)

val_g = nx.from_scipy_sparse_matrix(val_edges, create_using=nx.DiGraph())
val_g = DGLGraph(val_g)


model.load_state_dict(torch.load(os.path.join(model_save_path, 'best_dgi.pkl')))

dur = []
best = 1e9
cnt_wait = 0
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(args.classifier_epochs):
    classifier.train()
    if epoch >= 3:
        t0 = time.time()

    embeds = model.encoder(train_fea, train_g, corrupt=False)
    embeds = embeds.detach()

    classifier_optimizer.zero_grad()
    preds = classifier(embeds)

    if sampler.dataset=='coauthor_phy':
        loss = F.nll_loss(preds, labels[idx_train])
        logits = classifier(embeds)
    else:
        loss = F.nll_loss(preds[idx_train], labels[idx_train])
        logits = classifier(embeds)[idx_train]

    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels[idx_train])
    train_acc = correct.item() * 1.0 / len(labels[idx_train])


    loss.backward()
    classifier_optimizer.step()
    train_loss_list.append(loss.item())
    train_acc_list.append(train_acc)

    if epoch >= 3:
        dur.append(time.time() - t0)

    embeds = model.encoder(val_fea, val_g, corrupt=False)
    embeds = embeds.detach()
    preds = classifier(embeds)
    val_loss = F.nll_loss(preds[idx_val], labels[idx_val])
    val_acc = evaluate(classifier, embeds, labels, idx_val)
    print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f}".format(epoch, np.mean(dur),
                            loss.item(), train_acc, val_loss.item(), val_acc))

    val_loss_list.append(val_loss.item())
    val_acc_list.append(val_acc)

    # early stop
    if val_loss < best:
        best = val_loss
        best_t = epoch
        cnt_wait = 0
        torch.save(classifier.state_dict(), os.path.join(model_save_path, 'best_classifier.pkl'))
    else:
        cnt_wait += 1

    if cnt_wait == 400:
        print('Early stopping!')
        break


# Testing
classifier.load_state_dict(torch.load(os.path.join(model_save_path, 'best_classifier.pkl')))

(test_adj, test_fea) = sampler.get_test_set(normalization=args.normalization, cuda=args.cuda)
test_edges = test_adj._indices().data.cpu().numpy()
test_edges = sp.coo_matrix((np.ones(test_edges.shape[1]),
                                 (test_edges[0], test_edges[1])),
                                shape=(test_adj.shape[0], test_adj.shape[0]),
                                dtype=np.float32)

test_g = nx.from_scipy_sparse_matrix(test_edges, create_using=nx.DiGraph())
test_g = DGLGraph(test_g)

embeds = model.encoder(test_fea, test_g, corrupt=False)
embeds = embeds.detach()

##############  save plot data #######################
# # save embeds
# np.save(os.path.join(embed_save_path, 'embeds.npy'), embeds.cpu().data.numpy())
# np.save(os.path.join(embed_save_path, 'original_feats.npy'), test_fea.cpu().data.numpy())

# ####################### save plot continuous data ##########################
# (train_adj, train_fea) = sampler.randomedge_sampler(percent=args.sampling_percent, normalization=args.normalization,
#                                                             cuda=args.cuda)
# train_edges = train_adj._indices().data.cpu().numpy()
# train_edges = sp.coo_matrix((np.ones(train_edges.shape[1]),
#                                  (train_edges[0], train_edges[1])),
#                                 shape=(train_adj.shape[0], train_adj.shape[0]),
#                                 dtype=np.float32)
#
# train_g = nx.from_scipy_sparse_matrix(train_edges, create_using=nx.DiGraph())
# train_g = DGLGraph(train_g)
#
# embeds = model.encoder(train_fea, train_g, corrupt=False)
# embeds = classifier(embeds).cpu().data.numpy()
#
# np.save(os.path.join(embed_save_path, 'feats_sampling_{}_{}.npy'.format(args.sampling_percent, args.run_num)), embeds)
# np.save(os.path.join(embed_save_path, 'test_idx.npy'), idx_test.data.cpu().numpy())


# # save training data
# np.save(os.path.join(training_save_path, 'train_loss_n3.npy'), np.array(train_loss_list))
# np.save(os.path.join(training_save_path, 'val_loss_n3.npy'), np.array(val_loss_list))
# np.save(os.path.join(training_save_path, 'train_acc_n3.npy'), np.array(train_acc_list))
# np.save(os.path.join(training_save_path, 'val_acc_n3.npy'), np.array(val_acc_list))


preds = classifier(embeds)
test_loss = F.nll_loss(preds[idx_test], labels[idx_test])
test_acc = evaluate(classifier, embeds, labels, idx_test)
best_epoch = np.argmin(val_loss_list)
print("Best epoch: {}, Val loss: {:.4f}, Test loss: {:.4f}, Test acc {:.4f}".format(best_epoch,  val_loss_list[best_epoch], test_loss.item(),
                                                                                                                                test_acc))
print(args)
