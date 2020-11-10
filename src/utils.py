import pickle as pkl
import sys
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import json
from sklearn.utils import shuffle
from normalization import fetch_normalization, row_normalize

datadir = "data"

class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.
    """
    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):
        """Create an attributed graph.
        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.
        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.
        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.
        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.
        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.
        All changes are done inplace.
        """
        G = self.to_unweighted().to_undirected()
        G = eliminate_self_loops(G)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels

def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def load_citation(dataset_str="cora", normalization="AugNormAdj", porting_to_torch=True,data_path=datadir, task_type="full"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str.lower(), names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # degree = np.asarray(G.degree)
    degree = np.sum(adj, axis=1)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    if task_type == "full":
        print("Load full supervised task.")
        #supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(ally)- 500)
        idx_val = range(len(ally) - 500, len(ally))
    elif task_type == "semi":
        print("Load semi-supervised task.")
        #semi-supervised setting
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    elif task_type == 'self-supervised':
        print("Load self-supervised task.")
        # self-supervised leanring makes the following idx_test, idx_train, idx_val no use
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)
    else:
        raise ValueError("Task type: %s is not supported. Available option: full and semi.")

    adj, features = preprocess_citation(adj, features, normalization)
    features = np.array(features.todense())
    labels = np.argmax(labels, axis=1)
    # porting to pytorch
    if porting_to_torch:
        features = torch.FloatTensor(features).float()
        labels = torch.LongTensor(labels)
        # labels = torch.max(labels, dim=1)[1]
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        degree = torch.LongTensor(degree)
    learning_type = "transductive"
    return adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type

def sgc_precompute(features, adj, degree):
    #t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = 0 #perf_counter()-t
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def loadRedditFromNPZ(dataset_dir=datadir):
    adj = sp.load_npz(dataset_dir+"reddit/reddit_adj.npz")
    data = np.load(dataset_dir +"reddit/reddit_data.npz")
    nodes_types = data['node_types']
    train_index = np.where(nodes_types==1)[0]
    val_index = np.where(nodes_types==2)[0]
    test_index = np.where(nodes_types==3)[0]
    labels = data['label']

    return adj.tocsr(), data['feature'], labels, train_index, val_index, test_index


def load_reddit_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    adj, features, labels, train_index, val_index, test_index = loadRedditFromNPZ(data_path)
    # labels = np.zeros(adj.shape[0])
    # labels[train_index]  = y_train
    # labels[val_index]  = y_val
    # labels[test_index]  = y_test
    # the loaded data is symetric and with self-loop, so we do not need the following pre-process
    # adj = adj + adj.T + sp.eye(adj.shape[0])
    
    # sampling small set of train index because of our limited computation resources
    train_index = shuffle(train_index, random_state=42)[:int(0.1*len(train_index))]
    train_adj = adj[train_index, :][:, train_index]
    degree = np.sum(train_adj, axis=1)

    features = torch.FloatTensor(np.array(features))
    # features[:, :2] = features[:, :2]/torch.max(features[:, :2], dim=0, keepdim=True)[0]
    # features = features/torch.max(features, dim=0, keepdim=True)[0]
    features = (features-features.mean(dim=0))/features.std(dim=0)
    train_features = torch.index_select(features, 0, torch.LongTensor(train_index))
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "inductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type

def load_facebook_page_data(normalization, porting_to_torch, data_path=None):
    edges = pd.read_csv(os.path.join(data_path, 'facebook_page', 'musae_facebook_edges.csv'), header=0, sep=',')
    raw_feats = json.load(open(os.path.join(data_path, 'facebook_page', 'musae_facebook_features.json'), 'r'))
    # make adj
    adj = sp.coo_matrix((np.ones(len(edges)), (edges.values[:, 0], edges.values[:, 1])), shape=[len(raw_feats), len(raw_feats)])
    adj = adj.tocsr()
    adj = adj+adj.T+sp.eye(adj.shape[0])
    adj.data = np.ones_like(adj.data)

    adj = adj + adj.T + sp.eye(adj.shape[0])

    train_adj = adj  # transductive setting
    degree = np.sum(train_adj, axis=1)

    # make features
    feat_set = set()
    for k in raw_feats:
        feat_set = feat_set | set(raw_feats[k])
    feat_dim = len(list(feat_set))
    features = np.zeros(shape=[adj.shape[0], feat_dim])
    for k in raw_feats:
        features[int(k), :][raw_feats[k]] = 1.0

    # make labels
    raw_label_data = pd.read_csv(os.path.join(data_path, 'facebook_page', 'musae_facebook_target.csv'), header=0, sep=',')
    raw_labels = raw_label_data['page_type'].unique()
    label_map = pd.Series(data=range(len(raw_labels)), index=raw_labels)
    raw_label_data['label'] = label_map[raw_label_data['page_type'].values].values
    labels = raw_label_data['label'].values

    # split data
    if not os.path.exists(os.path.join(data_path, 'facebook_page', 'train_index.npy')):
        print('make split data.......')
        train_index = []
        val_index = []
        test_index = []
        for l in range(labels.max()+1):
            tmp_index = np.where(labels==l)[0]
            tmp_index = shuffle(tmp_index, random_state=42)
            tmp_train = tmp_index[:20]
            tmp_val = tmp_index[20:50]
            tmp_test = tmp_index[50:]

            train_index.append(tmp_train)
            val_index.append(tmp_val)
            test_index.append(tmp_test)
        train_index = shuffle(np.concatenate(train_index), random_state=42)
        val_index = shuffle(np.concatenate(val_index), random_state=42)
        test_index = shuffle(np.concatenate(test_index), random_state=42)

        np.save(os.path.join(data_path, 'facebook_page', 'train_index.npy'), train_index)
        np.save(os.path.join(data_path, 'facebook_page', 'val_index.npy'), val_index)
        np.save(os.path.join(data_path, 'facebook_page', 'test_index.npy'), test_index)
    else:
        print('load split data......')
        train_index = np.load(os.path.join(data_path, 'facebook_page', 'train_index.npy'))
        val_index = np.load(os.path.join(data_path, 'facebook_page', 'val_index.npy'))
        test_index = np.load(os.path.join(data_path, 'facebook_page', 'test_index.npy'))

    # process data
    features = torch.FloatTensor(features)
    features = features/torch.sum(features, dim=1, keepdim=True)
    train_features = features

    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "transductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type

def load_coauthor_cs_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    data = load_npz_to_sparse_graph(os.path.join(data_path, 'coauthor_cs', 'ms_academic_cs.npz'))

    # make adj
    adj = data.adj_matrix
    adj = adj + adj.T + sp.eye(adj.shape[0])
    adj.data = np.ones_like(adj.data)
    train_adj = adj
    degree = np.sum(train_adj, axis=1)

    # make features
    features = data.attr_matrix.todense()

    features = torch.FloatTensor(features)
    features = features/torch.sum(features, dim=1, keepdim=True)
    train_features = features  # transductive setting
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    # make labels
    labels = data.labels
    # split data
    # split data
    if not os.path.exists(os.path.join(data_path, 'coauthor_cs', 'train_index.npy')):
        print('make split data......')
        train_index = []
        val_index = []
        test_index = []
        for l in range(labels.max()+1):
            tmp_index = np.where(labels==l)[0]
            tmp_index = shuffle(tmp_index, random_state=42)
            tmp_train = tmp_index[:20]
            tmp_val = tmp_index[20:50]
            tmp_test = tmp_index[50:]

            train_index.append(tmp_train)
            val_index.append(tmp_val)
            test_index.append(tmp_test)
        train_index = shuffle(np.concatenate(train_index), random_state=42)
        val_index = shuffle(np.concatenate(val_index), random_state=42)
        test_index = shuffle(np.concatenate(test_index), random_state=42)

        np.save(os.path.join(data_path, 'coauthor_cs', 'train_index.npy'), train_index)
        np.save(os.path.join(data_path, 'coauthor_cs', 'val_index.npy'), val_index)
        np.save(os.path.join(data_path, 'coauthor_cs', 'test_index.npy'), test_index)
    else:
        print('load split data......')
        train_index = np.load(os.path.join(data_path, 'coauthor_cs', 'train_index.npy'))
        val_index = np.load(os.path.join(data_path, 'coauthor_cs', 'val_index.npy'))
        test_index = np.load(os.path.join(data_path, 'coauthor_cs', 'test_index.npy'))

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "transductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type

def load_coauthor_phy_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    data = load_npz_to_sparse_graph(os.path.join(data_path, 'coauthor_phy', 'ms_academic_phy.npz'))

    # make labels
    labels = data.labels

    # make adj
    adj = data.adj_matrix
    adj = adj + adj.T + sp.eye(adj.shape[0])
    adj.data = np.ones_like(adj.data)

    all_index = shuffle(np.arange(len(labels)), random_state=42)
    train_index = all_index[:20000]
    val_index = all_index[20000:25000]
    test_index = all_index[25000:]

    train_adj = adj[train_index, :][:, train_index]
    degree = np.sum(train_adj, axis=1)

    # make features
    features = data.attr_matrix.todense()

    features = torch.FloatTensor(np.array(features))
    features = features/torch.sum(features, dim=1, keepdim=True)
    train_features = torch.index_select(features, 0, torch.LongTensor(train_index))
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "inductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type

def load_cora_full_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    data = load_npz_to_sparse_graph(os.path.join(data_path, 'cora_full', 'cora_full.npz'))

    # delete labeled nodes less than 50
    adj = data.adj_matrix
    features = data.attr_matrix.todense()
    labels = data.labels

    mask = []
    count_dict = {}
    for l in labels:
        tmp_index = np.where(labels == l)[0]
        if l not in count_dict:
            count_dict[l] = len(tmp_index)

        if len(tmp_index) > 55:
            mask.append(True)
        else:
            mask.append(False)
    mask = np.array(mask)

    adj = adj[mask, :][:, mask]
    features = features[mask]
    labels = labels[mask]

    # re-assign labels
    label_map = pd.Series(index=np.unique(labels), data=np.arange(len(np.unique(labels))))
    labels = label_map[labels].values

    # make adj
    adj = adj + adj.T + sp.eye(adj.shape[0])
    adj.data = np.ones_like(adj.data)
    train_adj = adj
    degree = np.sum(train_adj, axis=1)


    # make features
    features = torch.FloatTensor(features)
    # features = features / torch.sum(features, dim=1, keepdim=True)
    train_features = features  # transductive setting
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    # split data
    if not os.path.exists(os.path.join(data_path, 'cora_full', 'train_index.npy')):
        print('make split data......')
        train_index = []
        val_index = []
        test_index = []
        for l in range(labels.max()+1):
            tmp_index = np.where(labels==l)[0]
            tmp_index = shuffle(tmp_index, random_state=42)
            tmp_train = tmp_index[:20]
            tmp_val = tmp_index[20:50]
            tmp_test = tmp_index[50:]

            train_index.append(tmp_train)
            val_index.append(tmp_val)
            test_index.append(tmp_test)
        train_index = shuffle(np.concatenate(train_index), random_state=42)
        val_index = shuffle(np.concatenate(val_index), random_state=42)
        test_index = shuffle(np.concatenate(test_index), random_state=42)

        np.save(os.path.join(data_path, 'cora_full', 'train_index.npy'), train_index)
        np.save(os.path.join(data_path, 'cora_full', 'val_index.npy'), val_index)
        np.save(os.path.join(data_path, 'cora_full', 'test_index.npy'), test_index)
    else:
        print('load split data......')
        train_index = np.load(os.path.join(data_path, 'cora_full', 'train_index.npy'))
        val_index = np.load(os.path.join(data_path, 'cora_full', 'val_index.npy'))
        test_index = np.load(os.path.join(data_path, 'cora_full', 'test_index.npy'))


    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "transductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type

def load_amazon_computer_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    data = load_npz_to_sparse_graph(os.path.join(data_path, 'amazon_computer', 'amazon_electronics_computers.npz'))

    # make adj
    adj = data.adj_matrix
    adj = adj + adj.T + sp.eye(adj.shape[0])
    adj.data = np.ones_like(adj.data)
    train_adj = adj
    degree = np.sum(train_adj, axis=1)

    # make features
    features = data.attr_matrix.todense()

    features = torch.FloatTensor(features)
    features = features/torch.sum(features, dim=1, keepdim=True)
    train_features = features  # transductive setting
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    # make labels
    labels = data.labels
    # split data
    if not os.path.exists(os.path.join(data_path, 'amazon_computer', 'train_index.npy')):
        print('make split data......')
        train_index = []
        val_index = []
        test_index = []
        for l in range(labels.max()+1):
            tmp_index = np.where(labels==l)[0]
            tmp_index = shuffle(tmp_index, random_state=42)
            tmp_train = tmp_index[:20]
            tmp_val = tmp_index[20:50]
            tmp_test = tmp_index[50:]

            train_index.append(tmp_train)
            val_index.append(tmp_val)
            test_index.append(tmp_test)
        train_index = shuffle(np.concatenate(train_index), random_state=42)
        val_index = shuffle(np.concatenate(val_index), random_state=42)
        test_index = shuffle(np.concatenate(test_index), random_state=42)

        np.save(os.path.join(data_path, 'amazon_computer', 'train_index.npy'), train_index)
        np.save(os.path.join(data_path, 'amazon_computer', 'val_index.npy'), val_index)
        np.save(os.path.join(data_path, 'amazon_computer', 'test_index.npy'), test_index)
    else:
        print('load split data......')
        train_index = np.load(os.path.join(data_path, 'amazon_computer', 'train_index.npy'))
        val_index = np.load(os.path.join(data_path, 'amazon_computer', 'val_index.npy'))
        test_index = np.load(os.path.join(data_path, 'amazon_computer', 'test_index.npy'))


    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "transductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type

def load_amazon_photo_data(normalization="AugNormAdj", porting_to_torch=True, data_path=datadir):
    data = load_npz_to_sparse_graph(os.path.join(data_path, 'amazon_photo', 'amazon_electronics_photo.npz'))

    # make adj
    adj = data.adj_matrix
    adj = adj + adj.T + sp.eye(adj.shape[0])
    adj.data = np.ones_like(adj.data)
    train_adj = adj
    degree = np.sum(train_adj, axis=1)

    # make features
    features = data.attr_matrix.todense()

    features = torch.FloatTensor(features)
    features = features/torch.sum(features, dim=1, keepdim=True)
    train_features = features  # transductive setting
    if not porting_to_torch:
        features = features.numpy()
        train_features = train_features.numpy()

    # make labels
    labels = data.labels
    # split data
    if not os.path.exists(os.path.join(data_path, 'amazon_photo', 'train_index.npy')):
        print('make split data......')
        train_index = []
        val_index = []
        test_index = []
        for l in range(labels.max()+1):
            tmp_index = np.where(labels==l)[0]
            tmp_index = shuffle(tmp_index, random_state=42)
            tmp_train = tmp_index[:20]
            tmp_val = tmp_index[20:50]
            tmp_test = tmp_index[50:]

            train_index.append(tmp_train)
            val_index.append(tmp_val)
            test_index.append(tmp_test)
        train_index = shuffle(np.concatenate(train_index), random_state=42)
        val_index = shuffle(np.concatenate(val_index), random_state=42)
        test_index = shuffle(np.concatenate(test_index), random_state=42)

        np.save(os.path.join(data_path, 'amazon_photo', 'train_index.npy'), train_index)
        np.save(os.path.join(data_path, 'amazon_photo', 'val_index.npy'), val_index)
        np.save(os.path.join(data_path, 'amazon_photo', 'test_index.npy'), test_index)
    else:
        print('load split data......')
        train_index = np.load(os.path.join(data_path, 'amazon_photo', 'train_index.npy'))
        val_index = np.load(os.path.join(data_path, 'amazon_photo', 'val_index.npy'))
        test_index = np.load(os.path.join(data_path, 'amazon_photo', 'test_index.npy'))

    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    train_adj = adj_normalizer(train_adj)

    if porting_to_torch:
        train_adj = sparse_mx_to_torch_sparse_tensor(train_adj).float()
        labels = torch.LongTensor(labels)
        adj = sparse_mx_to_torch_sparse_tensor(adj).float()
        degree = torch.LongTensor(degree)
        train_index = torch.LongTensor(train_index)
        val_index = torch.LongTensor(val_index)
        test_index = torch.LongTensor(test_index)
    learning_type = "transductive"
    return adj, train_adj, features, train_features, labels, train_index, val_index, test_index, degree, learning_type

    
def data_loader(dataset, data_path=datadir, normalization="AugNormAdj", porting_to_torch=True, task_type = "full"):
    if dataset == "reddit":
        return load_reddit_data(normalization, porting_to_torch, data_path)
    elif dataset == 'facebook_page':
        return load_facebook_page_data(normalization, porting_to_torch, data_path)
    elif dataset == 'coauthor_cs':
        return load_coauthor_cs_data(normalization, porting_to_torch, data_path)
    elif dataset == 'coauthor_phy':
        return load_coauthor_phy_data(normalization, porting_to_torch, data_path)
    elif dataset == 'cora_full':
        return load_cora_full_data(normalization, porting_to_torch, data_path)
    elif dataset == 'amazon_computer':
        return load_amazon_computer_data(normalization, porting_to_torch, data_path)
    elif dataset == 'amazon_photo':
        return load_amazon_photo_data(normalization, porting_to_torch, data_path)
    else:
        # pubmed data is loaded here
        (adj,
         features,
         labels,
         idx_train,
         idx_val,
         idx_test,
         degree,
         learning_type) = load_citation(dataset, normalization, porting_to_torch, data_path, task_type)
        train_adj = adj
        train_features = features
        return adj, train_adj, features, train_features, labels, idx_train, idx_val, idx_test, degree, learning_type

