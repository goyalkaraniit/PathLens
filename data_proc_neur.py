import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor, CoraFull, HeterophilousGraphDataset
from torch_geometric.utils import num_nodes
from cSBM_dataset import dataset_ContextualSBM
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import NodePropPredDataset
from Load import *
from sklearn.preprocessing import label_binarize


from collections import defaultdict
import torch.nn.functional as F
import scipy
import scipy.io
import pickle
import pandas as pd
from sklearn.preprocessing import label_binarize
import gdown
from torch_geometric.transforms import NormalizeFeatures
from torch_sparse import SparseTensor
import os

def generate_split(data, num_classes, seed=2021, train_num_per_c=20, val_num_per_c=30):
    np.random.seed(seed)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = np.random.permutation(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask

    return train_mask, val_mask, test_mask

def generate_split_neur(data, num_classes, seed=0, train_num_per_c=20, val_num_per_c=30):
    np.random.seed(seed)
    train_mask = torch.zeros(data.y.size(), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        train_num_per_c = int(0.6*all_c_idx.size(0))
        val_num_per_c = int(0.2*all_c_idx.size(0))
        """
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            print(test_mask)
            continue
        """
        perm = np.random.permutation(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask

    return train_mask, val_mask, test_mask

def generate_random_edges(data, random_rate=None, seed=2021):
    np.random.seed(seed)
    n_nodes = num_nodes.maybe_num_nodes(data.edge_index)
    if random_rate == 0:
        return data.edge_index
    elif random_rate == 1:
        num_new_edges = len(data.edge_index.T)
        rd_edge_index_1 = np.random.randint(n_nodes-1, size=(2, num_new_edges))
        rd_edge_index_2 = np.random.randint(n_nodes-1, size=(2, num_new_edges))
        new_edge_index = np.concatenate([rd_edge_index_1.T, rd_edge_index_2.T])
        new_edge_index = list(set([tuple(e_index) for e_index in new_edge_index]))
        new_edge_index = [list(v) for v in new_edge_index]
        new_edge_index = new_edge_index[:num_new_edges]
        new_edge_index.sort()
        new_edge_index = torch.LongTensor(new_edge_index)
        return new_edge_index.T
    else:
        num_new_edges = int(random_rate * len(data.edge_index.T))
        rd_edge_index = np.random.randint(n_nodes-1, size=(2, num_new_edges))
        old_edge_index = data.edge_index.numpy().T
        rm_id = np.random.choice(len(data.edge_index.T)-1, num_new_edges)
        old_edge_index = np.delete(old_edge_index, rm_id, 0)

        new_edge_index = np.concatenate([old_edge_index, rd_edge_index.T])
        new_edge_index = list(set([tuple(e_index) for e_index in new_edge_index]))
        new_edge_index = [list(v) for v in new_edge_index]
        new_edge_index.sort()
        
        new_edge_index = torch.LongTensor(new_edge_index)
        return new_edge_index.T

def load_data(args, root='dataset', rand_seed=0):
    dataset = args.input
    path = osp.join(root, dataset)
    dataset = dataset.lower()

    if 'csbm' in dataset:
        dataset = dataset_ContextualSBM(root, 
                                        name=dataset, 
                                        train_percent=args.train_rate, 
                                        val_percent=args.val_rate)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        return data, num_features, num_classes
    elif dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset)
        data = dataset[0]
    elif dataset in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, dataset)
        data = dataset[0]
    elif dataset == 'actor':
        dataset = Actor(path)
        data = dataset[0]
    elif dataset in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(path, dataset)
        data = dataset[0]
    elif dataset in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(path, dataset)
        data = dataset[0]
    elif dataset in ['chameleon_filtered', 'squirrel_filtered']:
        dataset = load(dataset)
        data = dataset
    elif dataset in ['computers', 'photo']:
        dataset = Amazon(path, dataset)
        data = dataset[0]
    
    elif dataset in ['cs', 'physics']:
        dataset = Coauthor(path, dataset)
        data = dataset[0]
    elif dataset in ['arxiv', 'mag', 'products']:
        dataset = PygNodePropPredDataset(name = 'ogbn-'+dataset)
        data = dataset[0]

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    #data = dataset[0]

    num_train = int(len(data.y) / num_classes * args.train_rate)
    num_val = int(len(data.y) / num_classes * args.val_rate)
    data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, rand_seed, num_train, num_val)

    return data, num_features, num_classes

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx

class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        
        Usage after construction: 
        
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        
        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/
        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def load_nc_dataset(dataname, sub_dataname=''):
    """ Loader for NCDataset, returns NCDataset. """
    if dataname == 'twitch':
        # twitch-explicit graph
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'ENGB'
        dataset = load_twitch_dataset(sub_dataname)
    elif dataname == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(sub_dataname)
    elif dataname == 'ogbn-proteins':
        dataset = load_proteins_dataset()
    elif dataname == 'deezer-europe':
        dataset = load_deezer_dataset()
    elif dataname == 'arxiv-year':
        dataset = load_arxiv_year_dataset()
    elif dataname == 'pokec':
        dataset = load_pokec_mat()
    elif dataname == 'snap-patents':
        dataset = load_snap_patents_mat()
    elif dataname == 'yelp-chi':
        dataset = load_yelpchi_dataset()
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(dataname)
    elif dataname == "genius":
        dataset = load_genius()
    elif dataname == "twitch-gamer":
        dataset = load_twitch_gamer_dataset() 
    else:
        raise ValueError('Invalid dataname')
    return dataset



def load_twitch_dataset(lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'), 'Invalid dataset'
    A, label, features = load_twitch(lang)
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_fb100_dataset(filename):
    A, metadata = load_fb100(filename)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack(
        (np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset

def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_arxiv_year_dataset(nclass=5):
    filename = 'arxiv-year'
    dataset = NCDataset(filename)
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    label = even_quantile_labels(
        dataset.graph['node_year'].flatten(), nclass, verbose=False)
    dataset.label = torch.as_tensor(label).reshape(-1, 1)
    return dataset


def load_proteins_dataset():
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins')
    dataset = NCDataset('ogbn-proteins')

    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}

    dataset.get_idx_split = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels

    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)
    return dataset


def load_ogb_dataset(name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name)
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx

    dataset.get_idx_split = ogb_idx_to_tensor  # ogb_dataset.get_idx_split
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


def load_pokec_mat():
    """ requires pokec.mat
    """
    if not osp.exists(f'{DATAPATH}pokec.mat'):
        dataset_drive_url  = '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y'
        gdown.download(id=dataset_drive_url, \
            output=f'{DATAPATH}pokec.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_snap_patents_mat(nclass=5):
    dataset_drive_url  = '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia'
    if not osp.exists(f'{DATAPATH}snap_patents.mat'):
        print(f"Snap patents url: {dataset_drive_url}")
        gdown.download(id=dataset_drive_url, \
            output=f'{DATAPATH}snap_patents.mat', quiet=False)

    fulldata = scipy.io.loadmat(f'{DATAPATH}snap_patents.mat')

    dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(
        fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_yelpchi_dataset():
    dataset_drive_url  = '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ'
    if not osp.exists(f'{DATAPATH}YelpChi.mat'):
        gdown.download(id=dataset_drive_url, \
            output=f'{DATAPATH}YelpChi.mat', quiet=False)
    fulldata = scipy.io.loadmat(f'{DATAPATH}YelpChi.mat')
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int).flatten()
    num_nodes = node_feat.shape[0]

    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    label = torch.tensor(label, dtype=torch.long)
    dataset.label = label
    return dataset


def load_genius():
    filename = 'genius'
    dataset = NCDataset(filename)
    fulldata = scipy.io.loadmat(f'dataset/genius.mat')

    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'], dtype=torch.float)
    label = torch.tensor(fulldata['label'], dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_twitch_gamer_dataset(task="mature", normalize=True):
    dataset_drive_url = '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR'
    if not osp.exists(f'{DATAPATH}twitch-gamer_feat.csv'):
        gdown.download(id=dataset_drive_url,
            output=f'{DATAPATH}twitch-gamer_feat.csv', quiet=False)
    dataset_drive_url = '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0'
    if not osp.exists(f'{DATAPATH}twitch-gamer_edges.csv'):
        gdown.download(id=dataset_drive_url,
            output=f'{DATAPATH}twitch-gamer_edges.csv', quiet=False)
    
    edges = pd.read_csv(f'{DATAPATH}twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{DATAPATH}twitch-gamer_feat.csv')
    edge_index = torch.tensor(edges.to_numpy()).t().type(torch.LongTensor)
    num_nodes = len(nodes)
    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)
    dataset = NCDataset("twitch-gamer")
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_wiki():

    dataset_drive_url = '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' 
    if not osp.exists(f'{DATAPATH}wiki_features2M.pt'):
        gdown.download(id=dataset_drive_url, \
            output=f'{DATAPATH}wiki_features2M.pt', quiet=False)
    dataset_drive_url = '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u'
    if not osp.exists(f'{DATAPATH}wiki_edges2M.pt'):
        gdown.download(id=dataset_drive_url, \
            output=f'{DATAPATH}wiki_edges2M.pt', quiet=False)
    dataset_drive_url = '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK'
    if not osp.exists(f'{DATAPATH}wiki_views2M.pt'):
        gdown.download(id=dataset_drive_url, \
            output=f'{DATAPATH}wiki_views2M.pt', quiet=False)


    dataset = NCDataset("wiki") 
    features = torch.load(f'{DATAPATH}wiki_features2M.pt')
    edges = torch.load(f'{DATAPATH}wiki_edges2M.pt').T
    row, col = edges
    print(f"edges shape: {edges.shape}")
    label = torch.load(f'{DATAPATH}wiki_views2M.pt') 
    num_nodes = label.shape[0]

    print(f"features shape: {features.shape[0]}")
    print(f"Label shape: {label.shape[0]}")
    dataset.graph = {"edge_index": edges, 
                     "edge_feat": None, 
                     "node_feat": features, 
                     "num_nodes": num_nodes}
    dataset.label = label 
    return dataset 
