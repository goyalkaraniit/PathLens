# coding: cp1252

import networkx as nx
from community import community_louvain
from collections import Counter
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
from networkx.algorithms import community

import os
import os.path as osp
import collections
import scipy.sparse as sp

from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.utils import num_nodes
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
import pyamg
import scipy.sparse as sp
from numpy.linalg import eig, eigh

from torch_geometric.data import Data
import torch.nn as nn
import math
from torch.nn.parameter import Parameter

import torch
import argparse
import time
import random
#import json
random.seed(2021)

from data_proc_neur import *
from Load import *
from models import *
import torch_geometric.transforms as T
from OGB import *

@not_implemented_for('multigraph')
def divrank(G, alpha=0.25, d=0.85, personalization=None,
            max_iter=10000, tol=1.0e-6, nstart=None, weight='weight',
            dangling=None):
    '''
    Returns the DivRank (Diverse Rank) of the nodes in the graph.
    This code is based on networkx.pagerank.
    Args: (diff from pagerank)
      alpha: strength of self-link [0.0-1.0]
      d: the damping factor
    Reference:
      Qiaozhu Mei and Jian Guo and Dragomir Radev,
      DivRank: the Interplay of Prestige and Diversity in Information Networks,
      http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.174.7982
    '''

    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # self-link (DivRank)
    for n in W.nodes:
        for n_ in W.nodes:
            if n != n_ :
                if n_ in W[n]:
                    W[n][n_][weight] *= alpha
            else:
                if n_ not in W[n]:
                    W.add_edge(n, n_)
                W[n][n_][weight] = 1.0 - alpha

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v/s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    W_ = W.copy()
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = d * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            D_n = sum(W_[n][nbr][weight] * xlast[nbr] for nbr in W_[n])
            for nbr in W[n]:
                #x[nbr] += d * xlast[n] * W[n][nbr][weight]
                x[nbr] += (
                    d * (W_[n][nbr][weight] * xlast[nbr] / D_n) * xlast[n]
                )
            x[n] += danglesum * dangling_weights[n] + (1.0 - d) * p[n]

            for nbr in W[n]:
                W[n][nbr][weight] = (
                    (1.0 - d) * p[nbr]
                    + d * (W_[n][nbr][weight] * xlast[nbr]) / D_n
                )

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N*tol:
            return x
    raise NetworkXError('pagerank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)

def build_model(args, num_features, num_classes, num_nodes):
    if args.model == 'pgnn':
        model = pGNNNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            mu=args.mu,
                            p=args.p,
                            K=args.K,
                            dropout=args.dropout)
    elif args.model == 'linkx':
        model = LINKX(in_channels = num_features,
                      out_channels = num_classes,
                      num_hid = args.num_hid,
                      num_nodes = num_nodes,
                      dropout = args.dropout)
    elif args.model == 'poly':
        model = GFK(in_channels = num_features,
                      out_channels = num_classes,
                      hid_channels = args.num_hid,
                      dropoutC = args.dropout,
                      dropoutM = args.dprate,
                      num_nodes = num_nodes)
    elif args.model == 'spec':
        model = Specformer(in_channels = num_features,
                      out_channels = num_classes,
                      num_hid = args.num_hid,
                      num_heads = args.num_heads,
                      tran_dropout = args.dropout,
                      feat_dropout = args.dprate,
                      prop_dropout = args.alpha)
    elif args.model == 'aero':
        model = AERO_GNN( in_channels=num_features,
                          hid_channels = args.num_hid, 
                          out_channels = num_classes, 
                          num_heads = args.num_heads, 
                          num_nodes = num_nodes, 
                          K = args.K,
                          dropout=args.dropout)

    elif args.model == 'mlp':
        model = MLPNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'pmlpgcn':
        model = PMLP_GCN(in_channels=num_features,
                        out_channels=num_classes,
                        hidden_channels=args.num_hid,
                        dropout=args.dropout,
                        num_node=num_nodes)
    elif args.model == 'pmlpappnp':
        model = PMLP_APPNP(in_channels=num_features,
                        out_channels=num_classes,
                        hidden_channels=args.num_hid,
                        dropout=args.dropout,
                        num_node=num_nodes)
    elif args.model == 'dirgcn':
        model = DirGNN(num_features=num_features,
                        num_classes=num_classes,
                        hidden_dim=args.num_hid,
                        alpha = args.alpha,
                        conv_type='dir-gcn',
                        jumping_knowledge='max',
                        normalize=True)
    elif args.model == 'dirsage':
        model = DirGNN(num_features=num_features,
                        num_classes=num_classes,
                        hidden_dim=args.num_hid,
                        alpha = args.alpha,
                        conv_type='dir-sage',
                        jumping_knowledge='max',
                        normalize=True)
        """
    elif args.model == 'chebgnn':
        model = ChebNN(in_feats=num_features, 
                        n_hidden=args.num_hid,
                        n_classes = num_classes, 
                        K=args.K, 
                        dropout=args.dropout)
        """
    elif args.model == 'dagnn':
        model = DAGNN(in_channels=num_features,
                        out_channels=num_classes,
                        hidden=args.num_hid,
                        K = args.K,
                        dropout=args.dropout)
        """
    elif args.model == 'fagcn':
        model = FAGCN(g=g, in_dim=num_features,
                            hidden_dim=args.num_hid,
                            out_dim=num_classes, 
                            dropout=args.dropout,
                            eps=args.alpha)
        """
    elif args.model == 'acmgcnpp':
        model = GCNACMPP(nfeat=num_features, 
                        nhid=args.num_hid, 
                        nclass=num_classes, 
                        nnodes=num_nodes, 
                        dropout=args.dropout,
                        nlayers=2)
    elif args.model == 'acmiigcnpp':
        model = GCNIIACMPP(nfeat=num_features, 
                        nhid=args.num_hid, 
                        nclass=num_classes, 
                        nnodes=num_nodes, 
                        dropout=args.dropout,
                        nlayers=2)
    elif args.model == 'bern':
        model = BernNet(in_channels=num_features, 
                        out_channels=num_classes, 
                        K=args.K, 
                        hidden=args.num_hid, 
                        dropout=args.dropout, 
                        dprate=args.dprate)
        """
    elif args.model == 'wrgcn':
        model = WRGCN(num_features=num_features, num_classes=num_classes, dims=args.num_hid)
    elif args.model == 'wrgat':
        model = WRGAT(num_features=num_features, num_classes=num_classes, dims=args.num_hid)
        """
    elif args.model == 'gcn':
        model = GCNNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'gcnii':
        model = GCN2Net(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        alpha = args.alpha,
                        dropout=args.dropout)
    elif args.model == 'sage':
        model = GraphSAGENet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'sgc':
        model = SGCNet(in_channels=num_features,
                        out_channels=num_classes,
                        K=args.K)
    elif args.model == 'gat':
        model = GATNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        num_heads=args.num_heads,
                        dropout=args.dropout)
    elif args.model == 'gatv2':
        model = GATv2Net(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        num_heads=args.num_heads,
                        dropout=args.dropout)
    elif args.model == 'jk':
        model = JKNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        K=args.K,
                        alpha=args.alpha,
                        dropout=args.dropout)
    elif args.model == 'appnp':
        model = APPNPNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            K=args.K,
                            alpha=args.alpha,
                            dropout=args.dropout)
    elif args.model == 'gprgnn':
        model = GPRGNNNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            ppnp=args.ppnp,
                            K=args.K,
                            alpha=args.alpha,
                            Init=args.Init,
                            Gamma=args.Gamma,
                            dprate=args.dprate,
                            dropout=args.dropout)
    return model

def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def eigen_decompositon(adj):
    G = normalized_degree_laplacian(adj)
    e, u = eigh(G)
    return e, u

def normalized_degree_laplacian(adj):
    adj = torch.tensor(adj.toarray(), dtype=torch.float)
    degrees = torch.sum(adj, dim=1)
    degree_matrix = torch.diag(degrees)
    inv_sqrt_degrees = torch.diag(torch.pow(degrees, -0.5))
    inv_sqrt_degrees[torch.isinf(inv_sqrt_degrees)] = 0
    normalized_adj = torch.matmul(torch.matmul(inv_sqrt_degrees, adj), inv_sqrt_degrees)
    identity = torch.eye(adj.size(0))
    normalized_laplacian = identity - normalized_adj
    
    return normalized_laplacian
    
def normalize_graph(g):
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** -0.5)
    adj = np.dot(np.dot(deg, g), deg)
    L = np.eye(g.shape[0]) - adj
    return L

#Added func
def filter_rels(data, r):
	data = copy.deepcopy(data)
	mask = data.edge_color <= r
	data.edge_index = data.edge_index[:, mask]
	data.edge_weight = data.edge_weight[mask]
	data.edge_color = data.edge_color[mask]
	return data

def train(model, optimizer, data, e, u, args, adj_low, adj_high, adj_low_unnormalized):
    model.train()
    optimizer.zero_grad()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.x = data.x.to(device)
    data.y = data.y.to(device)
    #target = data.y[data.train_mask].squeeze() 
    if args.model=='spec':
        F.nll_loss(model(data.x, e, u)[data.train_mask], data.y[data.train_mask]).backward()
    elif(args.model=='acmgcnpp' or args.model=='acmiigcnpp'):
        F.nll_loss(model(data.x, adj_low, adj_high, adj_low_unnormalized)[data.train_mask], data.y[data.train_mask]).backward()
    else:
        F.nll_loss(model(data.x, data.edge_index, data.edge_attr)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test(model, data, e, u, args, adj_low, adj_high, adj_low_unnormalized):
    model.eval()
    if args.model=='acmgcnpp' or args.model=='acmiigcnpp':
        logits, accs = model(data.x, adj_low, adj_high, adj_low_unnormalized), []        
    elif args.model=='spec':
        logits, accs = model(data.x, e, u), []
    else:
        logits, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def main(args):
    print(args)
    if args.partition:
        rand_seed=2021
       
        if args.input in ['fb100', 'twitch', 'twitch-gamer', 'snap-patents', 'genius', 'pokec', 'arxiv-year']:
            if args.input=='twitch':
                sub_dataname = 'ENGB'
            else:
                sub_dataname = 'Penn94'            
       
            dataset = load_nc_dataset(args.input, sub_dataname)
            edge_features=None
            labels = dataset.label.to(torch.long)
            print(labels.type())
            data = Data(x=dataset.graph['node_feat'], y=labels, edge_index=dataset.graph['edge_index'])
            num_features = dataset.graph['node_feat'].shape[1]
            classes = torch.unique(dataset.label)
            num_classes = classes.shape[0]
            num_nodes = dataset.graph['num_nodes']
            print(num_features, num_classes,  num_nodes)
            shapes = (dataset.graph['num_nodes'], 1)
            data.train_mask = torch.zeros(shapes, dtype=torch.bool).squeeze()
            data.val_mask = torch.zeros(shapes, dtype=torch.bool).squeeze()
            data.test_mask = torch.zeros(shapes, dtype=torch.bool).squeeze()
            #split_idx = dataset.get_idx_split(train_prop=args.train_rate, valid_prop=args.val_rate)
            #data.train_mask = split_idx['train']
            #data.val_mask = split_idx['valid']
            #data.test_mask = split_idx['test']
            num_train = args.train_rate * num_nodes
            num_val = args.val_rate * num_nodes
           
       
        else:
            data, num_features, num_classes = load_data(args, rand_seed=2021)
            #print(data, data.x, data.edge_index)
           
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        print(data.x)
        #print(g.nodes.data())
        print(data.y)
       
       
               
        if os.path.isfile('dataset/' + args.input + '/' + args.input + str(args.num_partitions) + str(args.num_neighbors) + args.affinity + args.eigen_solver + args.labels + '.json'):
            with open('dataset/' + args.input + '/' + args.input + str(args.num_partitions) + str(args.num_neighbors) + args.affinity + args.eigen_solver + args.labels + '.json', 'r') as file:
                clustered_graph = json.load(file)    
            clusters = [[] for _ in range(args.num_partitions)]
               
            g = nx.Graph()  # Create an empty NetworkX graph


            # Add nodes from JSON data
            for node_data in clustered_graph['nodes'].values():
                g.add_node(int(node_data['id']))
                g.nodes[node_data['id']]['x'] = node_data['x']
                g.nodes[node_data['id']]['y'] = node_data['y']
                clusters[node_data['cluster'] - 1].append(node_data['id'])

            # Add edges from JSON data
            for edge in clustered_graph['edges']:
                g.add_edge(edge['source'], edge['target'])  # Add edge attributes as needed  

            init_num_nodes = g.number_of_nodes()
           
        else:        
            edge_index = data.edge_index.numpy()
            g = nx.Graph()
            g.add_nodes_from(range(data.num_nodes))
            g.add_edges_from(data.edge_index.T.numpy())

            # Print the number of nodes and edges in the graph
            init_num_nodes = g.number_of_nodes()
            print('Number of nodes:', g.number_of_nodes())
            print('Number of edges:', g.number_of_edges())

            for i, node in enumerate(g.nodes()):
                g.nodes[node]['x']=data.x[i].tolist()
                g.nodes[node]['y'] = data.y[i].item()
            if data.edge_attr is not None:
                for edge, features in zip(g.edges, data.edge_attr):
                    g.edges[edge]['attr'] = features.item()

            os.environ['OPENBLAS_NUM_THREADS'] = '250'
            laplacian_matrix = nx.laplacian_matrix(g).toarray()
            # Perform spectral clustering
            spectral_clustering = SpectralClustering(n_clusters=args.num_partitions, affinity=args.affinity, eigen_solver=args.eigen_solver, n_neighbors=args.num_neighbors, assign_labels=args.labels)
            cluster_labels = spectral_clustering.fit_predict(laplacian_matrix)

            # Assign nodes to clusters
            clusters = [[] for _ in range(args.num_partitions)]  # List to store nodes in each cluster
            for node, cluster_label in enumerate(cluster_labels):
                clusters[cluster_label].append(node)

            # Display the clusters
            for i, cluster in enumerate(clusters):
                print(f"Cluster {i + 1}: {cluster}")
           
       
       
            # Create a dictionary to store the graph data
            graph_data = {'nodes': {}, 'edges': []}

            # Iterate through nodes and extract attributes
            for node, attributes in g.nodes(data=True):
                for i in range(0, args.num_partitions):
                    if node in clusters[i]:
                        cluster_num = i+1
                        break
               
                node_data = {'id': node, 'x': g.nodes[node]['x'], 'y': int(g.nodes[node]['y']), 'cluster': int(cluster_num)}
                graph_data['nodes'][node] = node_data

            # Iterate through edges and extract attributes
       
            for edge in g.edges(data=True):
                edge_data = {'source': int(edge[0]), 'target': int(edge[1])}
                graph_data['edges'].append(edge_data)
       
            print(graph_data['nodes'][0])
            print(type(graph_data['nodes'][0]['x']), type(graph_data['nodes'][0]['y']), type(graph_data['nodes'][0]['cluster']))
            # Save the graph data as a dictionary using JSON
            filename = 'dataset/' + args.input + '/' + args.input + str(args.num_partitions) + str(args.num_neighbors) + args.affinity + args.eigen_solver + args.labels + '.json'
            with open(filename, 'w') as json_file:
                json.dump(graph_data, json_file)
        print(g.number_of_nodes())

        """ Global rank """
        if args.connect_type=='global' or args.connect_type=='g&l':
            if args.rank=='page':
                global_scores = nx.pagerank(g)
                print(sum(global_scores.values()))
            elif args.rank=='div':
                global_scores = divrank(g)
                print(sum(global_scores.values()))
        Supernodes = []
        shape = (len(data.x[0]), 1)
        Subgraphs = []
        ClusterSuper = {}
       
        """ Add supernodes & connect supernodes to some nodes of graph """
        i=0
        index = 0
        for cluster in clusters:
            print("Cluster: " , i+1)
           
            if len(cluster)<args.min_num_connect:
                continue

            """ Local rank """
            if args.connect_type=='local' or args.connect_type=='g&l':
                subgraph = g.subgraph(cluster)
                Subgraphs.append(subgraph)
                if args.rank=='page':
                    local_scores = nx.pagerank(subgraph)
                    print(sum(local_scores.values()))
                elif args.rank=='div':
                    local_scores = divrank(subgraph)
                    print(sum(local_scores.values()))
       
            """ Dummy Supernode initialization """
            class_dist={}
           
            Supernodes.append(g.number_of_nodes())
            for node in cluster:
                if g.nodes[node]['y'] not in class_dist.keys():
                    class_dist[g.nodes[node]['y']] = 1
                    #values[g.nodes[node]['y']].append(node)
                else:
                    class_dist[g.nodes[node]['y']] = class_dist[g.nodes[node]['y']] + 1
            print(len(cluster))
            print(class_dist)
            maximum = max(class_dist.values())
            #print(maximum)
            y_label = random.randint(0, num_classes-1)

            #print(y_label)
            x_features = (torch.randint(2, size=shape)).squeeze()
            print(x_features)
            g.add_node(Supernodes[i], x=x_features, y=y_label)
            ClusterSuper[index] = Supernodes[i]

            """ No. of nodes to connect """
            j=0
            Nodes_connect=[]
            connect = int(args.percent_connect * len(cluster))
            to_connect = max(connect, args.min_num_connect)
            print(args.min_num_connect, connect, to_connect)

            """ Connect supernodes to random nodes of resp. partition """
            if args.connect_type=='rand':
                while j<to_connect:
                    node_connect = random.choice(cluster)
                    if node_connect not in Nodes_connect:
                        Nodes_connect.append(node_connect)
                        j=j+1
           
                """ Connect supernodes as per global rank """
            elif args.connect_type=='global':

                """ Connect supernodes to nodes with highest global rank """
                if args.mode=='high':
                    k=0
                    for j, n in enumerate(sorted(global_scores, key=lambda n: global_scores[n], reverse=True)):
                        if k>=to_connect:
                            break
                        elif n in cluster:
                            if k<10:
                                print('# {}: \t {} \t {}'.format(j+1, n, global_scores[n]))
                            Nodes_connect.append(n)
                            k=k+1

                    """ Connect supernodes to nodes with lowest global rank """
                elif args.mode=='low':
                    k=0
                    for j, n in enumerate(sorted(global_scores, key=lambda n: global_scores[n], reverse=False)):
                        if k>=to_connect:
                            break
                        elif n in cluster:
                            if k<10:
                                print('# {}: \t {} \t {}'.format(j+1, n, global_scores[n]))
                            Nodes_connect.append(n)
                            k=k+1

                    """ Connect supernodes to nodes with middlemost global rank """
                elif args.mode=='mid':
                    Nodes_in_cluster = []
                    low = len(cluster)//2 - to_connect//2
                    high = len(cluster)//2 + to_connect//2
                    for j, n in enumerate(sorted(global_scores, key=lambda n: global_scores[n], reverse=True)):
                        if n in cluster:
                            Nodes_in_cluster.append([n, global_scores[n]])
                    if to_connect%2==1:
                        for k in range(low, high+1):
                            Nodes_connect.append(Nodes_in_cluster[k][0])
                    elif to_connect%2==0:
                        for k in range(low, high):
                            Nodes_connect.append(Nodes_in_cluster[k][0])
                           
                    """ Connect supernode to nodes with highest, least, middlemost global rank """
                elif args.mode=='lmh':
                    Nodes_in_cluster=[]
                    if to_connect%3==0:
                        num_low = num_mid = num_high = to_connect//3
                    elif to_connect%3==1:
                        num_low = num_high = to_connect//3
                        num_mid = to_connect//3 + 1
                    else:
                        num_mid = to_connect//3
                        num_low = num_high = to_connect//3 + 1

                    low = len(cluster)//2 - num_mid//2
                    high = len(cluster)//2 + num_mid//2
                    print(num_low, num_mid, num_high)
                    print(low, high)
                    for j, n in enumerate(sorted(global_scores, key=lambda n: global_scores[n], reverse=True)):
                        if n in cluster:
                            Nodes_in_cluster.append([n, global_scores[n]])
                    for l in range(len(cluster)):
                        if l in range(0, num_low):
                            Nodes_connect.append(Nodes_in_cluster[l][0])
                        elif l in range(len(cluster)-num_high, len(cluster)):
                            Nodes_connect.append(Nodes_in_cluster[l][0])
                        if num_mid%2==1:
                            if l in range(low, high+1):
                                Nodes_connect.append(Nodes_in_cluster[l][0])
                        elif num_mid%2==0:
                            if l in range(low, high):
                                Nodes_connect.append(Nodes_in_cluster[l][0])
                   
               
                """ Connect supernodes as per local rank """
            elif args.connect_type=='local':

                """ Connect supernodes to nodes with highest local rank """
                if args.mode=='high':
                    k=0
                    for j, n in enumerate(sorted(local_scores, key=lambda n: local_scores[n], reverse=True)):
                        if k>=to_connect:
                            break
                        elif n in cluster:
                            if k<10:
                                print('# {}: \t {} \t {}'.format(j+1, n, local_scores[n]))
                            Nodes_connect.append(n)
                            k=k+1

                    """ Connect supernodes to nodes with lowest local rank """
                elif args.mode=='low':
                    k=0
                    for j, n in enumerate(sorted(local_scores, key=lambda n: local_scores[n], reverse=False)):
                        if k>=to_connect:
                            break
                        elif n in cluster:
                            if k<10:
                                print('# {}: \t {} \t {}'.format(j+1, n, local_scores[n]))
                            Nodes_connect.append(n)
                            k=k+1

                    """ Connect supernode to nodes with middlemost local rank """
                elif args.mode=='mid':
                    low = len(cluster)//2 - to_connect//2
                    high = len(cluster)//2 + to_connect//2
                    for j, n in enumerate(sorted(local_scores, key=lambda n: local_scores[n], reverse=True)):
                        print('# {}: \t {} \t {}'.format(j+1, n, local_scores[n]))
                        if to_connect%2==1:
                            if j in range(low, high+1):
                                Nodes_connect.append(n)
                        elif to_connect%2==0:
                            if j in range(low, high):
                                Nodes_connect.append(n)
                               
                        """ Connect supernode to nodes with least, highest and middlemost local ranks """
                elif args.mode=='lmh':
                    if to_connect%3==0:
                        num_low = num_mid = num_high = to_connect//3
                    elif to_connect%3==1:
                        num_low = num_high = to_connect//3
                        num_mid = to_connect//3 + 1
                    else:
                        num_mid = to_connect//3
                        num_low = num_high = to_connect//3 + 1

                    low = len(cluster)//2 - num_mid//2
                    high = len(cluster)//2 + num_mid//2
                    print(num_low, num_mid, num_high)
                    print(low, high)

                    for j, n in enumerate(sorted(local_scores, key=lambda n: local_scores[n], reverse=True)):
                        print('# {}: \t {} \t {}'.format(j+1, n, local_scores[n]))
                        if j in range(0, num_low):
                            Nodes_connect.append(n)
                        elif j in range(len(cluster)-num_high, len(cluster)):
                            Nodes_connect.append(n)
                        if num_mid%2==1:
                            if j in range(low, high+1):
                                Nodes_connect.append(n)
                        elif num_mid%2==0:
                            if j in range(low, high):
                                Nodes_connect.append(n)
               
            print(Nodes_connect)
            print("Number of edges before adding: ", g.number_of_edges())
            for node_connect in Nodes_connect:
                g.add_edge(Supernodes[i], node_connect)
            print("Number of edges after adding: ", g.number_of_edges())
            i = i+1
            index = index+1

        """ Connect supernodes to each other """
        for node1 in Supernodes:
            for node2 in Supernodes:
                if node1!=node2:
                    g.add_edge(node1, node2)

        print("Number of nodes after adding supernodes: ", g.number_of_nodes())
        print("Number of edges after connecting supernodes: ", g.number_of_edges())
        print(ClusterSuper)


       
       
        node_features = torch.tensor([g.nodes[n]['x'] for n in g.nodes])
        y_values = torch.tensor([g.nodes[n]['y'] for n in g.nodes])
        edge_index = torch.tensor(list(g.edges)).t().contiguous()
        edge_features=None
        if nx.get_edge_attributes(g, 'attr'):
            edge_features = torch.tensor([g.edges[u, v]['attr'] for u, v in g.edges])
        datas = Data(x=node_features, y=y_values,  edge_index=edge_index, edge_attr=edge_features)
        num_train = int(len(data.y) / num_classes * args.train_rate)
        num_val = int(len(data.y) / num_classes * args.val_rate)
        #data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, rand_seed, num_train, num_val)
       
       
        #data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, rand_seed, num_train, num_val)
       

        results = []
        print(len(ClusterSuper))
        rand_seed_list = [0, 5, 66, 244, 2020]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_nodes = datas.x.size(0)
        for run in range(args.runs):
       
            data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, seed=rand_seed_list[run%5], train_num_per_c=num_train, val_num_per_c=num_val)
            i=0
            filename = 'dataset/' + args.input + '/' + 'Probab' + args.input + str(args.num_partitions) + str(args.num_neighbors) + args.affinity + args.eigen_solver + args.labels + str(rand_seed_list[run%5]) + '.json'
            if os.path.exists(filename):
                print(f"Loading edge class data from {filename}")
                with open(filename, 'r') as f:
                    edge_class_data = json.load(f)
                edge_class_count = {eval(key): value for key, value in edge_class_data['edge_class_count'].items()}
                edge_class_probab = {eval(key): value for key, value in edge_class_data['edge_class_probab'].items()}

            else:
                print(f"Calculating edge class data for seed {rand_seed_list[run%5]}")

                edge_class_count = {}
                edge_class_probab = {}

                for c1 in range(num_classes):
                    for c2 in range(c1, num_classes):
                        edge_class_count[tuple(sorted([c1, c2]))] = 0
                        edge_class_probab[tuple(sorted([c1, c2]))] = 0
               
                edge_index = data.edge_index
                num_edges_train = 0
           
                for edge_num in range(edge_index.size(1)):
                    src, dst = edge_index[:, edge_num]
                    if data.train_mask[src] and data.train_mask[dst]:
                        num_edges_train += 1
                        src_class = data.y[src].item()
                        dst_class = data.y[dst].item()

                        # Create a sorted tuple of the class labels
                        class_pair = tuple(sorted([src_class, dst_class]))
                        edge_class_count[class_pair] += 1

                for c1 in range(num_classes):
                    for c2 in range(c1, num_classes):
                        class_pair = tuple(sorted([c1, c2]))
                        print(f"Class pair {class_pair}:", edge_class_count[class_pair])
                        edge_class_probab[class_pair] = edge_class_count[class_pair]/num_edges_train
                        print(f"Class pair {class_pair} probab: ", edge_class_probab[class_pair])

                edge_class_data = {'edge_class_count': {str(key): value for key, value in edge_class_count.items()}, 'edge_class_probab': {str(key): value for key, value in edge_class_probab.items()}}

                with open(filename, 'w') as f:
                    json.dump(edge_class_data, f)

            # Supernode Label Calculation
            for i in range(len(clusters)):
                ClusterLabels = []
                if len(clusters[i])<args.min_num_connect:
                    continue
                else:
                    if i not in ClusterSuper.keys():
                        continue
                    supernode = ClusterSuper[i]
                    cluster_class_dist = {c: 0 for c in range(num_classes)}
                    SupernodeCluster = clusters[i]

                    for cluster_node in SupernodeCluster:
                        if cluster_node not in Supernodes and data.train_mask[cluster_node]:
                            cluster_class_dist[data.y[cluster_node].item()] += 1

                        # Calculate temporary label for each node in validation/test set which is a part of supernode's cluster
                        elif cluster_node not in Supernodes and (not data.train_mask[cluster_node]):
                            likelihood_node = {c: 0 for c in range(num_classes)}
                            node_class_dist = {c: 0 for c in range(num_classes)}

                            Neighbours = g.neighbors(cluster_node)
                            for Neighbour in Neighbours:
                                if Neighbour not in Supernodes and data.train_mask[Neighbour]:
                                    node_class_dist[data.y[Neighbour].item()] += 1
                           
                            for a in range(num_classes):
                                for b in range(num_classes):
                                    class_pair = tuple(sorted([a, b]))
                                    likelihood_node[a] = likelihood_node[a] + node_class_dist[b]*edge_class_probab[class_pair]

                            # Determine the label with the highest likelihood
                            predicted_label = max(likelihood_node, key=likelihood_node.get)
                            cluster_class_dist[predicted_label] += 1

                    max_class = max(cluster_class_dist, key=cluster_class_dist.get)
                    datas.y[supernode] = max_class
           
            shape_add_nodes = (len(Supernodes), 1)
            Supernodes_Train = torch.ones(shape_add_nodes, dtype=torch.bool).squeeze()
            Supernodes_NotTest = torch.zeros(shape_add_nodes, dtype=torch.bool).squeeze()
            shapes = (g.number_of_nodes(), 1)
            datas.train_mask = torch.zeros(shapes, dtype=torch.bool).squeeze()
            datas.val_mask = torch.zeros(shapes, dtype=torch.bool).squeeze()
            datas.test_mask = torch.zeros(shapes, dtype=torch.bool).squeeze()
            datas.train_mask = torch.cat((data.train_mask, Supernodes_Train))
            datas.val_mask = torch.cat((data.val_mask, Supernodes_NotTest))
            datas.test_mask = torch.cat((data.test_mask, Supernodes_NotTest))
       
           
            model = build_model(args, num_features, num_classes, num_nodes)
            model = model.to(device)
            datas = datas.to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            adj = sp.coo_matrix((torch.ones(datas.edge_index.size(1)), (datas.edge_index[0].cpu().numpy(), datas.edge_index[1].cpu().numpy())),
                        shape=(num_nodes, num_nodes))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            filename = 'dataset/' + args.input + '/' + 'Probab spec' + args.input + str(args.num_partitions) + str(args.num_neighbors) + args.affinity + args.eigen_solver + args.labels + str(rand_seed_list[run%5]) + '.pt'
            if os.path.exists(filename) and args.model=='spec':
                print("Loading e, u")
                eudata = torch.load(filename)
                e = eudata['e']
                u = eudata['u']
                e = torch.tensor(e, dtype=torch.float).to(device)
                u = torch.tensor(u, dtype=torch.float).to(device)
            
            elif args.model=='spec':
                print("Calculating e, u")
                e, u = eigen_decompositon(adj)
                #print(adj)
                e = torch.tensor(e, dtype=torch.float).to(device)
                u = torch.tensor(u, dtype=torch.float).to(device)
                torch.save({'e': e, 'u': u}, filename)
            else:
                e = None
                u = None

            filename = 'dataset/' + args.input + '/' + 'Probab acm' + args.input + str(args.num_partitions) + str(args.num_neighbors) + args.affinity + args.eigen_solver + args.labels + str(rand_seed_list[run%5]) + '.pt'
            if os.path.exists(filename) and args.model=='acmgcnpp' or args.model=='acmiigcnpp':
                print("Loading Adjs")
                adjdata = torch.load(filename)
                adj_low_unnormalized = adjdata['adj_low_unnormalized']
                adj_low = adjdata['adj_low']
                adj_high = adjdata['adj_high']
                adj_low = adj_low.to(device)
                adj_low_unnormalized = adj_low_unnormalized.to(device)
                adj_high = adj_high.to(device)

            elif args.model=='acmgcnpp' or args.model=='acmiigcnpp':
                print("Calculating Adjs")
                adj_low_unnormalized = sparse_mx_to_torch_sparse_tensor(adj)
                adj_low = normalize_tensor(torch.eye(num_nodes) + adj_low_unnormalized.to_dense())
                adj_high = (torch.eye(num_nodes) - adj_low).to(device).to_sparse()
                adj_low = adj_low.to(device)
                adj_low_unnormalized = adj_low_unnormalized.to(device)
                torch.save({'adj_low_unnormalized': adj_low_unnormalized, 'adj_low': adj_low, 'adj_high': adj_high}, filename)
            else:
                adj_low = None
                adj_high = None
                adj_low_unnormalized = None
       
            t1 = time.time()
            best_val_acc = test_acc = 0
            for epoch in range(1, args.epochs+1):
                train(model, optimizer, datas, e, u, args, adj_low, adj_high, adj_low_unnormalized)
                train_acc, val_acc, tmp_test_acc = test(model, datas, e, u, args, adj_low, adj_high, adj_low_unnormalized)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                log = 'Run: {:02d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(run+1, epoch, train_acc, best_val_acc, test_acc))
            t2 = time.time()
            # print('{}, {}, Accuacy: {:.4f}, Time: {:.4f}'.format(args.model, args.input, test_acc, t2-t1))
            results.append(test_acc)
        results = 100 * torch.Tensor(results)
        print(results)
        print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f}')
       
        filename = "Results/Probabilistic" + args.input + args.rank + args.model + ".csv"
        file = open(filename, 'a')
        file.write(args.input)
        file.write('\t')
        file.write(str(args.train_rate))
        file.write('\t')
        file.write(str(args.val_rate))
        file.write('\t')
        file.write(str(args.partition))
        file.write('\t')
        file.write(str(args.num_partitions))
        file.write('\t')
        file.write(args.rank)
        file.write('\t')
        file.write(args.connect_type)
        file.write('\t')
        file.write(args.mode)
        file.write('\t')
        file.write(str(args.min_num_connect))
        file.write('\t')
        file.write(str(args.percent_connect))
        file.write('\t')
        file.write(str(args.num_neighbors))
        file.write('\t')
        file.write(args.affinity)
        file.write('\t')
        file.write(args.eigen_solver)
        file.write('\t')
        file.write(args.labels)
        file.write('\t')
        file.write(args.model)
        file.write('\t')
        file.write(str(args.runs))
        file.write('\t')
        file.write(str(args.epochs))
        file.write('\t')
        file.write(str(args.lr))
        file.write('\t')
        file.write(str(args.weight_decay))
        file.write('\t')
        file.write(str(args.num_hid))
        file.write('\t')
        file.write(str(args.dropout))
        file.write('\t')
        file.write(str(args.mu))
        file.write('\t')
        file.write(str(args.p))
        file.write('\t')
        file.write(str(args.K))
        file.write('\t')
        file.write(str(args.num_heads))
        file.write('\t')
        file.write(str(args.alpha))
        file.write('\t')
        file.write(str(args.dprate))
        file.write('\t')
        file.write(f"{results.mean():.2f} ± {results.std():.2f}")
        file.write('\n')
        file.close()
       
    elif args.input in ['ogbn-arxiv', 'ogbn-mag', 'ogbn-products']:
        OGBNDataset(args, rand_seed=2021)
    else:
        data, num_features, num_classes = load_data(args, rand_seed=2021)
        #print(data, data.x, data.edge_index)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = []
        for run in range(args.runs):
            model = build_model(args, num_features, num_classes)
            model = model.to(device)
            data = data.to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
       
            t1 = time.time()
            best_val_acc = test_acc = 0
            for epoch in range(1, args.epochs+1):
                train(model, optimizer, data)
                train_acc, val_acc, tmp_test_acc = test(model, data)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                log = 'Run: {:02d}, Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(run+1, epoch, train_acc, best_val_acc, test_acc))
            t2 = time.time()
            # print('{}, {}, Accuacy: {:.4f}, Time: {:.4f}'.format(args.model, args.input, test_acc, t2-t1))
            results.append(test_acc)
        results = 100 * torch.Tensor(results)
        print(results)
        print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f}')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default='texas',                    
                        help='Input graph.')
    parser.add_argument('--train_rate',
                        type=float,
                        default=0.6,
                        help='Training rate.')
    parser.add_argument('--val_rate',
                        type=float,
                        default=0.2,
                        help='Validation rate.')
    parser.add_argument('--partition',
                        type=bool,
                        default=True)
    parser.add_argument('--num_partitions',
                        type=int,
                        default=40,
                        help='Number of partitions.')
    parser.add_argument('--rank',
                        type=str,
                        default='page',
                        choices = ['page', 'div', 'rand'],
                        help='Manner of connections to add.')
    parser.add_argument('--connect_type',
                        type=str,
                        default='local',
                        choices = ['rand', 'global', 'local', 'g&l'],
                        help='Manner of connections to add.')
    parser.add_argument('--mode',
                        type=str,
                        default='low',
                        choices = ['low', 'mid', 'high', 'lmh'],
                        help='Which ordering of connections to add.')
    parser.add_argument('--min_num_connect',
                        type=int,
                        default=3,
                        help='Minimum Number of connections to add.')
    parser.add_argument('--percent_connect',
                        type=float,
                        default=0.3,
                        help='Percetage of connections to add.')
    parser.add_argument('--num_neighbors',
                        type=int,
                        default=10,
                        help='Number of neighbours in spectral clustering.')
    parser.add_argument('--affinity',
                        type=str,
                        default='nearest_neighbors',
                        choices=['nearest_neighbors', 'rbf'],
                        help='Construct affinity matrix.')
    parser.add_argument('--eigen_solver',
                        type=str,
                        default='lobpcg',
                        choices=['arpack', 'amg', 'lobpcg'],
                        help='Eigen solver in embedding space')
    parser.add_argument('--labels',
                        type=str,
                        default='cluster_qr',
                        choices=['cluster_qr', 'kmeans', 'discretize'],
                        help='Labels in embedding space')
    parser.add_argument('--model',
                        type=str,
                        default='linkx',
                        choices=['linkx', 'spec', 'pgnn', 'poly', 'mlp', 'gcn', 'gcnii', 'cheb', 'sgc', 'gat', 'gatv2', 'sage', 'jk', 
                        'dirgcn', 'dirsage', 'appnp', 'gprgnn', 'aero', 'dagnn', 'acmgcnpp', 'acmiigcnpp', 'bern', 'pmlpgcn', 'pmlpappnp'],
                        help='GNN model')
    parser.add_argument('--runs',
                        type=int,
                        default=15,
                        help='Number of repeating experiments.')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--num_hid',
                        type=int,
                        default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--mu',
                        type=float,
                        default=0.1,
                        help='mu.')
    parser.add_argument('--p',
                        type=float,
                        default=2,
                        help='p.')
    parser.add_argument('--K',
                        type=int,
                        default=10,
                        help='K.')
    parser.add_argument('--num_heads',
                        type=int,
                        default=8,
                        help='Number of heads.')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.1,
                        help='alpha.')
    parser.add_argument('--Init',
                        type=str,
                        default='PPR',
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'])
    parser.add_argument('--Gamma',
                        default=None)
    parser.add_argument('--ppnp',
                        type=str,
                        default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--dprate',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(get_args())
