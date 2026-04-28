import torch
import numpy as np
import argparse
import time
import networkx as nx
from data_proc_neur import *
from models import *
import torch_geometric.transforms as T
from OGB import *
from sklearn.cluster import SpectralClustering
import pyamg
from networkx.utils import not_implemented_for
import random
import copy
import warnings
warnings.filterwarnings("ignore")

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

def SpectralHighway(Graph, data, args, num_classes):
    g = Graph.copy()

    os.environ['OPENBLAS_NUM_THREADS'] = '250'
    laplacian_matrix = nx.laplacian_matrix(g).toarray()
    # Perform spectral clustering
    spectral_clustering = SpectralClustering(n_clusters=args.num_partitions, affinity=args.affinity, eigen_solver=args.eigen_solver, n_neighbors=args.num_neighbors, assign_labels=args.labels)
    cluster_labels = spectral_clustering.fit_predict(laplacian_matrix)

    # Assign nodes to clusters
    clusters = [[] for _ in range(args.num_partitions)]  # List to store nodes in each cluster
    for node, cluster_label in enumerate(cluster_labels):
        clusters[cluster_label].append(node)

    """ Global rank """
    if args.connect_type=='global' or args.connect_type=='g&l':
        if args.rank=='page':
            global_scores = nx.pagerank(g)
            #print(sum(global_scores.values()))
        elif args.rank=='div':
            global_scores = divrank(g)
            #print(sum(global_scores.values()))
    Supernodes = []
    shape = (len(data.x[0]), 1)
    Subgraphs = []
    ClusterSuper = {}
       
    """ Add supernodes & connect supernodes to some nodes of graph """
    i=0
    index = 0
    for cluster in clusters:
        #print("Cluster: " , i+1)
       
        if len(cluster)<args.min_num_connect:
            continue
        """ Local rank """
        if args.connect_type=='local' or args.connect_type=='g&l':
            subgraph = g.subgraph(cluster)
            Subgraphs.append(subgraph)
            if args.rank=='page':
                local_scores = nx.pagerank(subgraph)
                #print(sum(local_scores.values()))
            elif args.rank=='div':
                local_scores = divrank(subgraph)
                #print(sum(local_scores.values()))
       
        """ Dummy Supernode initialization """
        class_dist={}
        
        Supernodes.append(g.number_of_nodes())
        for node in cluster:
            if g.nodes[node]['y'] not in class_dist.keys():
                class_dist[g.nodes[node]['y']] = 1
                #values[g.nodes[node]['y']].append(node)
            else:
                class_dist[g.nodes[node]['y']] = class_dist[g.nodes[node]['y']] + 1
        #print(len(cluster))
        #print(class_dist)
        maximum = max(class_dist.values())
        #print(maximum)

        y_label = random.randint(0, num_classes-1)
        #print(y_label)
        x_features = (torch.randint(2, size=shape)).squeeze()
        #print(x_features)
        g.add_node(Supernodes[i], x=x_features, y=y_label)
        ClusterSuper[index] = Supernodes[i]

        """ No. of nodes to connect """
        j=0
        Nodes_connect=[]
        connect = int(args.percent_connect * len(cluster))
        to_connect = max(connect, args.min_num_connect)
        #print(args.min_num_connect, connect, to_connect)

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
                        Nodes_connect.append(n)
                        k=k+1
                """ Connect supernodes to nodes with lowest global rank """
            elif args.mode=='low':
                k=0
                for j, n in enumerate(sorted(global_scores, key=lambda n: global_scores[n], reverse=False)):
                    if k>=to_connect:
                        break
                    elif n in cluster:
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
                #print(num_low, num_mid, num_high)
                #print(low, high)
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
                        Nodes_connect.append(n)
                        k=k+1
                """ Connect supernodes to nodes with lowest local rank """
            elif args.mode=='low':
                k=0
                for j, n in enumerate(sorted(local_scores, key=lambda n: local_scores[n], reverse=False)):
                    if k>=to_connect:
                        break
                    elif n in cluster:
                        Nodes_connect.append(n)
                        k=k+1
                """ Connect supernode to nodes with middlemost local rank """
            elif args.mode=='mid':
                low = len(cluster)//2 - to_connect//2
                high = len(cluster)//2 + to_connect//2
                for j, n in enumerate(sorted(local_scores, key=lambda n: local_scores[n], reverse=True)):
                    #print('# {}: \t {} \t {}'.format(j+1, n, local_scores[n]))
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
                #print(num_low, num_mid, num_high)
                #print(low, high)
                for j, n in enumerate(sorted(local_scores, key=lambda n: local_scores[n], reverse=True)):
                    #print('# {}: \t {} \t {}'.format(j+1, n, local_scores[n]))
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
           
        #print(Nodes_connect)
        #print("Number of edges before adding: ", g.number_of_edges())
        for node_connect in Nodes_connect:
            g.add_edge(Supernodes[i], node_connect)
        #print("Number of edges after adding: ", g.number_of_edges())
        i = i+1
        index = index+1

    """ Connect supernodes to each other """
    for node1 in Supernodes:
        for node2 in Supernodes:
            if node1!=node2:
                g.add_edge(node1, node2)
    #print("Number of nodes after adding supernodes: ", g.number_of_nodes())
    #print("Number of edges after connecting supernodes: ", g.number_of_edges())

    node_features = torch.tensor([g.nodes[n]['x'] for n in g.nodes])
    y_values = torch.tensor([g.nodes[n]['y'] for n in g.nodes])
    edge_index = torch.tensor(list(g.edges)).t().contiguous()
    edge_features=None
    if nx.get_edge_attributes(g, 'attr'):
        edge_features = torch.tensor([g.edges[u, v]['attr'] for u, v in g.edges])

    return g, clusters, ClusterSuper, Supernodes

def Unigram(Graph, data, args, num_classes, clusters, ClusterSuper, Supernodes, seed=0):
    g = Graph.copy()
    node_features = torch.tensor([g.nodes[n]['x'] for n in g.nodes])
    y_values = torch.tensor([g.nodes[n]['y'] for n in g.nodes])
    edge_index = torch.tensor(list(g.edges)).t().contiguous()
    edge_features=None
    if nx.get_edge_attributes(g, 'attr'):
        edge_features = torch.tensor([g.edges[u, v]['attr'] for u, v in g.edges])
    datas = Data(x=node_features, y=y_values,  edge_index=edge_index, edge_attr=edge_features)
    num_train = int(len(data.y) / num_classes * args.train_rate)
    num_val = int(len(data.y) / num_classes * args.val_rate)
    data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, seed=seed, train_num_per_c=num_train, val_num_per_c=num_val)
    
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
            #print(f"Class pair {class_pair}:", edge_class_count[class_pair])
            edge_class_probab[class_pair] = edge_class_count[class_pair]/num_edges_train
            #print(f"Class pair {class_pair} probab: ", edge_class_probab[class_pair])

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

    return datas


def Bigram(Graph, data, args, num_classes, clusters, ClusterSuper, Supernodes, seed=0):
    g = Graph.copy()
    node_features = torch.tensor([g.nodes[n]['x'] for n in g.nodes])
    y_values = torch.tensor([g.nodes[n]['y'] for n in g.nodes])
    edge_index = torch.tensor(list(g.edges)).t().contiguous()
    edge_features=None
    if nx.get_edge_attributes(g, 'attr'):
        edge_features = torch.tensor([g.edges[u, v]['attr'] for u, v in g.edges])
    datas = Data(x=node_features, y=y_values,  edge_index=edge_index, edge_attr=edge_features)
    num_train = int(len(data.y) / num_classes * args.train_rate)
    num_val = int(len(data.y) / num_classes * args.val_rate)
    data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, seed=seed, train_num_per_c=num_train, val_num_per_c=num_val)

    edge_class_count_uni = {}
    edge_class_probab_uni = {}

    for c1 in range(num_classes):
        for c2 in range(c1, num_classes):
            edge_class_count_uni[tuple(sorted([c1, c2]))] = 0
            edge_class_probab_uni[tuple(sorted([c1, c2]))] = 0
               
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
            edge_class_count_uni[class_pair] += 1

    for c1 in range(num_classes):
        for c2 in range(c1, num_classes):
            class_pair = tuple(sorted([c1, c2]))
            #print(f"Class pair {class_pair}:", edge_class_count_uni[class_pair])
            edge_class_probab_uni[class_pair] = edge_class_count_uni[class_pair]/num_edges_train
            #print(f"Class pair {class_pair} probab: ", edge_class_probab_uni[class_pair])

    edge_class_count_bi = {}
    edge_class_probab_bi = {}

    for c3 in range(num_classes):
        for c1 in range(num_classes):
            for c2 in range(c1, num_classes):
                edge_class_count_bi[(tuple(sorted([c1, c2])), c3)] = 0
                edge_class_probab_bi[(tuple(sorted([c1, c2])), c3)] = 0
               
    num_neighbour_pair = 0
           
    for node in g.nodes():
        if node in Supernodes or not data.train_mask[node]:
            continue
        Train_Neighbours = []
        Neighbours = g.neighbors(node)
        for neighbor in Neighbours:
            if neighbor not in Supernodes and data.train_mask[neighbor]:
                Train_Neighbours.append(neighbor)
        Size = len(Train_Neighbours)
        for neighbor1 in range(0, Size, 1):
            for neighbor2 in range(neighbor1+1, Size, 1):
                num_neighbour_pair += 1
                class_triplet = (tuple(sorted([data.y[neighbor1].item(), data.y[neighbor2].item()])), data.y[node].item())
                edge_class_count_bi[class_triplet] += 1

           
    for c3 in range(num_classes):
        for c1 in range(num_classes):
            for c2 in range(c1, num_classes):
                class_triplet = (tuple(sorted([c1, c2])), c3)
                #print(f"Class triplet {class_triplet}:", edge_class_count_bi[class_triplet])
                edge_class_probab_bi[class_triplet] = edge_class_count_bi[class_triplet]/num_neighbour_pair
                #print(f"Class triplet {class_triplet} probab: ", edge_class_probab_bi[class_triplet])

                           

    # Supernode Label Calculation
    for i in range(len(clusters)):
        ClusterLabels = []
        if len(clusters[i])<args.min_num_connect:
            continue
        else:
            if i not in ClusterSuper:
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

                    num_neighbors = sum(node_class_dist.values())

                    # Use bigram model if >=2 neighbours
                    if num_neighbors>=2:
                        # Here we consider both neighbours belong to same class, now we calculate likelihood of node to be in each class
                        for a in range(num_classes):
                            if node_class_dist[a]>=2:
                                for b in range(num_classes):
                                    class_triplet = ((a, a), b)
                                    num_pairs = (node_class_dist[a]*(node_class_dist[a] - 1))/2
                                    likelihood_node[b] += num_pairs * edge_class_probab_bi[class_triplet]

                        # Here we consider both neighbours belong to different classes, now we calculate likelihood of node to be in each class
                        for a in range(num_classes):
                            if node_class_dist[a]>=1:
                                for b in range(a+1, num_classes, 1):
                                    if node_class_dist[b]>=1:
                                        for c in range(num_classes):
                                            class_triplet = (tuple(sorted([a, b])), c)
                                            num_pairs = node_class_dist[a] * node_class_dist[b]
                                            likelihood_node[c] += num_pairs * edge_class_probab_bi[class_triplet]

                    # Use unigram model if 1 neighbour  
                    else:                  
                        for a in range(num_classes):
                            for b in range(num_classes):
                                class_pair = tuple(sorted([a, b]))
                                likelihood_node[a] = likelihood_node[a] + node_class_dist[b]*edge_class_probab_uni[class_pair]
                               

                    # Determine the label with the highest likelihood
                    predicted_label = max(likelihood_node, key=likelihood_node.get)
                    cluster_class_dist[predicted_label] += 1

            max_class = max(cluster_class_dist, key=cluster_class_dist.get)
            datas.y[supernode] = max_class
    return datas

def FileWrite(filename, intra_path_len_v1, inter_path_len_v1, intra_path_len_v2, inter_path_len_v2, args):
    file = open(filename, 'a')
    file.write(args.input)
    file.write('\t')
    file.write(str(args.train_rate))
    file.write('\t')
    file.write(str(args.val_rate))
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
    file.write(args.model)
    file.write('\t')
    file.write(str(args.UNIBI))
    file.write('\n')
    file.write("Intra Version 1: \t")
    file.write(str(intra_path_len_v1))
    file.write('\n')
    file.write("Intra Version 2: \t")
    file.write(str(intra_path_len_v2))
    file.write('\n')
    file.write("Inter Version 1: \t")
    file.write(str(inter_path_len_v1))
    file.write('\n')
    file.write("Inter Version 2: \t")
    file.write(str(inter_path_len_v2))
    file.write('\n')
    file.close()

def AvgShortPathLen(g):
    classes = []
    for node in g.nodes():
        if g.nodes[node]['y'] not in classes:
            classes.append(g.nodes[node]['y'])

    PATHS = dict(nx.all_pairs_shortest_path(g))
    """
    for node in g.nodes():
        print(PATHS[0][node])
    """
    Diameter=0
    for path in PATHS:
        Diameter = max(Diameter, len(PATHS[path]))

    Avg_Path_Lens_Classes = {}
    Path_Lens_Classes = {}
    Path_Lens_Classes_sum_count = {}

    for c1 in range(0, len(classes), 1):
        for c2 in range(c1, len(classes), 1):
            class1 = classes[c1]
            class2 = classes[c2]
            Avg_Path_Lens_Classes[tuple(sorted([class1, class2]))] = 0
            Path_Lens_Classes[tuple(sorted([class1, class2]))] = []

    for node1 in g.nodes():
        for node2 in g.nodes():
            if node1 >= node2:  
                continue
            csrc = g.nodes[node1]['y']
            cdest = g.nodes[node2]['y']
            if nx.has_path(g, node1, node2)==False:
                Path_Lens_Classes[tuple(sorted([csrc, cdest]))].append(Diameter+1)
            else:
                Path_Lens_Classes[tuple(sorted([csrc, cdest]))].append(len(PATHS[node1][node2]) -1)

    
    for c1 in range(0, len(classes), 1):
        for c2 in range(c1, len(classes), 1):
            Class_Pair = tuple(sorted([classes[c1], classes[c2]]))
            if len(Path_Lens_Classes[Class_Pair]) != 0:
                Avg_Path_Lens_Classes[Class_Pair] = sum(Path_Lens_Classes[Class_Pair])/len(Path_Lens_Classes[Class_Pair])
                Path_Lens_Classes_sum_count[Class_Pair] = (sum(Path_Lens_Classes[Class_Pair]), len(Path_Lens_Classes[Class_Pair]))
            else:
                Avg_Path_Lens_Classes[Class_Pair] = Diameter + 1
                Path_Lens_Classes_sum_count[Class_Pair] = (Diameter + 1, 1)

    IntraClassAvgPathLen_v1 = 0
    InterClassAvgpathLen_v1 = 0
    IntraClassAvgPathLen_v2 = 0
    InterClassAvgpathLen_v2 = 0

    Path_lens_intraclass_sum = 0
    Path_lens_intraclass_count = 0
    Path_lens_interclass_sum = 0
    Path_lens_interclass_count = 0


    for c1 in range(0, len(classes), 1):
        Class_Pair = tuple(sorted([classes[c1], classes[c1]]))
        Path_lens_intraclass_sum += Path_Lens_Classes_sum_count[Class_Pair][0]
        Path_lens_intraclass_count += Path_Lens_Classes_sum_count[Class_Pair][1]
        IntraClassAvgPathLen_v2 += Avg_Path_Lens_Classes[Class_Pair]          
    IntraClassAvgPathLen_v1 = Path_lens_intraclass_sum/Path_lens_intraclass_count
    IntraClassAvgPathLen_v2 = IntraClassAvgPathLen_v2/len(classes)
    
    for c1 in range(0, len(classes), 1):
        for c2 in range(c1+1, len(classes), 1):
            Class_Pair = tuple(sorted([classes[c1], classes[c2]]))
            Path_lens_interclass_sum += Path_Lens_Classes_sum_count[Class_Pair][0]
            Path_lens_interclass_count += Path_Lens_Classes_sum_count[Class_Pair][1]
            InterClassAvgpathLen_v2 += Avg_Path_Lens_Classes[Class_Pair]
    InterClassAvgPathLen_v1 = Path_lens_interclass_sum/Path_lens_interclass_count
    Num_Diff_Class_Pairs = (len(classes) * (len(classes) - 1))/2
    InterClassAvgpathLen_v2 = InterClassAvgpathLen_v2/Num_Diff_Class_Pairs

    return IntraClassAvgPathLen_v1, InterClassAvgpathLen_v1, IntraClassAvgPathLen_v2, InterClassAvgpathLen_v2

def node_homophily(g):
    Summation=0
    for node in g.nodes():
        numerator=0
        neighbours = list(g.neighbors(node))
        num_neighbours = len(neighbours)
        for neighbour in neighbours:
            if g.nodes[node]['y']==g.nodes[neighbour]['y']:
                numerator += 1
        if num_neighbours!=0:
            Term = numerator/num_neighbours
        else:
            Term = 0
        Summation += Term
    node_homo = Summation/g.number_of_nodes()
    return node_homo

def edge_homophily(g):
    edge_homo_num = 0
    for edge in g.edges():
        if g.nodes[edge[0]]['y']==g.nodes[edge[1]]['y']:
            edge_homo_num += 1
    edge_homo = edge_homo_num/g.number_of_edges()
    return edge_homo

def adjusted_homophily(g, edge_homo):
    classes = []
    for node in g.nodes():
        if g.nodes[node]['y'] not in classes:
            classes.append(g.nodes[node]['y'])
    DegreeSumSquareClass = []
    for Class in classes:
        degree = 0
        for node in g.nodes():
            if g.nodes[node]['y']==Class:
                degree += g.degree[node]
        DegreeSumSquareClass.append(degree*degree)
    SumDegreeSquare = 0
    for degree in range(0, len(DegreeSumSquareClass)):
        SumDegreeSquare += DegreeSumSquareClass[degree]
    Parameter = SumDegreeSquare/(pow( 2*g.number_of_edges(), 2))
    Adj_Homo = (edge_homo - Parameter)/(1 - Parameter)
    return Adj_Homo

def improved_homophily(g):
    classes = []
    for node in g.nodes():
        if g.nodes[node]['y'] not in classes:
            classes.append(g.nodes[node]['y'])
    Class_hk_size = {}
    for Class in classes:
        Class_hk_size[Class] = []
        Numerator_hk = 0
        Denominator_hk = 0
        Size=0
        for node in g.nodes():
            if g.nodes[node]['y']==Class:
                Size += 1
                neighbours = list(g.neighbors(node))
                num_neighbours = len(neighbours)
                Denominator_hk += num_neighbours
                SameClassNeigh=0
                for neighbour in neighbours:
                    if g.nodes[node]['y']==g.nodes[neighbour]['y']:
                        SameClassNeigh += 1
                Numerator_hk += SameClassNeigh
        hk = Numerator_hk/Denominator_hk
        Class_hk_size[Class].append(Size)
        Class_hk_size[Class].append(hk)

    Sum_Max_hk_0=0
    for Class in classes:
        Term1 = Class_hk_size[Class][1] - (Class_hk_size[Class][0] / g.number_of_nodes())
        Max = max(Term1, 0)
        Sum_Max_hk_0 += Max
    improved_homo = Sum_Max_hk_0 / (len(classes) - 1)
    
    return improved_homo

def aggregation_homophily(features, adj, label, modified=True):
    
    inner_prod = torch.mm(
        torch.mm(adj, features), torch.mm(adj, features).transpose(0, 1)
    )
    labels = label
    weight_matrix = torch.zeros(
        adj.clone().detach().size(0), labels.clone().detach().max() + 1
    )
    #print(labels.shape)
    #print(weight_matrix.shape)
    
    for i in range(labels.max() + 1):
        weight_matrix[:, i] = torch.mean(inner_prod[:, labels == i], 1)
    return torch.mean(torch.argmax(weight_matrix, 1).eq(labels).float())  

def FileWriteOther(filename, homoscoredict, args):
    file = open(filename, 'a')
    file.write(args.input)
    file.write('\t')
    file.write(str(args.train_rate))
    file.write('\t')
    file.write(str(args.val_rate))
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
    file.write(args.model)
    file.write('\t')
    file.write(args.UNIBI)
    file.write('\n')
    file.write(str(homoscoredict))
    file.write('\n')
    file.close()

def main(args):
    print(args)
    if args.input in ['fb100', 'twitch', 'twitch-gamer', 'snap-patents', 'genius', 'pokec', 'arxiv-year']:
        if args.input=='twitch':
            sub_dataname = 'ENGB'
        else:
            sub_dataname = 'Penn94'            
        
        dataset = load_nc_dataset(args.input, sub_dataname)
        edge_features=None
        labels = dataset.label.to(torch.long)
        if args.input=='arxiv-year':
          labels = labels.squeeze()
        #print(labels.type())
        data = Data(x=dataset.graph['node_feat'], y=labels, edge_index=dataset.graph['edge_index'])
        num_features = dataset.graph['node_feat'].shape[1]
        classes = torch.unique(dataset.label)
        num_classes = classes.shape[0]
        num_nodes = dataset.graph['num_nodes']
        #print(num_features, num_classes,  num_nodes)
        
        
    elif args.input in ['ogbn-arxiv', 'ogbn-mag', 'ogbn-products']:
        OGBNDataset(args, rand_seed=2021)
    else:
        data, num_features, num_classes = load_data(args, rand_seed=2021)

    if args.input not in ['ogbn-arxiv', 'ogbn-mag', 'ogbn-products']:
        #print(data, data.x, data.edge_index)
        num_nodes = data.x.size(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = []

        edge_index = data.edge_index.numpy()
        edge_index = data.edge_index
        num_nodes = data.x.size(0)

    
    intra_path_len_v1 = {}
    intra_path_len_v2 = {}
    inter_path_len_v1 = {}
    inter_path_len_v2 = {}
    agg_homo = {}
    node_homo = {}
    edge_homo = {}
    Adj_Homo = {}
    improved_homo = {}
    g_dict = {}
    data_dict = {}
    rand_seed_list = [0, 5, 66, 244, 2020]

    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from(data.edge_index.T.numpy())
    
    for i, node in enumerate(g.nodes()):
        g.nodes[node]['x'] = data.x[i].tolist()
        g.nodes[node]['y'] = data.y[i].item()
    if data.edge_attr is not None:
        for edge, features in zip(g.edges, data.edge_attr):
            g.edges[edge]['attr'] = features.item()

    edge_index = data.edge_index.numpy()
    edge_index = data.edge_index
    adjacency_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))
    adjacency_matrix = adjacency_matrix.to_dense()

    if args.input not in ['chameleon_filtered', 'chameleon', 'squirrel', 'squirrel_filtered']:
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.t()) / 2
    
    agg_homo["orig"] = aggregation_homophily(data.x, adjacency_matrix, data.y)

    g_dict[f"orig"] = copy.deepcopy(g)
    data_dict[f"orig"] = copy.deepcopy(data)

    node_homo["orig"] = node_homophily(g)
    edge_homo["orig"] = edge_homophily(g)
    Adj_Homo["orig"] = adjusted_homophily(g, edge_homo["orig"])
    improved_homo["orig"] = improved_homophily(g)

    intra_path_len_v1["orig"], inter_path_len_v1["orig"], intra_path_len_v2["orig"], inter_path_len_v2["orig"] = AvgShortPathLen(g)
    print("Node Homophily: ", node_homo["orig"])
    print("Edge Homophily: ", edge_homo["orig"])
    print("Improved Homophily: ", improved_homo["orig"])
    print("Adjusted Homophily: ", Adj_Homo["orig"])
    print("Aggregation Homophily: ", agg_homo["orig"])
    print("Original Graph ", intra_path_len_v1["orig"], inter_path_len_v1["orig"], intra_path_len_v2["orig"], inter_path_len_v2["orig"])

    if args.UNIBI=="uni":
        for seed in rand_seed_list:
            print("Uni ", seed)
            g, clusters, ClusterSuper, Supernodes = SpectralHighway(g_dict["orig"], data_dict[f"orig"], args, num_classes)
            datasuni = Unigram(g, data_dict[f"orig"], args, num_classes, clusters, ClusterSuper, Supernodes, seed)
        
            g = nx.Graph()
            g.add_nodes_from(range(datasuni.num_nodes))
            g.add_edges_from(datasuni.edge_index.T.numpy())

            for i, node in enumerate(g.nodes()):
                g.nodes[node]['x'] = datasuni.x[i].tolist()
                g.nodes[node]['y'] = datasuni.y[i].item()
            if datasuni.edge_attr is not None:
                for edge, features in zip(g.edges, datasuni.edge_attr):
                    g.edges[edge]['attr'] = features.item()
        
            edge_index = datasuni.edge_index.numpy()
            edge_index = datasuni.edge_index
            num_nodes = datasuni.x.size(0)
            adjacency_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))
            adjacency_matrix = adjacency_matrix.to_dense()

            agg_homo[f"uni_{seed}"] = aggregation_homophily(datasuni.x, adjacency_matrix, datasuni.y)
            print("Aggregation Homophily: ", agg_homo[f"uni_{seed}"])

            node_homo[f"uni_{seed}"] = node_homophily(g)
            print("Node Homophily: ", node_homo[f"uni_{seed}"])
    
            edge_homo[f"uni_{seed}"] = edge_homophily(g)
            print("Edge Homophily: ", edge_homo[f"uni_{seed}"])
        
            Adj_Homo[f"uni_{seed}"] = adjusted_homophily(g, edge_homo[f"uni_{seed}"])
            print("Adjusted Homophily: ", Adj_Homo[f"uni_{seed}"])
        
            improved_homo[f"uni_{seed}"] = improved_homophily(g)
            print("Improved Homophily: ", improved_homo[f"uni_{seed}"])
        
            g_dict[f"uni_{seed}"] = copy.deepcopy(g)
            data_dict[f"uni_{seed}"] = copy.deepcopy(datasuni)
            intra_path_len_v1[f"uni_{seed}"], inter_path_len_v1[f"uni_{seed}"], intra_path_len_v2[f"uni_{seed}"], inter_path_len_v2[f"uni_{seed}"] = AvgShortPathLen(g)
        
        agg_homo["uni_avg"] = 0
        node_homo["uni_avg"] = 0
        edge_homo["uni_avg"] = 0
        Adj_Homo["uni_avg"] = 0
        improved_homo["uni_avg"] = 0

        for seed in rand_seed_list:
            agg_homo["uni_avg"] += agg_homo[f"uni_{seed}"]
            node_homo["uni_avg"] += node_homo[f"uni_{seed}"]
            edge_homo["uni_avg"] += edge_homo[f"uni_{seed}"]
            Adj_Homo["uni_avg"] += Adj_Homo[f"uni_{seed}"]
            improved_homo["uni_avg"] += improved_homo[f"uni_{seed}"]
        
        agg_homo["uni_avg"] = agg_homo["uni_avg"]/5
        node_homo["uni_avg"] = node_homo["uni_avg"]/5
        edge_homo["uni_avg"] = edge_homo["uni_avg"]/5
        Adj_Homo["uni_avg"] = Adj_Homo["uni_avg"]/5
        improved_homo["uni_avg"] = improved_homo["uni_avg"]/5

        intra_path_len_v1["uni_avg"]=0
        inter_path_len_v1["uni_avg"]=0
        intra_path_len_v2["uni_avg"]=0
        inter_path_len_v2["uni_avg"]=0

        for seed in rand_seed_list:
            intra_path_len_v1["uni_avg"] += intra_path_len_v1[f"uni_{seed}"]
            inter_path_len_v1["uni_avg"] += inter_path_len_v1[f"uni_{seed}"]
            intra_path_len_v2["uni_avg"] += intra_path_len_v2[f"uni_{seed}"]
            inter_path_len_v2["uni_avg"] += inter_path_len_v2[f"uni_{seed}"]

        intra_path_len_v1["uni_avg"] = intra_path_len_v1["uni_avg"]/5
        inter_path_len_v1["uni_avg"] = inter_path_len_v1["uni_avg"]/5
        intra_path_len_v2["uni_avg"] = intra_path_len_v2["uni_avg"]/5
        inter_path_len_v2["uni_avg"] = inter_path_len_v2["uni_avg"]/5


        FileWriteOther("Homophily/NodeHomophily.csv", node_homo, args)
        FileWriteOther("Homophily/EdgeHomophily.csv", edge_homo, args)
        FileWriteOther("Homophily/ImprovedHomophily.csv", improved_homo, args)
        FileWriteOther("Homophily/AdjustedHomophily.csv", Adj_Homo, args)
        FileWriteOther("Homophily/AggregationHomophily.csv", agg_homo, args)
        print(intra_path_len_v1, inter_path_len_v1, intra_path_len_v2, inter_path_len_v2)
        FileWrite("Homophily/AvgShortPathLens.csv", intra_path_len_v1, inter_path_len_v1, intra_path_len_v2, inter_path_len_v2, args)

    elif args.UNIBI=="bi":
        for seed in rand_seed_list:
            print("Bi ", seed)
            g, clusters, ClusterSuper, Supernodes = SpectralHighway(g_dict["orig"], data_dict[f"orig"], args, num_classes)
            datasbi = Bigram(g, data_dict[f"orig"], args, num_classes, clusters, ClusterSuper, Supernodes, seed)
        
            g = nx.Graph()
            g.add_nodes_from(range(datasbi.num_nodes))
            g.add_edges_from(datasbi.edge_index.T.numpy())

            for i, node in enumerate(g.nodes()):
                g.nodes[node]['x'] = datasbi.x[i].tolist()
                g.nodes[node]['y'] = datasbi.y[i].item()
            if datasbi.edge_attr is not None:
                for edge, features in zip(g.edges, datasbi.edge_attr):
                    g.edges[edge]['attr'] = features.item()

            edge_index = datasbi.edge_index.numpy()
            edge_index = datasbi.edge_index
            num_nodes = datasbi.x.size(0)
            adjacency_matrix = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (num_nodes, num_nodes))
            adjacency_matrix = adjacency_matrix.to_dense()

            agg_homo[f"bi_{seed}"] = aggregation_homophily(datasbi.x, adjacency_matrix, datasbi.y)
            print("Aggregation Homophily: ", agg_homo[f"bi_{seed}"])

            node_homo[f"bi_{seed}"] = node_homophily(g)
            print("Node Homophily: ", node_homo[f"bi_{seed}"])
    
            edge_homo[f"bi_{seed}"] = edge_homophily(g)
            print("Edge Homophily: ", edge_homo[f"bi_{seed}"])
        
            Adj_Homo[f"bi_{seed}"] = adjusted_homophily(g, edge_homo[f"bi_{seed}"])
            print("Adjusted Homophily: ", Adj_Homo[f"bi_{seed}"])
        
            improved_homo[f"bi_{seed}"] = improved_homophily(g)
            print("Improved Homophily: ", improved_homo[f"bi_{seed}"])
        
            g_dict[f"bi_{seed}"] = copy.deepcopy(g)
            data_dict[f"bi_{seed}"] = copy.deepcopy(datasbi)  
            intra_path_len_v1[f"bi_{seed}"], inter_path_len_v1[f"bi_{seed}"], intra_path_len_v2[f"bi_{seed}"], inter_path_len_v2[f"bi_{seed}"] = AvgShortPathLen(g)     

        agg_homo["bi_avg"] = 0
        node_homo["bi_avg"] = 0
        edge_homo["bi_avg"] = 0
        Adj_Homo["bi_avg"] = 0
        improved_homo["bi_avg"] = 0
    
        for seed in rand_seed_list:
            agg_homo["bi_avg"] += agg_homo[f"bi_{seed}"]
            node_homo["bi_avg"] += node_homo[f"bi_{seed}"]
            edge_homo["bi_avg"] += edge_homo[f"bi_{seed}"]
            Adj_Homo["bi_avg"] += Adj_Homo[f"bi_{seed}"]
            improved_homo["bi_avg"] += improved_homo[f"bi_{seed}"]

        agg_homo["bi_avg"] = agg_homo["bi_avg"]/5
        node_homo["bi_avg"] = node_homo["bi_avg"]/5
        edge_homo["bi_avg"] = edge_homo["bi_avg"]/5
        Adj_Homo["bi_avg"] = Adj_Homo["bi_avg"]/5
        improved_homo["bi_avg"] = improved_homo["bi_avg"]/5

        intra_path_len_v1["bi_avg"]=0
        inter_path_len_v1["bi_avg"]=0
        intra_path_len_v2["bi_avg"]=0
        inter_path_len_v2["bi_avg"]=0
        for seed in rand_seed_list:
            intra_path_len_v1["bi_avg"] += intra_path_len_v1[f"bi_{seed}"]
            intra_path_len_v2["bi_avg"] += intra_path_len_v2[f"bi_{seed}"]
            inter_path_len_v1["bi_avg"] += inter_path_len_v1[f"bi_{seed}"]
            inter_path_len_v2["bi_avg"] += inter_path_len_v2[f"bi_{seed}"]


        intra_path_len_v1["bi_avg"] = intra_path_len_v1["bi_avg"]/5
        intra_path_len_v2["bi_avg"] = intra_path_len_v2["bi_avg"]/5
        inter_path_len_v1["bi_avg"] = inter_path_len_v1["bi_avg"]/5
        inter_path_len_v2["bi_avg"] = inter_path_len_v2["bi_avg"]/5

        FileWriteOther("Homophily/NodeHomophily.csv", node_homo, args)
        FileWriteOther("Homophily/EdgeHomophily.csv", edge_homo, args)
        FileWriteOther("Homophily/ImprovedHomophily.csv", improved_homo, args)
        FileWriteOther("Homophily/AdjustedHomophily.csv", Adj_Homo, args)
        FileWriteOther("Homophily/AggregationHomophily.csv", agg_homo, args)
        print(intra_path_len_v1, inter_path_len_v1, intra_path_len_v2, inter_path_len_v2)
        FileWrite("Homophily/AvgShortPathLens.csv", intra_path_len_v1, inter_path_len_v1, intra_path_len_v2, inter_path_len_v2, args)

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
                        default=30,
                        help='Number of partitions.')
    parser.add_argument('--rank',
                        type=str,
                        default='div',
                        choices = ['page', 'div'],
                        help='Manner of connections to add.')
    parser.add_argument('--connect_type',
                        type=str,
                        default='global',
                        choices = ['rand', 'global', 'local', 'g&l'],
                        help='Manner of connections to add.')
    parser.add_argument('--mode',
                        type=str,
                        default='lmh',
                        choices = ['low', 'mid', 'high', 'lmh'],
                        help='Which ordering of connections to add.')
    parser.add_argument('--min_num_connect',
                        type=int,
                        default=3,
                        help='Minimum Number of connections to add.')
    parser.add_argument('--percent_connect',
                        type=float,
                        default=0.6,
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
                        default='dagnn',
                        choices=['pgnn', 'mlp', 'gcn', 'gcnii', 'linkx', 'cheb', 'sgc', 'gat', 'gatv2', 'sage', 'jk', 'appnp', 'gprgnn', 'dagnn', 'aero', 'bern'],
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
    parser.add_argument('--UNIBI',
                        type=str,
                        choices=['uni', 'bi'])
    parser.add_argument('--dprate',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(get_args())
