import torch
import argparse
import time
import scipy.sparse as sp
from numpy.linalg import eig, eigh
#import copy #Added

from data_proc_neur import *
from models import *
import torch_geometric.transforms as T
from OGB import *
#from src.build_multigraph import * #Added

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
        print(labels.type())
        data = Data(x=dataset.graph['node_feat'], y=labels, edge_index=dataset.graph['edge_index'])
        num_nodes = data.x.size(0)
        adj = sp.coo_matrix((torch.ones(data.edge_index.size(1)), (data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy())),
                        shape=(num_nodes, num_nodes))
        if args.model=='spec':
          e, u = eigen_decompositon(adj)
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          e = torch.tensor(e, dtype=torch.float).to(device)
          u = torch.tensor(u, dtype=torch.float).to(device)
        else:
          e = None
          u = None
        
        if args.model=='acmgcnpp' or args.model=='acmiigcnpp':
          adj_low_unnormalized = sparse_mx_to_torch_sparse_tensor(adj)
          adj_low = normalize_tensor(torch.eye(num_nodes) + adj_low_unnormalized.to_dense())
          adj_high = (torch.eye(num_nodes) - adj_low).to(device).to_sparse()
          adj_low = adj_low.to(device)
        else:
          adj_low = None
          adj_high = None
          adj_low_unnormalized = None
        #x = data.x
        num_features = dataset.graph['node_feat'].shape[1]
        classes = torch.unique(dataset.label)
        num_classes = classes.shape[0]
        num_nodes = dataset.graph['num_nodes']
        #print(num_features, num_classes,  num_nodes)
        shapes = (dataset.graph['num_nodes'], 1)
        data.train_mask = torch.zeros(shapes, dtype=torch.bool).squeeze()
        data.val_mask = torch.zeros(shapes, dtype=torch.bool).squeeze()
        data.test_mask = torch.zeros(shapes, dtype=torch.bool).squeeze()
        split_idx = dataset.get_idx_split(train_prop=args.train_rate, valid_prop=args.val_rate)
        #data.train_mask = split_idx['train']
        #data.val_mask = split_idx['valid']
        #data.test_mask = split_idx['test']
        num_train = int(len(data.y) / num_classes * args.train_rate)
        num_val = int(len(data.y) / num_classes * args.val_rate)
        
        data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, seed=2021, train_num_per_c=num_train, val_num_per_c=num_val)
        
        
    elif args.input in ['ogbn-arxiv', 'ogbn-mag', 'ogbn-products']:
        OGBNDataset(args, rand_seed=2021)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data, num_features, num_classes = load_data(args, rand_seed=2021)
        num_nodes = data.x.size(0)
        adj = sp.coo_matrix((torch.ones(data.edge_index.size(1)), (data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy())),
                        shape=(num_nodes, num_nodes))
        #print(adj)

        if args.model=='spec':
          e, u = eigen_decompositon(adj)
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          e = torch.tensor(e, dtype=torch.float).to(device)
          u = torch.tensor(u, dtype=torch.float).to(device)
        else:
          e = None
          u = None
        
        if args.model=='acmgcnpp' or args.model=='acmiigcnpp':
          adj_low_unnormalized = sparse_mx_to_torch_sparse_tensor(adj)
          adj_low = normalize_tensor(torch.eye(num_nodes) + adj_low_unnormalized.to_dense())
          adj_high = (torch.eye(num_nodes) - adj_low).to(device).to_sparse()
          adj_low = adj_low.to(device)
        else:
          adj_low = None
          adj_high = None
          adj_low_unnormalized = None
        #print("e: ", e, "u: ", u)
        #x = data.x

    if args.input not in ['ogbn-arxiv', 'ogbn-mag', 'ogbn-products']:
        #print(data, data.x, data.edge_index)
        num_nodes = data.x.size(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = []
        rand_seed_list = [0, 5, 66, 244, 2020]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_train = int(len(data.y) / num_classes * args.train_rate)
        num_val = int(len(data.y) / num_classes * args.val_rate)
        """
        #Added
        if args.model in ['wrgcn', 'wrgat']:
            copy = data
            data = build_pyg_struc_multigraph(data)
            data = filter_rels(data, args.K)
        """
        
        for run in range(args.runs):
        
            if args.input in ['fb100', 'twitch-gamer', 'arxiv-year']:
                data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, seed=rand_seed_list[run%5], train_num_per_c=num_train, val_num_per_c=num_val)
            else:
                data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, seed=rand_seed_list[run%5], train_num_per_c=num_train, val_num_per_c=num_val)

            #Added
            model = build_model(args, num_features, num_classes, num_nodes)
            model = model.to(device)
            data = data.to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        
            t1 = time.time()
            best_val_acc = test_acc = 0
            for epoch in range(1, args.epochs+1):
                train(model, optimizer, data, e, u, args, adj_low, adj_high, adj_low_unnormalized)
                train_acc, val_acc, tmp_test_acc = test(model, data, e, u, args, adj_low, adj_high, adj_low_unnormalized)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(epoch, train_acc, best_val_acc, test_acc))
            t2 = time.time()
            # print('{}, {}, Accuacy: {:.4f}, Time: {:.4f}'.format(args.model, args.input, test_acc, t2-t1))
            results.append(test_acc)
        results = 100 * torch.Tensor(results)
        print(results)
        print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f}')

        file = open('DAGNN.csv', 'a')
        file.write(args.input)
        file.write('\t')
        file.write(str(args.train_rate))
        file.write('\t')
        file.write(str(args.val_rate))
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
        file.write(f"{results.mean():.2f}")
        file.write('\t')
        file.write(f"{results.std():.2f}")
        file.write('\n')
        file.close()



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', 
                        type=str, 
                        default='cornell',                    
                        help='Input graph.')
    parser.add_argument('--train_rate', 
                        type=float, 
                        default=0.6,
                        help='Training rate.')
    parser.add_argument('--val_rate', 
                        type=float, 
                        default=0.2,
                        help='Validation rate.')
    parser.add_argument('--model',
                        type=str,
                        default='bern',
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
                        help='p')
    parser.add_argument('--K', 
                        type=int, 
                        default=10,
                        help='K, or r i.e., filter structure relation number')
    parser.add_argument('--num_heads', 
                        type=int, 
                        default=8,
                        help='Number of heads.')
    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.5,
                        help='alpha or eps')
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
