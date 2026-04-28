import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, GATConv, GATv2Conv, APPNP, JumpingKnowledge, SAGEConv, GCN2Conv, LINKX, ChebConv
from torch.nn import ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, ones, zeros
from src.pgnn_conv import pGNNConv
from src.gpr_conv import GPR_prop
#from src.wrgat import *
#from src.wrgcn import *
from torch_sparse import coalesce
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch.nn.modules.module import Module
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix

import math
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch_scatter import scatter_add
from torch_geometric.utils import get_laplacian
from scipy.special import comb
from torch_sparse import sum as sparsesum
from torch_sparse import mul
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import imp
import torch.nn.init as init
#from layers.ChebClenshawConv import ChebConv

def pmlp_gcn_conv(h, edge_index):
    N = h.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    
    src, dst = edge_index
    deg = degree(dst, num_nodes=N)

    deg_src = deg[src].pow(-0.5) 
    deg_src.masked_fill_(deg_src == float('inf'), 0)
    deg_dst = deg[dst].pow(-0.5)
    deg_dst.masked_fill_(deg_dst == float('inf'), 0)
    edge_weight = deg_src * deg_dst

    a = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).t()
    h_prime = a @ h 
    return h_prime

class PMLP_GCN(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_node, num_layers=2):
        super(PMLP_GCN, self).__init__()
        self.dropout = dropout
        self.num_node = num_node
        self.num_layers = num_layers
        self.ff_bias = True  

        self.bns = nn.BatchNorm1d(hidden_channels, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, edge_weight):
        for i in range(self.num_layers):
            x = x @ self.fcs[i].weight.t() 
            if not self.training:    
                x = pmlp_gcn_conv(x, edge_index) 
            if self.ff_bias: x = x + self.fcs[i].bias
            if i != self.num_layers - 1:
                x = self.activation(self.bns(x))
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

class PMLP_APPNP(nn.Module): #residual connection
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_node, num_layers=2, num_mps=2):
        super(PMLP_APPNP, self).__init__()
        self.dropout = dropout
        self.num_node = num_node
        self.num_layers = num_layers
        self.num_mps = num_mps
        self.ff_bias = True

        self.bns = nn.BatchNorm1d(hidden_channels, eps=1e-10, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_channels, hidden_channels, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(nn.Linear(hidden_channels, hidden_channels, bias=self.ff_bias)) #1s
        self.fcs.append(nn.Linear(hidden_channels, out_channels, bias=self.ff_bias)) #1
        self.reset_parameters()
    

    def reset_parameters(self):
        for mlp in self.fcs: 
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x, edge_index, edge_weight):
        for i in range(self.num_layers - 1):
            x = self.fcs[i](x) 
            x = self.activation(self.bns(x))
            x = F.dropout(x, p=self.dropout, training=self.training)    
        x = self.fcs[-1](x) 
        for i in range(self.num_mps):
            if not self.training: 
                x = pmlp_gcn_conv(x, edge_index)    
        return x

def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj

def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")

def Dir_get_conv(conv_type, input_dim, output_dim, alpha):
    if conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-sage":
        return DirSageConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gat":
        return DirGATConv(input_dim, output_dim, heads=1, alpha=alpha)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")

class DirGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_layers=2, dropout=0, conv_type="dir-gcn", jumping_knowledge=None,
        normalize=False, alpha=1 / 2, learn_alpha=False):
        super(DirGNN, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([Dir_get_conv(conv_type, num_features, output_dim, self.alpha)])
        else:
            self.convs = ModuleList([Dir_get_conv(conv_type, num_features, hidden_dim, self.alpha)])
            for _ in range(num_layers - 2):
                self.convs.append(Dir_get_conv(conv_type, hidden_dim, hidden_dim, self.alpha))
            self.convs.append(Dir_get_conv(conv_type, hidden_dim, output_dim, self.alpha))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index, edge_weight):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)

class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index, edge_weight):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(self.adj_t_norm @ x)


class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight):
        return (self.lin_self(x) + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(x, edge_index))


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads)
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        return (1 - self.alpha) * self.conv_src_to_dst(x, edge_index) + self.alpha * self.conv_dst_to_src(x, edge_index_t)

class GraphConvolutionACM(Module):

    def __init__(self, in_features, out_features, nnodes, model_type, output_layer = 0, variant = False, structure_info=0):
        super(GraphConvolutionACM, self).__init__()
        self.in_features, self.out_features, self.output_layer, self.model_type, self.structure_info, self.variant = in_features, out_features, output_layer, model_type, structure_info, variant
        self.att_low, self.att_high, self.att_mlp = 0,0,0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features).to(device)), Parameter(torch.FloatTensor(in_features, out_features).to(device)), Parameter(torch.FloatTensor(in_features, out_features).to(device))           
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(1*out_features, 1).to(device)), Parameter(torch.FloatTensor(1*out_features, 1).to(device)), Parameter(torch.FloatTensor(1*out_features, 1).to(device))
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = nn.LayerNorm(out_features), nn.LayerNorm(out_features), nn.LayerNorm(out_features)
        self.layer_norm_struc_low, self.layer_norm_struc_high = nn.LayerNorm(out_features), nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(torch.FloatTensor(1*out_features, 1).to(device))
        self.struc_low = Parameter(torch.FloatTensor(nnodes, out_features).to(device))
        if self.structure_info == 0:
            self.att_vec = Parameter(torch.FloatTensor(3, 3).to(device))
        else:
            self.att_vec = Parameter(torch.FloatTensor(4, 4).to(device))
        self.reset_parameters()

    def reset_parameters(self): 
        
        
        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt( self.att_vec_mlp.size(1))
        std_att_vec = 1. / math.sqrt( self.att_vec.size(1))
        
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.struc_low.data.uniform_(-stdv, stdv)

        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_struc_low.data.uniform_(-std_att, std_att)
        
        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)
        
        self.layer_norm_low.reset_parameters()
        self.layer_norm_high.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        self.layer_norm_struc_low.reset_parameters()
        self.layer_norm_struc_high.reset_parameters()
        

    def attention3(self, output_low, output_high, output_mlp): 
        T = 3
        if self.model_type == 'acmgcnp' or self.model_type == 'acmgcnpp':
            output_low, output_high, output_mlp = self.layer_norm_low(output_low), self.layer_norm_high(output_high), self.layer_norm_mlp(output_mlp)
        logits = torch.mm(torch.sigmoid(torch.cat([torch.mm((output_low), self.att_vec_low), torch.mm((output_high), self.att_vec_high), torch.mm((output_mlp), self.att_vec_mlp)],1)), self.att_vec)/T
        att = torch.softmax(logits,1)
        return att[:,0][:,None],att[:,1][:,None],att[:,2][:,None]
    
    def attention4(self, output_low, output_high, output_mlp, struc_low):
        T = 4
        if self.model_type == 'acmgcnp' or self.model_type == 'acmgcnpp':
            feature_concat = torch.cat([torch.mm(self.layer_norm_low(output_low), self.att_vec_low), torch.mm(self.layer_norm_high(output_high), self.att_vec_high), torch.mm(self.layer_norm_mlp(output_mlp), self.att_vec_mlp), torch.mm(self.layer_norm_struc_low(struc_low), self.att_struc_low)],1)
        else:
            feature_concat = torch.cat([torch.mm((output_low), self.att_vec_low), torch.mm((output_high), self.att_vec_high), torch.mm((output_mlp), self.att_vec_mlp), torch.mm((struc_low), self.att_struc_low)],1) 
        
        logits = torch.mm(torch.sigmoid(feature_concat), self.att_vec)/T
        
        att = torch.softmax(logits, 1)
        return att[:,0][:,None],att[:,1][:,None],att[:,2][:,None] ,att[:,3][:,None]


    def forward(self, input, adj_low, adj_high, adj_low_unnormalized):
        output = 0
        if self.model_type == 'mlp':
            output_mlp = (torch.mm(input, self.weight_mlp))
            return output_mlp
        elif self.model_type == 'sgc' or self.model_type == 'gcn':
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.model_type == 'acmsgc':
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            output_high = torch.spmm(adj_high,  torch.mm(input, self.weight_high)) 
            output_mlp = torch.mm(input, self.weight_mlp)
            
            self.att_low, self.att_high, self.att_mlp = self.attention3((output_low), (output_high), (output_mlp))
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)
        else:
            if self.variant:
                
                output_low = (torch.spmm(adj_low, F.relu(torch.mm(input, self.weight_low)))) 
                
                output_high = (torch.spmm(adj_high, F.relu(torch.mm(input, self.weight_high))))
                output_mlp = (F.relu(torch.mm(input, self.weight_mlp)))
                
            else:
                output_low = (F.relu(torch.spmm(adj_low, (torch.mm(input, self.weight_low)))))
                output_high = (F.relu(torch.spmm(adj_high, (torch.mm(input, self.weight_high)))))
                output_mlp = (F.relu(torch.mm(input, self.weight_mlp)))
            
            if self.model_type == 'acmgcn' or self.model_type == 'acmsnowball':
                self.att_low, self.att_high, self.att_mlp = self.attention3((output_low), (output_high), (output_mlp)) 
                return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)
            else:
                if self.structure_info:
                    output_struc_low = (F.relu(torch.mm(adj_low_unnormalized, self.struc_low)))
                    self.att_low, self.att_high, self.att_mlp, self.att_struc_vec_low = self.attention4((output_low), (output_high), (output_mlp), output_struc_low) 
                    return 1*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp + self.att_struc_vec_low*output_struc_low) 
                else:
                    self.att_low, self.att_high, self.att_mlp = self.attention3((output_low), (output_high), (output_mlp)) 
                    return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               

class MLPACM(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLPACM, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
            self.bns.append(nn.BatchNorm1d(out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        # x = self.bns[-1](x)
        return x

class GCNIIACMPP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, nnodes, dropout, structure_info=0, model_type='acmgcnpp', variant=True):
        super(GCNIIACMPP, self).__init__()
        if model_type =='acmgcnpp':
            self.mlpX = MLPACM(nfeat, nhid, nhid, num_layers=1, dropout=0)
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.structure_info, self.nlayers, self.nnodes = model_type, structure_info, nlayers, nnodes
        
        if self.model_type =='acmgcn' or self.model_type =='acmgcnp' or self.model_type =='acmgcnpp':
            self.gcns.append(GraphConvolutionACM(nfeat, nhid, nnodes, model_type = model_type, variant = variant,  structure_info = structure_info))
            self.gcns.append(GraphConvolutionACM(1*nhid, nclass, nnodes, model_type = model_type, output_layer=1, variant = variant, structure_info = structure_info))
        elif self.model_type =='acmsgc':
            self.gcns.append(GraphConvolution(nfeat, nclass, model_type = model_type))
        elif self.model_type =='acmsnowball':
            for k in range(nlayers):
                self.gcns.append(GraphConvolutionACM(k * nhid + nfeat, nhid, model_type = model_type, variant = variant))
            self.gcns.append(GraphConvolutionACM(nlayers * nhid + nfeat, nclass, model_type = model_type, variant = variant))
        self.dropout = dropout
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fea_param, self.xX_param  = Parameter(torch.FloatTensor(1,1).to(device)), Parameter(torch.FloatTensor(1,1).to(device))          
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.model_type =='acmgcnpp':
            self.mlpX.reset_parameters()
        else:
            pass

    def forward(self, x, adj_low, adj_high, adj_low_unnormalized):

        device = x.device

        if self.model_type =='acmgcn' or self.model_type =='acmsgc' or self.model_type =='acmsnowball' or self.model_type =='acmgcnp' or self.model_type =='acmgcnpp':

            x = F.dropout(x, self.dropout, training=self.training)
            if self.model_type =='acmgcnpp':
                xX = F.dropout(F.relu(self.mlpX(x, input_tensor=True)), self.dropout, training=self.training)
        if self.model_type =='acmsnowball':
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(F.dropout(F.relu(layer(x, adj_low, adj_high)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(F.dropout(F.relu(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj_low, adj_high)), self.dropout, training=self.training))
            return self.gcns[-1](torch.cat([x] + list_output_blocks, 1), adj_low, adj_high)

        fea1 = (self.gcns[0](x, adj_low, adj_high, adj_low_unnormalized))
        
        if  self.model_type =='acmgcn' or self.model_type =='acmgcnp' or self.model_type =='acmgcnpp': 
            
            fea1 = F.dropout((F.relu(fea1)), self.dropout, training=self.training)
            
            if self.model_type =='acmgcnpp':
                fea2 = self.gcns[1](fea1+xX, adj_low, adj_high, adj_low_unnormalized)
            else:
                fea2 = self.gcns[1](fea1, adj_low, adj_high, adj_low_unnormalized)
        return fea2

class GCNACMPP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers, nnodes, dropout, structure_info=0, model_type='acmgcnpp', variant=False):
        super(GCNACMPP, self).__init__()
        if model_type =='acmgcnpp':
            self.mlpX = MLPACM(nfeat, nhid, nhid, num_layers=1, dropout=0)
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.structure_info, self.nlayers, self.nnodes = model_type, structure_info, nlayers, nnodes
        
        if self.model_type =='acmgcn' or self.model_type =='acmgcnp' or self.model_type =='acmgcnpp':
            self.gcns.append(GraphConvolutionACM(nfeat, nhid, nnodes, model_type = model_type, variant = variant,  structure_info = structure_info))
            self.gcns.append(GraphConvolutionACM(1*nhid, nclass, nnodes, model_type = model_type, output_layer=1, variant = variant, structure_info = structure_info))
        elif self.model_type =='acmsgc':
            self.gcns.append(GraphConvolution(nfeat, nclass, model_type = model_type))
        elif self.model_type =='acmsnowball':
            for k in range(nlayers):
                self.gcns.append(GraphConvolutionACM(k * nhid + nfeat, nhid, model_type = model_type, variant = variant))
            self.gcns.append(GraphConvolutionACM(nlayers * nhid + nfeat, nclass, model_type = model_type, variant = variant))
        self.dropout = dropout
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fea_param, self.xX_param  = Parameter(torch.FloatTensor(1,1).to(device)), Parameter(torch.FloatTensor(1,1).to(device))          
        self.reset_parameters()
        
    def reset_parameters(self):
        if self.model_type =='acmgcnpp':
            self.mlpX.reset_parameters()
        else:
            pass

    def forward(self, x, adj_low, adj_high, adj_low_unnormalized):

        device = x.device

        if self.model_type =='acmgcn' or self.model_type =='acmsgc' or self.model_type =='acmsnowball' or self.model_type =='acmgcnp' or self.model_type =='acmgcnpp':

            x = F.dropout(x, self.dropout, training=self.training)
            if self.model_type =='acmgcnpp':
                xX = F.dropout(F.relu(self.mlpX(x, input_tensor=True)), self.dropout, training=self.training)
        if self.model_type =='acmsnowball':
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(F.dropout(F.relu(layer(x, adj_low, adj_high)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(F.dropout(F.relu(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj_low, adj_high)), self.dropout, training=self.training))
            return self.gcns[-1](torch.cat([x] + list_output_blocks, 1), adj_low, adj_high)

        fea1 = (self.gcns[0](x, adj_low, adj_high, adj_low_unnormalized))
        
        if  self.model_type =='acmgcn' or self.model_type =='acmgcnp' or self.model_type =='acmgcnpp': 
            
            fea1 = F.dropout((F.relu(fea1)), self.dropout, training=self.training)
            
            if self.model_type =='acmgcnpp':
                fea2 = self.gcns[1](fea1+xX, adj_low, adj_high, adj_low_unnormalized)
            else:
                fea2 = self.gcns[1](fea1, adj_low, adj_high, adj_low_unnormalized)
        return fea2

class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=16):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [N]
        # output: [N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000)/self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class SpecLayer(nn.Module):

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm='none'):
        super(SpecLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == 'none': 
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == 'layer':    # Arxiv
            self.norm = nn.LayerNorm(ncombines)
        elif norm == 'batch':  # Penn
            self.norm = nn.BatchNorm1d(ncombines)
        else:                  # Others
            self.norm = None 

    def forward(self, x):
        x = self.prop_dropout(x) * self.weight      # [N, m, d] * [1, m, d]
        x = torch.sum(x, dim=1)

        if self.norm is not None:
            x = self.norm(x)
            x = F.relu(x)

        return x


class Specformer(nn.Module):

    def __init__(self, in_channels, out_channels, nlayer=2, num_hid=16, num_heads=1,
                tran_dropout=0.0, feat_dropout=0.0, prop_dropout=0.0, norm='none'):
        super(Specformer, self).__init__()

        self.norm = norm
        self.out_channels = out_channels
        self.nlayer = nlayer
        self.num_heads = num_heads
        self.num_hid = num_hid
        
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_channels, num_hid),
            nn.ReLU(),
            nn.Linear(num_hid, out_channels),
        )

        # for arxiv & penn
        self.linear_encoder = nn.Linear(out_channels, num_hid)
        self.classify = nn.Linear(num_hid, in_channels)

        self.eig_encoder = SineEncoding(num_hid)
        self.decoder = nn.Linear(num_hid, num_heads)

        self.mha_norm = nn.LayerNorm(num_hid)
        self.ffn_norm = nn.LayerNorm(num_hid)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(num_hid, num_heads, tran_dropout)
        self.ffn = FeedForwardNetwork(num_hid, num_hid, num_hid)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)
        if norm == 'none':
            self.layers = nn.ModuleList([SpecLayer(num_heads+1, out_channels, prop_dropout, norm=norm) for i in range(nlayer)])
        else:
            self.layers = nn.ModuleList([SpecLayer(num_heads+1, num_hid, prop_dropout, norm=norm) for i in range(nlayer)])
        

    def forward(self, x, e, u):
        N = e.size
        ut = u.permute(1, 0)

        if self.norm == 'none':
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        eig = self.eig_encoder(e)   # [N, d]

        mha_eig = self.mha_norm(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        new_e = self.decoder(eig)   # [N, m]

        for conv in self.layers:
            basic_feats = [h]
            utx = ut @ h
            for i in range(self.num_heads):
                basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))  # [N, d]
            basic_feats = torch.stack(basic_feats, axis=1)                # [N, m, d]
            h = conv(basic_feats)

        if self.norm == 'none':
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h

class Bern_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index,edge_weight=None):
        TEMP=F.relu(self.temp)

        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
        #2I-L
        edge_index2, norm2=add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
            x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
            tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
            x=tmp[self.K-i-1]
            x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
            for j in range(i):
                x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

            out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x
        return out
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class BernNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K, hidden, dropout, dprate):
        
        """
        BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation
        Datasets 
        Linear layer learning rate
        Propagation layer learning rate
        Hidden  dimension
        Propagation layer dropout
        Linear layer dropout
        K
        Weight  decay
        Cora 0.01 0.01 64 0.0 0.5 10 0.0005
        CiteSeer 0.01 0.01 64 0.5 0.5 10 0.0005
        PubMed 0.01 0.01 64 0.0 0.5 10 0.0
        Computers 0.01 0.05 64 0.6 0.5 10 0.0005
        Photo 0.01 0.01 64 0.5 0.5 10 0.0005
        Chameleon 0.05 0.01 64 0.7 0.5 10 0.0
        Actor 0.05 0.01 64 0.9 0.5 10 0.0
        Squirrel 0.05 0.01 64 0.6 0.5 10 0.0
        Texas 0.05 0.002 64 0.5 0.5 10 0.0005
        Cornell 0.05 0.001 64 0.5 0.5 10 0.0005
        """

        super(BernNet, self).__init__()
        self.lin1 = Linear(in_channels, hidden)
        self.lin2 = Linear(hidden, out_channels)
        self.m = torch.nn.BatchNorm1d(out_channels)
        self.prop1 = Bern_prop(K)

        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

class Dense(nn.Module):

    def __init__(self, in_features, out_features, bias='none'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
            
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))        
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):    
        #print("Shape of input tensor:", input.shape)
        #print("Shape of weight tensor:", self.weight.shape)        
        output = torch.mm(input, self.weight)
        #output = input * self.weight
        output = self.bias(output)
        if self.in_features == self.out_features:
            output = output + input
        return output

class MLPPoly(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, bias):
        super(MLPPoly, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(nfeat, nhidden, bias))
        for _ in range(nlayers-2):
            self.fcs.append(Dense(nhidden, nhidden, bias))
        self.fcs.append(Dense(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)        
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return x

class Combination(nn.Module):
    '''
    A mod combination the bases of polynomial filters.
    Args:
        channels (int): number of feature channels.
        level (int): number of bases to combine.
        sole (bool): whether or not use the same filter for all output channels.
    '''
    def __init__(self, channels, level, dropout, num_nodes, sole=False):
        super().__init__()
        self.dropout = dropout
        self.K=level
        self.channels = channels
        self.num_nodes = num_nodes
        self.comb_weight = nn.Parameter(torch.ones((channels, level)))
        self.reset_parameters()            

    def reset_parameters(self):
        bound = 1.0/self.K
        #TEMP = np.random.uniform(bound, bound, self.K)  
        #self.comb_weight=nn.Parameter(torch.FloatTensor(TEMP).view(self.channels,self.K, 1))

        TEMP = torch.full((self.channels, self.K), bound)  # Adjust based on desired dimensions
        self.comb_weight = nn.Parameter(TEMP)  # Adding a singleton dimension if necessary

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        #print(x.shape, self.comb_weight.shape)
        #x = torch.matmul(x, self.comb_weight)
        #x = x * self.comb_weight
        x = torch.mm(x, self.comb_weight)
        #x = torch.sum(x, dim=1)
        return x

class GFK(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, dropoutC, dropoutM, num_nodes, level=10, bias='none', sole=True, nlayers=2):
        super(GFK, self).__init__()
        self.in_channels = in_channels
        self.level = level + 1
        self.comb = Combination(in_channels, self.level, dropoutC, sole, num_nodes)
        self.mlp = MLPPoly(self.level, nlayers, hid_channels, out_channels, dropoutM, bias)


    def forward(self, x, edge_index, edge_weight):
        x = self.comb(x)                     
        x = self.mlp(x)
        return x

"""
class WRGCN(torch.nn.Module):
	def __init__(self, num_features, num_classes, num_relations=10, dims=16, drop=0, root=True):
		super(WRGCN, self).__init__()
		self.conv1 = WeightedRGCNConv(num_features, dims, num_relations=num_relations, num_bases=None, root_weight=root)
		self.conv2 = WeightedRGCNConv(dims, num_classes, num_relations=num_relations, num_bases=None, root_weight=root)

		self.drop = torch.nn.Dropout(p=drop)
	def forward(self, x, edge_index, edge_weight, edge_color):

		x = F.relu(self.conv1(x, edge_index,edge_weight=edge_weight,edge_type=edge_color))
		x = self.drop(x)

		x = self.conv2(x, edge_index,edge_weight=edge_weight, edge_type=edge_color)

		return F.log_softmax(x, dim=1)


class WRGAT(torch.nn.Module):
	def __init__(self, num_features, num_classes, num_relations=10, dims=16, drop=0, root=True):
		super(WRGAT, self).__init__()
		self.conv1 = WeightedRGATConv(num_features, dims, num_relations=num_relations, root_weight=root)
		self.conv2 = WeightedRGATConv(dims, num_classes, num_relations=num_relations, root_weight=root)

		self.drop = torch.nn.Dropout(p=drop)
	def forward(self, x, edge_index, edge_weight, edge_color):

		x = F.relu(self.conv1(x, edge_index,edge_weight=edge_weight,edge_type=edge_color))
		x = self.drop(x)

		x = self.conv2(x, edge_index,edge_weight=edge_weight, edge_type=edge_color)

		return F.log_softmax(x, dim=1)
"""

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class Prop(MessagePassing):
    def __init__(self, num_classes, K, bias=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm(edge_index, edge_weight, x.size(0), dtype=x.dtype)


        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()

    
class DAGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden, K, dropout):

        """
        Towards Deeper Graph Neural Networks
        For our DAGNN,
        we tune the following hyperparameters: (1) k ∈ {5, 10, 20}, (2) weight
        decay ∈ {0, 2e-2, 5e-3, 5e-4, 5e-5}, and (3) dropout rate ∈ {0.5, 0.8}.
        """

        super(DAGNN, self).__init__()
        self.lin1 = Linear(in_channels, hidden)
        self.lin2 = Linear(hidden, out_channels)
        self.prop = Prop(out_channels, K)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)

class AERO_GNN(MessagePassing):

    def __init__(self, in_channels, hid_channels, out_channels, num_heads, num_nodes, K, dropout):
        super().__init__(node_dim=0, )

        """
        Towards Deep Attention in Graph Neural Networks: Problems and Remedies
        WDft ∈ {4e−2, 2e−2, 1e−2, 5e−3, 1e−3, 5e−4, 1e−4}
        WDprop ∈ {2e − 2, 1e − 2, 5e − 3, 1e − 3, 5e − 4, 1e − 4}
        dropout ∈ {0.5, 0.6, 0.7, 0.8}
        kmax ∈ {4, 8, 16, 32}
        weight decay λ ∈ {0.25, 0.5, 1.0}
        """

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.hid_channels = hid_channels
        self.hid_channels_ = self.heads * self.hid_channels
        self.K = K
        self.lambd=1
        #Paper by default has lambda = 1
                
        self.setup_layers()
        self.reset_parameters()


    def setup_layers(self):

        self.dropout = nn.Dropout(self.dropout)
        self.elu = nn.ELU()
        self.softplus = nn.Softplus()

        self.dense_lins = nn.ModuleList()
        self.atts = nn.ParameterList()
        self.hop_atts = nn.ParameterList()
        self.hop_biases = nn.ParameterList()
        self.decay_weights = []
        #Our other GNNs have 2 layers

        self.dense_lins.append(Linear(self.in_channels, self.hid_channels_, bias=True, weight_initializer='glorot'))
        for _ in range(4): self.dense_lins.append(Linear(self.hid_channels_, self.hid_channels_, bias=True, weight_initializer='glorot'))
        self.dense_lins.append(Linear(self.hid_channels_, self.out_channels, bias=True, weight_initializer='glorot'))

        for k in range(self.K + 1): 
            self.atts.append(nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels)))
            self.hop_atts.append(nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels*2)))
            self.hop_biases.append(nn.Parameter(torch.Tensor(1, self.heads)))
            self.decay_weights.append( np.log((self.lambd / (k+1)) + (1 + 1e-6)) )
        self.hop_atts[0]=nn.Parameter(torch.Tensor(1, self.heads, self.hid_channels))
        self.atts = self.atts[1:]


    def reset_parameters(self):
        
        for lin in self.dense_lins: lin.reset_parameters()
        for att in self.atts: glorot(att) 
        for att in self.hop_atts: glorot(att) 
        for bias in self.hop_biases: ones(bias) 


    def hid_feat_init(self, x):
        
        x = self.dropout(x)
        x = self.dense_lins[0](x)

        for l in range(4):
            x = self.elu(x)
            x = self.dropout(x)
            x = self.dense_lins[l+1](x)
        
        return x


    def aero_propagate(self, h, edge_index):
        
        self.k = 0
        h = h.view(-1, self.heads, self.hid_channels)
        g = self.hop_att_pred(h, z_scale=None)
        z = h * g
        z_scale = z * self.decay_weights[self.k]

        for k in range(self.K):

            self.k = k+1
            h = self.propagate(edge_index, x = h, z_scale = z_scale)            
            g = self.hop_att_pred(h, z_scale)
            z += h * g
            z_scale = z * self.decay_weights[self.k]
                
        return z


    def node_classifier(self, z):
        
        z = z.view(-1, self.heads * self.hid_channels)
        z = self.elu(z)
        #z = self.dropout(z)
        z = self.dense_lins[-1](z)
        
        return z


    def forward(self, x, edge_index, edge_weight):
        
        h0 = self.hid_feat_init(x)
        z_k_max = self.aero_propagate(h0, edge_index)
        z_star =  self.node_classifier(z_k_max)

        return z_star


    def hop_att_pred(self, h, z_scale):

        if z_scale is None: 
            x = h
        else:
            x = torch.cat((h, z_scale), dim=-1)

        g = x.view(-1, self.heads, int(x.shape[-1]))
        g = self.elu(g)
        g = (self.hop_atts[self.k] * g).sum(dim=-1) + self.hop_biases[self.k]
        
        return g.unsqueeze(-1)


    def edge_att_pred(self, z_scale_i, z_scale_j, edge_index):
        
        # edge attention (alpha_check_ij)
        a_ij = z_scale_i + z_scale_j
        a_ij = self.elu(a_ij)
        a_ij = (self.atts[self.k-1] * a_ij).sum(dim=-1)
        a_ij = self.softplus(a_ij) + 1e-6

        # symmetric normalization (alpha_ij)
        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(a_ij, col, dim=0, dim_size=self.num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        a_ij = deg_inv_sqrt[row] * a_ij * deg_inv_sqrt[col]        

        return a_ij


    def message(self, edge_index, x_j, z_scale_i, z_scale_j):
        a = self.edge_att_pred(z_scale_i, z_scale_j, edge_index)
        return a.unsqueeze(-1) * x_j



class LINKX(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, out_channels, num_hid, num_nodes, num_layers=2, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()	
        self.mlpA = MLPLinkx(num_nodes, num_hid, num_hid, init_layers_A, dropout=0)
        self.mlpX = MLPLinkx(in_channels, num_hid, num_hid, init_layers_X, dropout=0)
        self.W = nn.Linear(2*num_hid, num_hid)
        self.mlp_final = MLPLinkx(num_hid, num_hid, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	

    def forward(self, x, edge_index, edge_weight):	
        m = x.size(0)
        feat_dim = x.size(1)	
        row, col = edge_index
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()

        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(x, input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        return F.log_softmax(x, dim=1)

class MLPLinkx(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, num_hid, out_channels, num_layers,
                 dropout=.5):
        super(MLPLinkx, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, num_hid))
            self.bns.append(nn.BatchNorm1d(num_hid))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(num_hid, num_hid))
                self.bns.append(nn.BatchNorm1d(num_hid))
            self.lins.append(nn.Linear(num_hid, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class pGNNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_hid=16, 
                 mu=0.1,
                 p=2,
                 K=2,
                 dropout=0.5,
                 cached=True):
        super(pGNNNet, self).__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = pGNNConv(num_hid, out_channels, mu, p, K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)        
        return F.log_softmax(x, dim=1)


class MLPNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=16,
                 dropout=0.5):
        super(MLPNet, self).__init__()
        self.dropout = dropout
        self.layer1 = torch.nn.Linear(in_channels, num_hid)
        self.layer2 = torch.nn.Linear(num_hid, out_channels)

    def forward(self, x, edge_index=None, edge_weight=None):
        x = torch.relu(self.layer1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x)
        return F.log_softmax(x, dim=1)


class GCNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=16,
                 dropout=0.5,
                 cached=True):
        super(GCNNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid, cached=cached)
        self.conv2 = GCNConv(num_hid, out_channels, cached=cached)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output


class GCN2Net(nn.Module):
    def __init__(self, in_channels, out_channels, nlayers=2, num_hid=16, dropout=0.5, lamda=1, alpha=0.1, variant=False):
        super(GCN2Net, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(num_hid, num_hid,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, num_hid))
        self.fcs.append(nn.Linear(num_hid, out_channels))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, edge_index, edge_weight):
        _layers = []
        num_nodes = x.size(0)
        adj = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.shape[1]).to(x.device), torch.Size([num_nodes, num_nodes]))

        # Coalesce the adjacency matrix to ensure that each edge appears only once
        #adj = coalesce(adj, None, num_nodes, num_nodes)[0]
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)


class GCN_Encoder(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 num_hid=16):
        super(GCN_Encoder, self).__init__()
        self.conv = GCNConv(in_channels, num_hid, cached=True)
        self.prelu = torch.nn.PReLU(num_hid)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x


class SGCNet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 K=2,
                 cached=True):
        super(SGCNet, self).__init__()
        self.conv1 = SGConv(in_channels, out_channels, K=K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=8,
                 num_heads=8,
                 dropout=0.6,
                 concat=False):

        super(GATNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, num_hid, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(num_heads * num_hid, out_channels, heads=1, concat=concat, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=-1)

class GATv2Net(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_hid=8,
                 num_heads=8,
                 dropout=0.6,
                 concat=False):

        super(GATv2Net, self).__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, num_hid, heads=num_heads, dropout=dropout)
        self.conv2 = GATv2Conv(num_heads * num_hid, out_channels, heads=1, concat=concat, dropout=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=-1)

class GraphSAGENet(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_hid=16,
                 dropout=0.5):
        
        super(GraphSAGENet, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, num_hid)
        self.conv2 = SAGEConv(num_hid, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)  

class JKNet(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0,
                 num_layes=4,
                 dropout=0.5):
        super(JKNet, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, num_hid)
        self.conv2 = GCNConv(num_hid, num_hid)
        self.lin1 = torch.nn.Linear(num_hid, out_channels)
        self.one_step = APPNP(K=K, alpha=alpha)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=num_hid,
                                   num_layers=num_layes)

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index, edge_weight)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)


class APPNPNet(torch.nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels,
                 num_hid=16,
                 K=1,
                 alpha=0.1,
                 dropout=0.5):
        super(APPNPNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GPRGNNNet(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 num_hid,
                 ppnp,
                 K=10,
                 alpha=0.1,
                 Init='PPR',
                 Gamma=None,
                 dprate=0.5,
                 dropout=0.5):
        super(GPRGNNNet, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_channels)

        if ppnp == 'PPNP':
            self.prop1 = APPNP(K, alpha)
        elif ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, edge_weight)
            return F.log_softmax(x, dim=1)



