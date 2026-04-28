import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from ogb.nodeproppred import PygNodePropPredDataset
from models import *

# Set random seed for reproducibility
torch.manual_seed(42)
def build_model(args, num_features, num_classes):
    if args.model == 'pgnn':
        model = pGNNNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            mu=args.mu,
                            p=args.p,
                            K=args.K,
                            dropout=args.dropout)
    elif args.model == 'mlp':
        model = MLPNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'gcn':
        model = GCNNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'gcnii':
        model = GCN2Net(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
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


def OGBNDataset(args, root='dataset', rand_seed=2021):
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load the dataset
    dataset = PygNodePropPredDataset(name=args.input)
    data = dataset[0]

    # Split the dataset into training, validation, and testing sets
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    val_idx = split_idx['valid']
    test_idx = split_idx['test']

    # Define the device for computation (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data loaders for training, validation, and testing
    train_loader = DataLoader(data[train_idx], batch_size=1024, shuffle=True)
    val_loader = DataLoader(data[val_idx], batch_size=1024)
    test_loader = DataLoader(data[test_idx], batch_size=1024)

    # Set the number of input features, hidden units, and output classes
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    # Initialize the GNN model
    model = build_model(args, num_features, num_classes)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for run in range(args.runs):
        print(f"Run {run + 1}:")
        model.reset_parameters()  # Reset model parameters for each run

        for epoch in range(args_epochs):
            model.train()
            total_loss = 0
            correct = 0

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index)
                loss = criterion(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (out.argmax(dim=1) == data.y).sum().item()

            train_loss = total_loss / len(train_loader)
            train_acc = correct / len(train_idx)
            model.eval()
            correct = 0

            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index)
                    correct += (out.argmax(dim=1) == data.y).sum().item()

            val_acc = correct / len(val_idx)
            print(f'Epoch {epoch + 1}:')
            print(f'Training Loss: {train_loss:.4f} | Training Accuracy: {train_acc:.4f}')
            print(f'Validation Accuracy: {val_acc:.4f}')
            # Testing
            model.eval()
            correct = 0

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index)
                    correct += (out.argmax(dim=1) == data.y).sum().item()

            test_acc = correct / len(test_idx)
            print(f'Testing Accuracy: {test_acc:.4f}')

"""
def OGBNDataset(args, root='dataset', rand_seed=2021):
    # Load the OGBN dataset
    dataset = PygNodePropPredDataset(name=args.input)
    # Get the number of classes in the dataset
    num_classes = dataset.num_classes
    num_features = dataset.num_features
    # Perform 5 runs
    for run in range(args.runs):
        print(f"Run {run+1}:")
        # Split the dataset into train, validation, and test sets
        data = dataset[0]
        data.train_mask = data.val_mask = data.test_mask = None
        data = dataset.get(0)
        split_idx = dataset.get_idx_split()
        data.train_mask = split_idx['train']
        data.val_mask = split_idx['valid']
        data.test_mask = split_idx['test']
        # Define the GNN model
        model = build_model(args, num_features, num_classes)
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        data = data.to(device)

        # Create data loader
        loader = DataLoader([data], batch_size=64, shuffle=True)
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = F.nll_loss(out[batch.train_mask].squeeze(), batch.y[batch.train_mask].squeeze())
                loss.backward()
                optimizer.step()

            # Testing
             model.eval()
             with torch.no_grad():
                 logits = model(data.x, data.edge_index)
                 pred = logits.argmax(dim=1)
                 test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
                 print(f"Test Accuracy: {test_acc:.4f}")



import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

target_dataset = 'ogbn-mag'

# This will download the ogbn-arxiv to the 'networks' folder
dataset = PygNodePropPredDataset(name=target_dataset)

data = dataset[0]

split_idx = dataset.get_idx_split() 
        
train_idx = split_idx['train']
valid_idx = split_idx['valid']
test_idx = split_idx['test']

train_loader = NeighborLoader(data, input_nodes=train_idx,
                              shuffle=True, num_workers=os.cpu_count() - 2,
                              batch_size=100, num_neighbors=[30] * 2)
total_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                               batch_size=200, shuffle=False,
                               num_workers=os.cpu_count() - 2)
class SAGE(torch.nn.Module):
    def __init__(self, in_channels,
                 hidden_channels, out_channels,
                 n_layers=2):
        
        super(SAGE, self).__init__()
        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList()
        if n_layers == 1:
            self.layers.append(SAGEConv(in_channels, out_channels,   normalize=False))
        elif n_layers == 2:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        else:
            self.layers.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(n_layers - 2):
            self.layers.append(SAGEConv(hidden_channels,  hidden_channels, normalize=False))
            self.layers_bn.append(torch.nn.BatchNorm1d(hidden_channels))
            
            self.layers.append(SAGEConv(hidden_channels, out_channels, normalize=False))
            
        for layer in self.layers:
            layer.reset_parameters()
        
    def forward(self, x, edge_index):
        if len(self.layers) > 1:
            looper = self.layers[:-1]
        else:
            looper = self.layers
        
        for i, layer in enumerate(looper):
            x = layer(x, edge_index)
            try:
                x = self.layers_bn[i](x)
            except Exception as e:
                abs(1)
            finally:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        if len(self.layers) > 1:
            x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=-1), torch.var(x)
    
    def inference(self, total_loader, device):
        xs = []
        var_ = []
        for batch in total_loader:
            out, var = self.forward(batch.x.to(device), batch.edge_index.to(device))
            out = out[:batch.batch_size]
            xs.append(out.cpu())
            var_.append(var.item())
        
        out_all = torch.cat(xs, dim=0)
        
        return out_all, var_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.x.shape[1], 256, dataset.num_classes, n_layers=2)
model.to(device)
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=7)

def test(model, device):
    evaluator = Evaluator(name=target_dataset)
    model.eval()
    out, var = model.inference(total_loader, device)
    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, val_acc, test_acc, torch.mean(torch.Tensor(var))

for epoch in range(1, epochs):
    model.train()
    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')
    total_loss = total_correct = 0
    for batch in train_loader:
        batch_size = batch.batch_size
        optimizer.zero_grad()
        out, _ = model(batch.x.to(device), batch.edge_index.to(device))
        out = out[:batch_size]
        batch_y = batch.y[:batch_size].to(device)
        batch_y = torch.reshape(batch_y, (-1,))
        loss = F.nll_loss(out, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_y).sum())
        pbar.update(batch.batch_size)
    pbar.close()
    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)
    train_acc, val_acc, test_acc, var = test(model, device)
    
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Var: {var:.4f}')
"""
