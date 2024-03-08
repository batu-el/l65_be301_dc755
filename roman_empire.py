from dgl.data import RomanEmpireDataset
import torch
from torch_geometric.data import Data

def preprocess(graph, train_ratio = 0.7, val_ratio = 0.15, test_ratio = 0.15):
    x = graph.ndata['feat']
    y = graph.ndata['label']
    edge_index = torch.stack([graph.edges()[0], graph.edges()[1]])

    num_nodes = len(y)
    indices = torch.randperm(num_nodes)
    num_train, num_val = int(num_nodes * train_ratio), int(num_nodes * val_ratio)
    num_test = num_nodes - num_train - num_val

    train_mask, val_mask, test_mask = torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool), torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train+num_val]] = True
    test_mask[indices[num_train+num_val:]] = True
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data

print(RomanEmpireDataset)

def preprocess_roman_empire():
    dataset = RomanEmpireDataset()[0]
    return preprocess(graph=dataset)

print(preprocess_roman_empire())