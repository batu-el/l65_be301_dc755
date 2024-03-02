
import os
import sys
import time
import math
import random
import itertools
from datetime import datetime
from typing import Mapping, Tuple, Sequence, List

import pandas as pd
import networkx as nx
import numpy as np
import scipy as sp

from tqdm.notebook import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ReLU, BatchNorm1d, LayerNorm, Module, ModuleList, Sequential
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention
from torch.optim import Adam

import torch_geometric
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid

import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, dense_to_sparse, to_dense_batch, to_dense_adj

from torch_geometric.nn import GCNConv, GATConv

from torch_scatter import scatter, scatter_mean, scatter_max, scatter_sum

import lovely_tensors as lt
lt.monkey_patch()

import matplotlib.pyplot as plt
import seaborn as sns

data = Planetoid(root='/tmp/Cora', name='Cora')[0]
dataset = 'Cora'

def download_cora():
    # Download and load Cora dataset
    dataset = 'Cora'
    # transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm(), T.ToSparseTensor()])
    dataset = Planetoid('/tmp/Cora', dataset)
    data = dataset[0]

def get_shortest_path_matrix(adjacency_matrix):
    # Convert the adjacency matrix to a NetworkX graph
    graph = nx.from_numpy_array(adjacency_matrix.cpu().numpy(), create_using=nx.DiGraph)
    # Compute the shortest path matrix using Floyd-Warshall algorithm in NetworkX
    shortest_path_matrix = nx.floyd_warshall_numpy(graph)
    # Convert numpy array back to torch tensor
    shortest_path_matrix = torch.tensor(shortest_path_matrix).float()
    return shortest_path_matrix

dense_adj = to_dense_adj(data.edge_index, max_num_nodes = data.x.shape[0])[0]
dense_shortest_path_matrix = get_shortest_path_matrix(dense_adj)

class DenseGraphTransformerModel(Module):

    def __init__(
            self,
            in_dim: int = dataset.num_features,
            pos_enc_dim: int = 16,
            hidden_dim: int = 128,
            num_heads: int = 4,
            num_layers: int = 3,
            out_dim: int = dataset.num_classes,
            dropout: float = 0.5,
        ):
        super().__init__()

        self.lin_in = Linear(in_dim, hidden_dim)
        self.lin_pos_enc = Linear(pos_enc_dim, hidden_dim)
        self.lin_out = Linear(hidden_dim, out_dim)

        self.layers = ModuleList()
        for layer in range(num_layers):
            self.layers.append(
                # TransformerEncoderLayer(
                #     d_model = hidden_dim,
                #     nhead = num_heads,
                #     dim_feedforward = hidden_dim * 2,
                #     dropout = dropout,
                # )
                MultiheadAttention(
                    embed_dim = hidden_dim,
                    num_heads = num_heads,
                    dropout = dropout
                )
            )
        # self.encoder = TransformerEncoder(
        #     TransformerEncoderLayer(
        #         d_model = hidden_dim,
        #         nhead = num_heads,
        #         dim_feedforward = hidden_dim * 2,
        #         dropout = dropout,
        #     ),
        #     num_layers
        # )

        self.attn_bias_scale = torch.nn.Parameter(torch.tensor([10.0]))  # controls how much we initially bias our model to nearby nodes
        self.dropout = dropout

    def forward(self, x, pos_enc, dense_sp_matrix):

        # x = self.lin_in(x) + self.lin_pos_enc(pos_enc)
        x = self.lin_in(x)  # no node positional encoding

        # attention bias
        # [i, j] -> inverse of shortest path distance b/w node i and j
        # diagonals -> self connection, set to 0
        # disconnected nodes -> -1
        attn_bias = self.attn_bias_scale * torch.nan_to_num(
            (1 / (torch.nan_to_num(dense_sp_matrix, nan=-1, posinf=-1, neginf=-1))),
            nan=0, posinf=0, neginf=0
        )

        # TransformerEncoder
        # x = self.encoder(x, mask = attn_bias)

        self.attn_weights_list = []

        for layer in self.layers:
            # # TransformerEncoderLayer
            # # float mask adds learnable additive attention bias
            # x = layer(x, src_mask = attn_bias)

            # MHSA layer
            # float mask adds learnable additive attention bias
            x_in = x
            x, attn_weights = layer(
                x, x, x,
                attn_mask = attn_bias,
                average_attn_weights = False
            )
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x_in + x

            self.attn_weights_list.append(attn_weights)

        x = self.lin_out(x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DenseGraphTransformerModel().to(device)

data = T.AddLaplacianEigenvectorPE(k = 16, attr_name = 'pos_enc')(data)
# data = T.AddRandomWalkPE(walk_length = 16, attr_name = 'pos_enc')(data)
data.dense_adj = to_dense_adj(data.edge_index, max_num_nodes = data.x.shape[0])[0]
data.dense_sp_matrix = dense_shortest_path_matrix.float()  # pre-computed in previous cell
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.pos_enc, data.dense_sp_matrix)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred, accs = model(data.x, data.pos_enc, data.dense_sp_matrix).argmax(dim=-1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
times = []
for epoch in range(1, 100):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
          f'Final Test: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

# Notes
# - Dense Transformer needs to be trained for a bit longer to reach low loss value
# - Node positional encodings are not particularly useful
# - Edge distance encodings are very useful
# - Since Cora is highly homophilic, it is important to bias the attention towards nearby nodes