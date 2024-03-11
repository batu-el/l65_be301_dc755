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





class GNNModel(Module):

    def __init__(
            self,
            data: Data,
            # in_dim: int = data.x.shape[-1],
            hidden_dim: int = 128,
            num_heads: int = 1,
            num_layers: int = 1,
            # out_dim: int = len(data.y.unique()),
            dropout: float = 0.5,
        ):
        super().__init__()

        in_dim = data.x.shape[-1]
        out_dim = len(data.y.unique())
        self.lin_in = Linear(in_dim, hidden_dim)
        self.lin_out = Linear(hidden_dim, out_dim)

        self.layers = ModuleList()
        for layer in range(num_layers):
            self.layers.append(
                # GCNConv(hidden_dim, hidden_dim)
                GATConv(hidden_dim, hidden_dim // num_heads, num_heads)
            )
        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.lin_in(x)

        for layer in self.layers:
            # conv -> activation ->  dropout -> residual
            x_in = x
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x_in + x

        x = self.lin_out(x)

        return x.log_softmax(dim=-1)


class SparseGraphTransformerModel(Module):
    def __init__(
            self,
            data,
            # dataset,
            # in_dim: int = data.x.shape[-1],
            hidden_dim: int = 128,
            num_heads: int = 1,
            num_layers: int = 1,
            # out_dim: int = len(data.y.unique()),
            dropout: float = 0.5,
        ):
        super().__init__()

        in_dim = data.x.shape[-1]
        out_dim = len(data.y.unique())

        self.lin_in = Linear(in_dim, hidden_dim)
        self.lin_out = Linear(hidden_dim, out_dim)

        self.layers = ModuleList()
        for layer in range(num_layers):
            self.layers.append(
                MultiheadAttention(
                    embed_dim = hidden_dim,
                    num_heads = num_heads,
                    dropout = dropout
                )
            )
        self.dropout = dropout

    def forward(self, x, dense_adj):

        x = self.lin_in(x)

        # TransformerEncoder
        # x = self.encoder(x, mask = ~dense_adj.bool())

        self.attn_weights_list = []

        for layer in self.layers:
            # # TransformerEncoderLayer
            # # boolean mask enforces graph structure
            # x = layer(x, src_mask = ~dense_adj.bool())

            # MHSA layer
            # boolean mask enforces graph structure
            x_in = x
            x, attn_weights = layer(
                x, x, x,
                attn_mask = ~dense_adj.bool(),
                average_attn_weights = False
            )
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x_in + x

            self.attn_weights_list.append(attn_weights)

        x = self.lin_out(x)

        return x.log_softmax(dim=-1)

class DenseGraphTransformerModel(Module):

    def __init__(
            self,
            data: Data,
            # in_dim: int = data.x.shape[-1],
            pos_enc_dim: int = 16,
            hidden_dim: int = 128,
            num_heads: int = 1,
            num_layers: int = 1,
            # out_dim: int = len(data.y.unique()),
            dropout: float = 0.5,
        ):
        super().__init__()

        in_dim = data.x.shape[-1]
        out_dim = len(data.y.unique())

        self.lin_in = Linear(in_dim, hidden_dim)
        self.lin_pos_enc = Linear(pos_enc_dim, hidden_dim)
        self.lin_out = Linear(hidden_dim, out_dim)

        self.layers = ModuleList()
        for layer in range(num_layers):
            self.layers.append(
                MultiheadAttention(
                    embed_dim = hidden_dim,
                    num_heads = num_heads,
                    dropout = dropout
                )
            )


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
        #attn_bias = torch.ones_like(attn_bias)

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



class DenseGraphTransformerModel_V2(Module):

    def __init__(
            self,
            data: Data,
            # in_dim: int = data.x.shape[-1],
            pos_enc_dim: int = 16,
            hidden_dim: int = 128,
            num_heads: int = 1,
            num_layers: int = 1,
            # out_dim: int = len(data.y.unique()),
            dropout: float = 0.5,
        ):
        super().__init__()

        in_dim = data.x.shape[-1]
        out_dim = len(data.y.unique())

        self.lin_in = Linear(in_dim, hidden_dim)
        self.lin_pos_enc = Linear(pos_enc_dim, hidden_dim)
        self.lin_out = Linear(hidden_dim, out_dim)

        self.layers = ModuleList()
        for layer in range(num_layers):
            self.layers.append(
                MultiheadAttention(
                    embed_dim = hidden_dim,
                    num_heads = num_heads,
                    dropout = dropout
                )
            )


        self.attn_bias_scale = torch.nn.Parameter(torch.tensor([10.0]))  # controls how much we initially bias our model to nearby nodes
        self.dropout = dropout

    def forward(self, x, pos_enc, dense_sp_matrix):

        x = self.lin_in(x) + self.lin_pos_enc(pos_enc)
        # x = self.lin_in(x)  # no node positional encoding

        # attention bias
        # [i, j] -> inverse of shortest path distance b/w node i and j
        # diagonals -> self connection, set to 0
        # disconnected nodes -> -1
        # attn_bias = self.attn_bias_scale * torch.nan_to_num(
        #     (1 / (torch.nan_to_num(dense_sp_matrix, nan=-1, posinf=-1, neginf=-1))),
        #     nan=0, posinf=0, neginf=0
        # )
        #attn_bias = torch.ones_like(attn_bias)

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
                # attn_mask = attn_bias,
                average_attn_weights = False
            )
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = x_in + x

            self.attn_weights_list.append(attn_weights)

        x = self.lin_out(x)

        return x.log_softmax(dim=-1)
    
def train_sparse(data=None, model=None, optimizer=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.dense_adj)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test_sparse(data=None, model=None):
    model.eval()
    pred, accs = model(data.x, data.dense_adj).argmax(dim=-1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def sparse_training_loop(data=None, model=None, optimizer=None):
    best_val_acc = test_acc = 0
    times = []
    for epoch in range(1, 100):
        start = time.time()
        loss = train_sparse(data, model, optimizer)
        train_acc, val_acc, tmp_test_acc = test_sparse(data, model)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
            f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
            f'Final Test: {test_acc:.4f}')
        times.append(time.time() - start)

    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

def train_dense(data=None, model=None, optimizer=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, 0, data.dense_sp_matrix)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test_dense(data=None, model=None):
    model.eval()
    pred, accs = model(data.x, 0, data.dense_sp_matrix).argmax(dim=-1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

def dense_training_loop(data=None, model=None, optimizer=None):
    """
    Train a dense graph transformer model.
    Note: data.dense_sp_matrix is the shortest path matrix of the graph,
    and must be defined in the data object.

    Args:
    data: Data: The graph data
    model: DenseGraphTransformerModel: The model
    optimizer: Adam: The optimizer

    Returns:
    None

    """
    best_val_acc = test_acc = 0
    times = []
    for epoch in range(1, 100):
        start = time.time()
        loss = train_dense(data, model, optimizer)
        train_acc, val_acc, tmp_test_acc = test_dense(data, model)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
            f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
            f'Final Test: {test_acc:.4f}')
        times.append(time.time() - start)

    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")