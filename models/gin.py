import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
import dgl


class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_size)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_size = output_size

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_size, output_size)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_size, hidden_size))

            for layer in range(num_layers-2):
                self.linears.append(nn.Linear(hidden_size, hidden_size))
            self.linears.append(nn.Linear(hidden_size, output_size))

            for layer in range(num_layers-1):
                self.batch_norms.append(
                    nn.BatchNorm1d(hidden_size)
                )

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers-1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class UnsupervisedGIN(nn.Module):
    def __init__(
        self,
        input_size=64,
        hidden_size=512,
        output_size=512,
        num_layers=2,
        num_mlp_layers=2,
        dropout=0,
        learn_eps=False,
        graph_pooling_type='mean',
        neighbor_pooling_type='mean',
    ):
        super(UnsupervisedGIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.ginlayer = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        i = 0
        for layer in range(self.num_layers):
            print('layer', layer)
            if layer == 0:
                mlp = MLP(
                    num_mlp_layers, input_size, hidden_size, hidden_size
                )
            else:
                mlp = MLP(
                    num_mlp_layers,
                    hidden_size,
                    hidden_size,
                    hidden_size if layer < self.num_layers-1 else output_size,
                )
            self.ginlayer.append(
                GINConv(
                    ApplyNodeFunc(mlp),
                    neighbor_pooling_type,
                    0,
                    self.learn_eps,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))

        self.liners_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.liners_prediction.append(
                    nn.Linear(input_size, hidden_size))
            else:
                self.liners_prediction.append(
                    nn.Linear(hidden_size, output_size))
        self.drop = nn.Dropout(dropout)

        if graph_pooling_type == "sum":
            self.pool = SumPooling()
        elif graph_pooling_type == "mean":
            self.pool = AvgPooling()
        elif graph_pooling_type == "max":
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, features):
        x = features
        for i in range(self.num_layers):
            x = self.ginlayer[i](g, x)
            x = F.relu(x)
        pool_x = self.pool(g, x)

        # score_over_layer = 0
        # all_outputs = []
        # for i, feature in list(enumerate(hidden_rep)):
        #     pooled_h = self.pool(g, feature)
        #     all_outputs.append(pooled_h)
        #     # score_over_layer += self.drop(self.liners_prediction[i](pooled_h))
        # return all_outputs[-1], all_outputs[1:], x
        return pool_x, x


if __name__ == "__main__":
    model = UnsupervisedGIN(num_layers=1)
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    feat = torch.rand(3, 64)
    score, outputs = model(g, feat)
    print(score, outputs)
