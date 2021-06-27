import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import AvgPooling, MaxPooling, SumPooling, Set2Set
from dgl.nn.pytorch import GraphConv


class GCNLayer(nn.Module):
    def __init__(self, input_size, output_size, activation=F.relu,
                 residual=False, norm='both', batchnorm=False, dropout=0.5):
        super(GCNLayer, self).__init__()
        self.activation = activation
        self.graph_conv = GraphConv(
            in_feats=input_size, out_feats=output_size, norm=norm, activation=activation, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(input_size, output_size)
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(output_size)

    def forward(self, g, feature, idx):
        new_feats = feature
        # print('idx:', idx)
        if idx != 0:
            new_feats = self.dropout(new_feats)
        new_feats = self.graph_conv(g, new_feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feature))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        return new_feats


class UnsupervisedGCN(nn.Module):
    def __init__(
        self,
        input_size=1433,
        hidden_size=16,
        output_size=16,
        num_layers=2,
        readout='avg',
        norm='both',
        dropout=0.5,
        layernorm=False,
        set2set_lstm_layer=1,
        set2set_iter=10
    ):
        super(UnsupervisedGCN, self).__init__()
        print('dropout:', dropout)
        self.layers = nn.ModuleList(
            [
                GCNLayer(
                    input_size=input_size if i == 0 else hidden_size,
                    output_size=hidden_size if i+1 < num_layers else output_size,
                    activation=F.relu if i+1 < num_layers else None,
                    residual=False,
                    norm=norm,
                    batchnorm=False,
                    dropout=dropout
                )
                for i in range(num_layers)
            ]
        )

        self.readout1 = MaxPooling()
        self.readout2 = AvgPooling()
        self.readout3 = SumPooling()
        self.readout4 = self.readout = Set2Set(
            hidden_size, n_iters=set2set_iter, n_layers=set2set_lstm_layer)
        self.lin_readout = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.layernorm = layernorm
        if layernorm:
            self.ln = nn.LayerNorm(output_size, elementwise_affine=False)

    def forward(self, g, features, efeats=None):
        x = features
        for idx, layer in enumerate(self.layers):
            x = layer(g, x, idx)
        pool_x_1 = self.readout1(g, x)
        pool_x_2 = self.readout2(g, x)

        pool_x = pool_x_1+pool_x_2
        if self.layernorm:
            x = self.ln(x)
            pool_x = self.ln(pool_x)
        return pool_x, x


if __name__ == "__main__":
    model = UnsupervisedGCN(num_layers=2)
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    feat = torch.rand(3, 64)
    pool_feats, feats = model(g, feat)
    print(pool_feats.shape, feats.shape)
