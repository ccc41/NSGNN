import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import AvgPooling, MaxPooling, SumPooling


class GATLayer(nn.Module):
    def __init__(self, input_size, output_size, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_mode='flatten', activation=None):
        super(GATLayer, self).__init__()
        self.gnn = GATConv(in_feats=input_size,
                           out_feats=output_size,
                           num_heads=num_heads,
                           feat_drop=feat_drop,
                           attn_drop=attn_drop,
                           negative_slope=alpha,
                           activation=activation,
                           residual=residual,
                           allow_zero_in_degree=True,)
        assert agg_mode in ['flatten', 'mean']
        self.agg_mode = agg_mode
        self.activation = activation

    def forward(self, g, features):
        new_feats = self.gnn(g, features)
        # print('new_feats_before.shape', new_feats.shape)
        if self.agg_mode == 'flatten':
            new_feats = new_feats.flatten(1)
        else:
            new_feats = new_feats.mean(1)
        # print('new_feats_after.shape', new_feats.shape)
        # if self.activation is not None:
        #     new_feats = self.activation(new_feats)

        return new_feats


class UnsupervisedGAT(nn.Module):
    def __init__(
        self,
        input_size=1433,
        hidden_size=512,
        output_size=512,
        num_layers=2,
        num_heads=4,
        readout='avg',
        feat_drop=0.5,
        attn_drop=0.6,
        layernorm=False,
        agg_mode='flatten'
    ):
        super(UnsupervisedGAT, self).__init__()
        assert hidden_size % num_heads == 0
        out_size_mean = [hidden_size for i in range(num_layers)]
        out_size_flatten = [hidden_size//num_heads for i in range(num_layers)]
        self.layers = nn.ModuleList(
            [
                GATLayer(
                    input_size=input_size if i == 0 else hidden_size,
                    output_size=hidden_size//num_heads if i +
                    1 < num_layers else output_size,
                    num_heads=num_heads if i+1 < num_layers else 1,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    alpha=0.2,
                    residual=False,
                    agg_mode=agg_mode,
                    activation=F.leaky_relu if i+1 < num_layers else None,
                )
                for i in range(num_layers)
            ]
        )
        if readout == 'avg':
            self.readout = AvgPooling()
        elif readout == 'sum':
            self.readout = SumPooling()
        elif readout == 'max':
            self.readout = MaxPooling()
        elif readout == 'root':
            self.readout = lambda _, x: x
        else:
            raise NotImplementedError
        self.layernorm = layernorm
        if layernorm:
            self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, g, features):
        x = features
        for i, layer in enumerate(self.layers):
            x = layer(g, x)
        pool_x = self.readout(g, x)
        if self.layernorm:
            x = self.ln(x)
            pool_x = self.ln(pool_x)
        return pool_x, x


if __name__ == "__main__":
    model = UnsupervisedGAT(num_layers=1)
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1], [1, 2, 2])
    feat = torch.rand(3, 64)
    pool_x, x = model(g, feat)
    print(pool_x.shape, x.shape)
