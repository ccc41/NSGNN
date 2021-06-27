import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gcn import UnsupervisedGCN
from models.gat import UnsupervisedGAT
from models.gin import UnsupervisedGIN


class GraphEncoder(nn.Module):
    def __init__(
        self,
        input_size=1433,
        hidden_size=512,
        output_size=512,
        num_layers=2,
        norm=False,
        dropout=0.5,
        gnn_model='gcn',
        readout='avg',
    ):
        super(GraphEncoder, self).__init__()
        self.gnn = UnsupervisedGCN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            readout=readout,
            norm='both',
            dropout=dropout,
            layernorm=norm,
        )
        self.gnn_model = gnn_model

        self.lin_readout = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )
        self.norm = norm

    def forward(self, g, features, return_all_outputs=False, train=True, unpool=False):
        pool_x, all_outputs = self.gnn(g, features)
        return pool_x, all_outputs

    # def contrast(self, x_q, x_k, shuf_x_q, shuf_x_k):
    #     c_q = torch.mean(x_q, dim=0, keepdim=True)
    #     c_k = torch.mean(x_k, dim=0, keepdim=True)
    #     s1 = torch.matmal(c_q, x_k.t())
    #     s2 = torch.matmal(c_k,x_q.t())
    #     s3  = torch.matmul(c_q,shuf_x_k)
    #     s4 = torch.matmul(c_k,shuf_x_q)
    #     return

    def embed(self, g, features, return_all_outputs=False, train=True, unpool=False):
        pool_x, all_outputs = self.gnn(g, features)
        return pool_x.detach(), all_outputs.detach()


if __name__ == "__main__":
    model = GraphEncoder(gnn_model="gin", input_size=16)
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1, 2], [1, 2, 2, 1])
    g.ndata["pos_directed"] = torch.rand(3, 16)
    g.ndata["pos_undirected"] = torch.rand(3, 16)
    g.ndata["seed"] = torch.zeros(3, dtype=torch.long)
    g.ndata["nfreq"] = torch.ones(3, dtype=torch.long)
    g.edata["efreq"] = torch.ones(4, dtype=torch.long)
    batch_g = dgl.batch([g, g, g])
    print(batch_g.ndata['seed'])
    y = model(batch_g, torch.cat(
        [g.ndata["pos_directed"], g.ndata["pos_directed"], g.ndata["pos_directed"]]))
    print(y.shape)
    print(y)
