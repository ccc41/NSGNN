from dgl.batch import batch
from sklearn import neighbors
import torch
from torch import nn
import dgl
import dpp
import numpy as np
from scipy.linalg import fractional_matrix_power, inv
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PNClassifyData(Dataset):
    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y
        self.len = x.shape[0]
        self.y_data = torch.tensor(self.y_data, dtype=torch.long)
        self.x_data = torch.tensor(self.x_data, dtype=torch.float)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


class PNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        """
        return probability to be label 1
        :param x:
        :return:
        """
        pred = F.softmax(self.forward(x))
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(1 - t[0])
            else:
                ans.append(t[1])
        return torch.tensor(ans)


def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count


def get_mask_origin(out):
    batch_size = out.shape[0]
    neg_mask = torch.ones((batch_size, batch_size), dtype=torch.float).cuda()
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    for i in range(batch_size):
        pos_mask[i][i] = 1
        neg_mask[i][i] = 0
    return pos_mask, neg_mask


def get_mask_dpp(out, dpp_value):
    batch_size = out.shape[0]
    neg_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    index_dpp = dpp.dpp_sampling(
        out.detach().cpu().numpy(), dpp_value)
    for i in range(batch_size):
        pos_mask[i][i] = 1
        neg_mask[i][index_dpp[i]] = 1*1.5
        neg_mask[i][i] = 0
    return pos_mask, neg_mask


def get_mask_cor(out, cor, ratio):
    batch_size = out.shape[0]
    alpha = np.quantile(cor[0], ratio)
    neg_mask = torch.ones((batch_size, batch_size), dtype=torch.float).cuda()
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    for i in range(batch_size):
        pos_mask[i][i] = 1
        neg_mask[i][i] = 0
        cor_pos_index = np.where(cor[i] > alpha)
        for j in cor_pos_index:
            neg_mask[i][j] = 0

    return pos_mask, neg_mask


def get_mask_cor_dpp(out, cor, ratio, dpp_value):
    batch_size = out.shape[0]
    alpha = np.quantile(cor[0], ratio)
    neg_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    index_dpp = dpp.dpp_sampling(
        out.detach().cpu().numpy(), dpp_value)
    for i in range(batch_size):
        pos_mask[i][i] = 1
        neg_mask[i][index_dpp[i]] = 1*1.5
        neg_mask[i][i] = 0
        cor_pos_index = np.where(cor[i] > alpha)
        for j in cor_pos_index:
            neg_mask[i][j] = 0

    return pos_mask, neg_mask

def get_mask_neighbor(out, neighbor_nodes):
    batch_size = out.shape[0]
    # neg_mask = torch.ones((batch_size, batch_size), dtype=torch.float).cuda()
    neg_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    index_dpp = dpp.dpp_sampling(
        out.detach().cpu().numpy(), int(batch_size/(1.5)))
    for i in range(batch_size):
        pos_mask[i][i] = 1
        neg_mask[i][index_dpp[i]] = 1*1.5
        neg_mask[i][i] = 0
        neigh = neighbor_nodes[i]
        for j in neigh:
            pos_mask[i][j] = 1
            neg_mask[i][j] = 0
    return pos_mask, neg_mask


def get_mask_guassian(out, graph_k, graph_q, feat_q, feat_k, x_q, x_k):
    batch_size = out.shape[0]
    embeddings = ((feat_q + feat_k) / 2).detach().cpu()
    pca = PCA(n_components=1)
    embeddings = pca.fit_transform(embeddings)
    q_ID = list(np.array(graph_q.ndata['_ID'].detach().cpu()))
    k_ID = list(np.array(graph_k.ndata['_ID'].detach().cpu()))

    neg_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    index_dpp = dpp.dpp_sampling(
        out.detach().cpu().numpy(), int(batch_size/(1.5)))
    for i in range(batch_size):
        pos_mask[i][i] = 1
        neg_mask[i][index_dpp[i]] = 1*1.5
        neg_mask[i][i] = 0

        q_pos_samples_IDS = list(np.array(dgl.unbatch(graph_q)[i].ndata['_ID'].detach().cpu()))
        k_pos_samples_IDS = list(np.array(dgl.unbatch(graph_k)[i].ndata['_ID'].detach().cpu()))
        q_pos_sample_index = [q_ID.index(i) for i in q_pos_samples_IDS]
        k_pos_sample_index = [k_ID.index(i) for i in k_pos_samples_IDS]
        # positive sample embeddings
        q_pos_sample_embedding = x_q[q_pos_sample_index]
        k_pos_sample_embedding = x_k[k_pos_sample_index]
        pos_sample_embedding = torch.cat([q_pos_sample_embedding, k_pos_sample_embedding], 0)
        # negative sample embeddings
        q_neg_sample_embedding = feat_q[torch.arange(feat_q.size(0)) != i]
        k_neg_sample_embedding = feat_k[torch.arange(feat_k.size(0)) != i]

        neg_sample_embedding = torch.cat([q_neg_sample_embedding, k_neg_sample_embedding], 0)
        # dimension reduction and fit positive gaussian

        tmp_pos = pca.transform(pos_sample_embedding.detach().cpu())
        mean_pos = np.mean(tmp_pos, axis=0)
        cov_pos = np.cov(tmp_pos, rowvar=0)
        var_pos = multivariate_normal(mean_pos, cov_pos)
        # dimension reduction and fit negative gaussian
        tmp_neg = pca.transform(neg_sample_embedding.detach().cpu())
        mean_neg = np.mean(tmp_neg, axis=0)
        cov_neg = np.cov(tmp_neg, rowvar=0)
        var_neg = multivariate_normal(mean_neg, cov_neg)

        # all samples in the batch to fit in the pos/neg gaussian
        ratio_list =[]
        for j in embeddings:
            ratio_list.append(var_pos.pdf(j) / var_neg.pdf(j))

        alpha = np.quantile(ratio_list, 0.1)
        ratio_index = np.where(ratio_list[i] < alpha)
        for k in ratio_index:
            # pos_mask[i][j] = 1
            neg_mask[i][k] = 0
    return pos_mask, neg_mask


def get_mask_classifier(out, graph_k, graph_q, feat_q, feat_k, x_q, x_k, device):
    batch_size = out.shape[0]
    embeddings = ((feat_q + feat_k) / 2).detach().cpu()
    # pca = PCA(n_components=10)
    # embeddings = pca.fit_transform(embeddings)
    q_ID = list(np.array(graph_q.ndata['_ID'].detach().cpu()))
    k_ID = list(np.array(graph_k.ndata['_ID'].detach().cpu()))

    neg_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.float).cuda()
    index_dpp = dpp.dpp_sampling(
        out.detach().cpu().numpy(), int(batch_size/(1.5)))
    for i in range(batch_size):
        pos_mask[i][i] = 1
        neg_mask[i][index_dpp[i]] = 1*1.5
        neg_mask[i][i] = 0

        q_pos_samples_IDS = list(np.array(dgl.unbatch(graph_q)[i].ndata['_ID'].detach().cpu()))
        k_pos_samples_IDS = list(np.array(dgl.unbatch(graph_k)[i].ndata['_ID'].detach().cpu()))
        q_pos_sample_index = [q_ID.index(i) for i in q_pos_samples_IDS]
        k_pos_sample_index = [k_ID.index(i) for i in k_pos_samples_IDS]
        # positive sample embeddings
        q_pos_sample_embedding = x_q[q_pos_sample_index]
        k_pos_sample_embedding = x_k[k_pos_sample_index]
        pos_sample_embedding = torch.cat([q_pos_sample_embedding, k_pos_sample_embedding], 0)
        # negative sample embeddings
        q_neg_sample_embedding = feat_q[torch.arange(feat_q.size(0)) != i]
        k_neg_sample_embedding = feat_k[torch.arange(feat_k.size(0)) != i]

        neg_sample_embedding = torch.cat([q_neg_sample_embedding, k_neg_sample_embedding], 0)

        # dimension reduction
        # tmp_pos = pca.transform(pos_sample_embedding.detach().cpu())
        # tmp_neg = pca.transform(neg_sample_embedding.detach().cpu())
        tmp_data = torch.cat([pos_sample_embedding, neg_sample_embedding], 0)
        pos_label = torch.ones(pos_sample_embedding.shape[0])
        neg_label = torch.zeros(neg_sample_embedding.shape[0])
        tmp_label = torch.cat([pos_label, neg_label], 0)
        dataset = PNClassifyData(tmp_data, tmp_label)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        input_size = tmp_data.shape[1]
        hidden_size = input_size * 2
        model = PNClassifier(input_size, hidden_size).to(device)
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam梯度优化器
        epochs = 5
        for i in range(epochs):
            for idx, batch in enumerate(data_loader):
                y_pred = model.forward(batch[0].to(device))
                loss = criterion(y_pred.view(-1, y_pred.shape[-1]), batch[1].to(device).contiguous().view(-1))
                # losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # all samples in the batch to fit in the pos/neg gaussian
        ratio_list =[]
        for j in embeddings:
            ratio = model.predict(torch.tensor(j, dtype=torch.float32, device=device).unsqueeze(0))
            ratio_list.append(ratio.item())

        alpha = np.quantile(ratio_list, 0.9)
        ratio_index = np.where(ratio_list[i] > alpha)
        for k in ratio_index:
            # pos_mask[i][j] = 1
            if i != k:
                neg_mask[i][k] = 0
    return pos_mask, neg_mask


def subgraph_to_nodes(g,  ind, n_hops, label=None):
    neighbors = []
    # count = 1
    # count_1 = 0
    # count_2 = 0
    # pos = 0
    for id in ind:
        subgraph = dgl.in_subgraph(g, id)
        block = dgl.to_block(subgraph, id)
        # lbl_seed = label[id].item()
        for _ in range(n_hops-1):
            subgraph = dgl.in_subgraph(g, block.srcdata[dgl.NID])
            block = dgl.to_block(subgraph, block.srcdata[dgl.NID])

        neighbor = block.srcdata[dgl.NID].tolist()
        ind_pos = [ind.index(item)
                   for item in neighbor if item in ind]
        # ind_neighbor = [ind.index(item)
        # for item in neighbor if item in ind and label[item] == lbl_seed]
        neighbors.append(ind_pos)
        # lbl_neighbor = label[neighbor].tolist()
        # pos += lbl_neighbor.count(lbl_seed)
        # count += len(neighbor)
        # count_1 += len(ind_pos)
        # count_2 += len(ind_neighbor)
    return neighbors,  # pos/count

def neighbor_nodes(g, node_index, degree):
    """
    find neighbors given node_index in graph g.
    :param g:
    :param node_index:
    :param degree:
    :return:
    """
    neighbors = []
    count = 1
    while count <= degree:
        neighbors.extend(list(np.array(g.edges()[1][g.edges()[0] == node_index])))
        count += 1
    return np.unique(neighbors)
