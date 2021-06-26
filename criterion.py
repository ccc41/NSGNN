'''
Author: your name
Date: 2021-02-21 21:43:53
LastEditTime: 2021-03-20 15:52:50
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /csy/ours/criterion.py
'''
import torch
import torch.nn as nn
from utils import get_mask_cor, get_mask_origin, get_mask_neighbor, get_mask_guassian, get_mask_classifier, get_mask_dpp, get_mask_cor_dpp


class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()

    def forward(self, x, device, labels):
        batch_size = x.shape[0]
        pos_mask = get_mask_1(labels)
        loss = (-torch.log(torch.sum(torch.exp(x) * pos_mask, dim=1)) +
                x.logsumexp(dim=1)).mean()
        return loss


class NCESoftmaxLoss_2(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss_2, self).__init__()

    def forward(self, x, **kwargs):
        x = (x - torch.mean(x)) / torch.std(x)
        if kwargs['method'] == 'correlation':
            cor = kwargs['cor']
            pos_mask, neg_mask = get_mask_cor(x, cor, kwargs['ratio'])
        elif kwargs['method'] == 'origin':
            pos_mask, neg_mask = get_mask_origin(x)
        elif kwargs['method'] == 'dpp':
            pos_mask, neg_mask = get_mask_dpp(x, kwargs['dpp_value'])
        elif kwargs['method'] == 'cor_dpp':
            cor = kwargs['cor']
            pos_mask, neg_mask = get_mask_cor_dpp(x, cor, kwargs['ratio'], kwargs['dpp_value'])
        elif kwargs['method'] == 'neighbor':
            neighbor_nodes = kwargs['neighbor']
            pos_mask, neg_mask = get_mask_neighbor(x, neighbor_nodes)
        elif kwargs['method'] == 'gaussian':
            pos_mask, neg_mask = get_mask_guassian(x, kwargs['graph_k'], kwargs['graph_q'], kwargs['feat_q'],
                                                   kwargs['feat_k'], kwargs['x_q'], kwargs['x_k'])
        elif kwargs['method'] == 'classifier':
            pos_mask, neg_mask = get_mask_classifier(x, kwargs['graph_k'], kwargs['graph_q'], kwargs['feat_q'],
                                               kwargs['feat_k'], kwargs['x_q'], kwargs['x_k'], kwargs['device'])

        pos = torch.sum(torch.exp(x) * pos_mask, dim=1)
        neg = torch.sum(torch.exp(x) * neg_mask, dim=1)
        Ng = pos + neg
        loss = (-torch.log(pos) + torch.log(Ng)).mean()
        return loss


class NCESoftmaxLossNS(nn.Module):
    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, x, device, neighbors):
        bsz = x.shape[0]
        x = x.squeeze().cuda()
        x = x - torch.max(x)
        label = torch.arange(bsz).long().to(device)
        loss = self.criterion(x, label)
        return loss
