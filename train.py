import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import random
import dgl
import time
import os
import math
from dgl.data import load_data
from tqdm import tqdm
from dgl.nn.pytorch import AvgPooling, MaxPooling, SumPooling
from sklearn.neighbors import NearestNeighbors
from models.graph_encoder import GraphEncoder
from models.logreg import LogReg
from dataset import RandomWalkDataset, batcher
from criterion import NCESoftmaxLossNS, NCESoftmaxLoss_2, NCESoftmaxLoss
from utils import clip_grad_norm, AverageMeter, subgraph_to_nodes, neighbor_nodes
import warnings
import json
warnings.filterwarnings('ignore')

count = 0
best_loss = float('inf')


def set_seed(seed=16):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='GraphSAGE')
    # ========================dataset===========================
    parser.add_argument("--dataset", type=str,
                        default="pubmed", help='cora,citeseer,pubmed')
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="num of samples per batch")
    parser.add_argument("--n_pos", type=int, default=2,
                        help='positive examples per sample')
    parser.add_argument("--batch_size", type=int,
                        default=100, help="batch_size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="number of training epochs")
    parser.add_argument("--n_hops", type=int, default=2,
                        help="number of hops for positive samping")
    parser.add_argument('--patience', type=int, default=30,
                        help='Patient epochs to wait before early stopping.')

    # ========================model==============================
    parser.add_argument("--model", type=str, default="gcn",
                        help="model: gcn/gat/gin")
    parser.add_argument("--readout", type=str, default="avg",
                        help="readout: avg/sum/max")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--output_size", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="number of heads for gat layers")
    parser.add_argument("--dropout", type=float, default=0.7,
                        help="dropout probability")

    # ======================= random walk =======================
    parser.add_argument("--rw_hops", type=int, default=25)
    parser.add_argument("--restart_prob", type=float, default=0.0)

    # ====================== loss function ======================
    parser.add_argument("--nce_k", type=int, default=32)
    parser.add_argument("--nce_t", type=float, default=0.07)
    parser.add_argument("--method", type=str, default='correlation',
                        choices=['correlation', 'origin', 'neighbor', 'gaussian', 'classifier', 'dpp', 'cor_dpp'],
                        help='The method to mask out negative examples when cal loss')
    parser.add_argument("--dpp_value", type=int, default=3)
    parser.add_argument("--ratio", type=float, default=0.75)
    # ======================== optimizer ========================
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight for L2 loss")
    parser.add_argument("--clip_norm", type=float,
                        default=1.0, help="clip norm")

    # ======================= specify folder ====================
    parser.add_argument("--model-path", type=str,
                        default='saved', help="path to save model")
    parser.add_argument("--tb_path", type=str, default=None,
                        help="path to tensorboard")
    parser.add_argument("--load_path", type=str, default=None,
                        help="loading checkpoint at test time")

    # ========================== setting ========================
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--save_freq", type=int,
                        default=100, help="save frequency")
    parser.add_argument("--print-freq", type=int,
                        default=1, help="print frequency")
    parser.add_argument("--tb-freq", type=int,
                        default=250, help="tb frequency")

    args = parser.parse_args()
    return args


def args_update(args):
    args.model_name = "dataset_{}_model_{}_n_layer_{}_batch_size_{}_hidden_size_{}_num_samples_{}_rw_hops_{}_restart_prob_{}_nce_t_{}_nce_k_{}_lr_{}_weight_decay_{}_method_{}".format(
        args.dataset,
        args.model,
        args.n_layers,
        args.batch_size,
        args.hidden_size,
        args.num_samples,
        args.rw_hops,
        args.restart_prob,
        args.nce_t,
        args.nce_k,
        args.lr,
        args.weight_decay,
        args.method,
    )

    if args.load_path is None:
        if args.model_path is None:
            args.model_folder = os.path.join(
                'saved/model_path', args.model_name)
        else:
            args.model_folder = os.path.join(args.model_path, args.model_name)
        if not os.path.isdir(args.model_folder):
            os.makedirs(args.model_folder)
        args.load_path = args.model_folder
    else:
        args.model_folder = args.load_path

    if args.tb_path is None:
        args.tb_folder = os.path.join('saved/tb_path', args.model_name)
    else:
        args.tb_folder = os.path.join(args.tb_path, args.model_name)
    if not os.path.isdir(args.tb_folder):
        os.makedirs(args.tb_folder)
    return args


def evaluate(model, g, args):
    model.eval()
    labels = g.ndata['label'].cuda()
    features = g.ndata['feat'].cuda()
    train_nid = g.ndata['train_mask']
    val_nid = g.ndata['val_mask']
    test_nid = g.ndata['test_mask']
    n_classes = torch.max(labels).item() + 1

    xent = nn.CrossEntropyLoss()

    _, embeds = model.embed(g.int().to(args.gpu), features,
                            return_all_outputs=True, train=False)
    train_embs = embeds[train_nid]
    val_embs = embeds[val_nid]
    test_embs = embeds[test_nid]
    train_lbls = labels[train_nid]
    val_lbls = labels[val_nid]
    test_lbls = labels[test_nid]

    wd = 0.01 if args.dataset == 'citeseer' else 0.0
    accs_train = []
    accs_val = []
    accs_test = []

    for _ in tqdm(range(2)):
        log = LogReg(embeds.shape[1], n_classes)
        log = log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        for _ in range(300):
            log.train()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            # logits = log(embeds)
            # loss = xent(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

        log.eval()
        logits_train = log(train_embs)
        logits_val = log(val_embs)
        logits_test = log(test_embs)
        preds_train = torch.argmax(logits_train, dim=1)
        preds_val = torch.argmax(logits_val, dim=1)
        preds_test = torch.argmax(logits_test, dim=1)
        acc_train = torch.sum(
            preds_train == train_lbls).float() / train_lbls.shape[0]
        acc_val = torch.sum(
            preds_val == val_lbls).float() / val_lbls.shape[0]
        acc_test = torch.sum(
            preds_test == test_lbls).float() / test_lbls.shape[0]
        accs_train.append(acc_train * 100)
        accs_val.append(acc_val * 100)
        accs_test.append(acc_test * 100)
    accs_train = torch.stack(accs_train)
    accs_val = torch.stack(accs_val)
    accs_test = torch.stack(accs_test)
    return accs_train, accs_val, accs_test


def train_batch(g, epoch, train_loader, model, criterion_1, criterion_2, optimizer, args, labels):
    global count, best_loss
    n_batch = train_loader.dataset.total // args.batch_size
    batch_time = AverageMeter()
    data_time = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0
    t = time.time()

    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - t)
        index, graph_q, graph_k = batch
        graph_q, graph_k = graph_q.to(args.gpu), graph_k.to(args.gpu)
        bsz = graph_q.batch_size

        # =======================

        feat_q, feat_k = graph_q.ndata['feat'], graph_k.ndata['feat']
        feat_q, x_q = model(graph_q, feat_q, unpool=False)
        feat_k, x_k = model(graph_k, feat_k, unpool=False)

        # idx_q = np.random.permutation(feat_q.shape[0])
        # idx_k = np.random.permutation(feat_k.shape[0])
        # shuf_feat_q = feat_q[idx_q]
        # shuf_feat_k = feat_k[idx_k]
        out_1 = torch.matmul(feat_q, feat_k.t()) / args.nce_t
        # out_2 = torch.matmul(feat_q, shuf_feat_k.t())/args.nce_t
        # out_3 = torch.matmul(feat_k, shuf_feat_q.t())/args.nce_t
        if args.method == 'correlation':
            """
            Mask out negative example based on correlation between nodes using their embedding.
            The mask ratio can be tuned in the corresponding get mask function in utils.
            """
            cor_q = np.corrcoef(feat_q.detach().cpu())
            cor_k = np.corrcoef(feat_k.detach().cpu())
            cor = (cor_k + cor_q) / 2
            loss_1 = criterion_1(out_1, method=args.method, cor=cor, ratio=args.ratio, device=args.gpu)
        elif args.method == 'neighbor':
            """
            Mask out negative example based on neighbor information where edges exists between nodes.
            """
            # find neighbor in big graph
            one_degree_nodes = [list(neighbor_nodes(g, i, 2)) for i in index]
            # find neighbor index in small graph(the matrix)
            one_degree_nodes_in_index = []
            for i in one_degree_nodes:
                tmp = []
                for j in i:
                    if j in index:
                        tmp.append(index.index(j))
                one_degree_nodes_in_index.append(tmp)
            loss_1 = criterion_1(out_1, method=args.method, neighbor=one_degree_nodes_in_index, device=args.gpu)
        elif args.method == 'origin':
            """
            No additional masks.
            """
            loss_1 = criterion_1(out_1, method=args.method, device=args.gpu)
        elif args.method == 'dpp':
            """
            No additional masks.
            """
            loss_1 = criterion_1(out_1, method=args.method, dpp_value=args.dpp_value, device=args.gpu)
        elif args.method == 'cor_dpp':
            """
            No additional masks.
            """
            cor_q = np.corrcoef(feat_q.detach().cpu())
            cor_k = np.corrcoef(feat_k.detach().cpu())
            cor = (cor_k + cor_q) / 2
            loss_1 = criterion_1(out_1, method=args.method, cor=cor, ratio=args.ratio, dpp_value=args.dpp_value, device=args.gpu)
        elif args.method == 'gaussian':
            """
            Guassian model to fit the pos and neg samples
            need dimension reduction
            """
            loss_1 = criterion_1(out_1, graph_k=graph_k, graph_q=graph_q, feat_q=feat_q, feat_k=feat_k, x_q=x_q, x_k=x_k,
                                 method=args.method, device=args.gpu)
        elif args.method == 'classifier':
            """
            Guassian model to fit the pos and neg samples
            need dimension reduction
            """
            loss_1 = criterion_1(out_1, graph_k=graph_k, graph_q=graph_q, feat_q=feat_q, feat_k=feat_k, x_q=x_q, x_k=x_k,
                                 method=args.method, device=args.gpu)
        # loss_2 = criterion_1(out_2, args.gpu)
        # loss_3 = criterion_1(out_3, args.gpu)

        loss = loss_1  # +0.0*(loss_2 + loss_3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if idx != n_batch:
        #     if(loss < best_loss):
        #         count = 0
        #         best_loss = loss
        #     else:
        #         count += 1

        #     if(count == args.patience):
        #         print('early stopping')
        #         return loss.item(), True

        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())
        batch_time.update(time.time() - t)

        # =======================print info========================
        if (idx + 1) % args.print_freq == 0:
            print(
                "Train:[{0}][{1}/{2}]\t"
                "batch_time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "loss {loss:.3f} ({loss:.3f})\t".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    loss=loss.item(),
                )
            )
    return loss.item(), False


def train(args):
    data = load_data(args)
    g = data[0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    features = g.ndata['feat']
    labels = g.ndata['label']
    accs = []
    train_dataset = RandomWalkDataset(
        graph=g,
        rw_hops=args.rw_hops,
        restart_prob=args.restart_prob,
        n_pos=args.n_pos,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=True
    )

    model = GraphEncoder(
        input_size=features.shape[1],
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.n_layers,
        norm=False,
        dropout=args.dropout,
        gnn_model=args.model,
        readout=args.readout,
    )
    model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion_1 = NCESoftmaxLoss_2()
    criterion_2 = nn.BCEWithLogitsLoss()
    best_epoch = 0
    best_acc_val = 0
    best_acc_test = 0
    for epoch in range(args.epochs):

        # =================== train model =======================

        print('epoch:', epoch)
        print("==> training...")
        model.train()
        loss, flag = train_batch(g,
                                 epoch,
                                 train_loader,
                                 model,
                                 criterion_1,
                                 criterion_2,
                                 optimizer,
                                 args,
                                 labels,
                                 )
        model.eval()
        print("epoch {}, loss {:.3f}".format(
            epoch, loss))

        # ==================== save model ========================

        # if epoch % args.save_freq == 0:
        #     print("==> Saving...")
        #     state = {
        #         "opt": args,
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch,
        #     }
        #     save_file = os.path.join(
        #         args.model_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
        #     )
        #     torch.save(state, save_file)
        #     del state

        # ================== evaluate model ======================

        accs_train, accs_val, accs_test = evaluate(
            model, g, args)
        accs.append(accs_test.max().item())
        if accs_val.max().item() > best_acc_val:
            best_acc_val = accs_val.max().item()
        if accs_test.max().item() > best_acc_test:
            best_acc_test = accs_test.max().item()
            best_epoch = epoch

        time_eval_end = time.time()
        print(
            "epoch {}, acc_train {:.4f}, acc_train_std {:.4f}\t acc_val {:.4f}, acc_val_std {:.4f}\t accs_test {:.4f}, acc_test_std {:.4f}".format(
                epoch,
                accs_train.max().item(),
                accs_train.std().item(),
                accs_val.max().item(),
                accs_val.std().item(),
                accs_test.max().item(),
                accs_test.std().item()))

        if (flag):
            break
    print('best_epoch:', best_epoch, 'best_acc_val:',
          best_acc_val, '\tbest_acc_test:', best_acc_test)
    print('accs', accs)
    return best_acc_test


def optimize_dpp(dataset_name, dpp_value):
    res_dict = {}
    for value in dpp_value:
        args = parse_args()
        args.dpp_value = value
        args.dataset = dataset_name
        args.method = 'dpp'
        args = args_update(args)
        print(args)
        best = 0
        set_seed(16)
        best += train(args)
        # set_seed(15)
        # best += train(args)
        # set_seed(14)
        # best += train(args)
        # set_seed(13)
        # best += train(args)
        # set_seed(17)
        # best += train(args)
        # print(best / 5)
        res_dict[value] = best
    print(res_dict)
    with open("{}_dpp_res.json".format(dataset_name), "w") as fp:
        json.dump(res_dict, fp)


def optimize_cor_ratio(dataset_name, ratio_list):
    res_dict = {}
    for value in ratio_list:
        args = parse_args()
        args.ratio = value
        args.dataset = dataset_name
        args.method = 'correlation'
        args = args_update(args)
        print(args)
        best = 0
        set_seed(16)
        best += train(args)
        # set_seed(15)
        # best += train(args)
        # set_seed(14)
        # best += train(args)
        # set_seed(13)
        # best += train(args)
        # set_seed(17)
        # best += train(args)
        # print(best / 5)
        res_dict[value] = best
    print(res_dict)
    with open("{}_cor_res.json".format(dataset_name), "w") as fp:
        json.dump(res_dict, fp)

def optimize_cor_dpp_ratio(dataset_name, dpp_value_list, ratio_list):
    res_dict = {}
    for ratio in ratio_list:
        for dpp in dpp_value_list:
            args = parse_args()
            args.ratio = ratio
            args.dpp_value = dpp
            args.dataset = dataset_name
            args.method = 'cor_dpp'
            args = args_update(args)
            print(args)
            best = 0
            set_seed(16)
            best += train(args)
            # set_seed(15)
            # best += train(args)
            # set_seed(14)
            # best += train(args)
            # set_seed(13)
            # best += train(args)
            # set_seed(17)
            # best += train(args)
            print(best)
            res_dict['r{}_d{}'.format(ratio, dpp)] = best
    print(res_dict)
    with open("{}_cor_dpp_res.json".format(dataset_name), "w") as fp:
        json.dump(res_dict, fp)


def optimize_batch_size(dataset_name, batch_list):
    res_dict = {}
    for batch_size in batch_list:
        args = parse_args()
        args.dataset = dataset_name
        args.method = 'cor_dpp'
        args.batch_size = batch_size
        if dataset_name == 'cora':
            args.ratio = 0.9300000000000002
            args.dpp_value = 23
        elif dataset_name == 'citeseer':
            args.ratio = 0.8600000000000001
            args.dpp_value = 22
        elif dataset_name == 'pubmed':
            args.ratio = 0.9700000000000002
            args.dpp_value = 24
        else:
            raise ValueError
        args = args_update(args)
        print(args)
        best = 0
        set_seed(16)
        best += train(args)
        # set_seed(15)
        # best += train(args)
        # set_seed(14)
        # best += train(args)
        # set_seed(13)
        # best += train(args)
        # set_seed(17)
        # best += train(args)
        print(best)
        res_dict['{}'.format(batch_size)] = best
    print(res_dict)
    with open("{}_batch_influence.json".format(dataset_name), "w") as fp:
        json.dump(res_dict, fp)


def optimize_rw_hops(dataset_name, rw_list):
    res_dict = {}
    for rw in rw_list:
        args = parse_args()
        args.dataset = dataset_name
        args.method = 'cor_dpp'
        args.rw_hops = rw
        if dataset_name == 'cora':
            args.ratio = 0.9300000000000002
            args.dpp_value = 23
        elif dataset_name == 'citeseer':
            args.ratio = 0.8600000000000001
            args.dpp_value = 22
        elif dataset_name == 'pubmed':
            args.ratio = 0.9700000000000002
            args.dpp_value = 24
        else:
            raise ValueError
        args = args_update(args)
        print(args)
        best = 0
        set_seed(16)
        best += train(args)
        # set_seed(15)
        # best += train(args)
        # set_seed(14)
        # best += train(args)
        # set_seed(13)
        # best += train(args)
        # set_seed(17)
        # best += train(args)
        print(best)
        res_dict['{}'.format(rw)] = best
    print(res_dict)
    with open("{}_rwhops_influence.json".format(dataset_name), "w") as fp:
        json.dump(res_dict, fp)


def optimize_cor_given_dpp(dataset_name, cor_list):
    res_dict = {}
    for cor in cor_list:
        args = parse_args()
        args.dataset = dataset_name
        args.method = 'cor_dpp'
        args.ratio = cor
        if dataset_name == 'cora':
            args.dpp_value = 23
        elif dataset_name == 'citeseer':
            args.dpp_value = 22
        elif dataset_name == 'pubmed':
            args.dpp_value = 24
        else:
            raise ValueError
        args = args_update(args)
        print(args)
        best = 0
        set_seed(16)
        best += train(args)
        # set_seed(15)
        # best += train(args)
        # set_seed(14)
        # best += train(args)
        # set_seed(13)
        # best += train(args)
        # set_seed(17)
        # best += train(args)
        print(best)
        res_dict['{}'.format(cor)] = best
    print(res_dict)
    with open("{}_cor_given_dpp.json".format(dataset_name), "w") as fp:
        json.dump(res_dict, fp)


if __name__ == '__main__':


    # for i in range(10):
    # args = parse_args()
    # args = args_update(args)
    # print(args)
    # best = 0
    # set_seed(16)
    # best += train(args)
    # set_seed(15)
    # best += train(args)
    # set_seed(14)
    # best += train(args)
    # set_seed(13)
    # best += train(args)
    # set_seed(17)
    # best += train(args)
    # print(best / 5)
    """
    param optimizer
    """
    # for dataset_name in ['cora']:
    for dataset_name in ['pubmed']:
        # optimize_batch_size(dataset_name, list(range(20, 301, 20)))
        # optimize_rw_hops(dataset_name, list(range(5, 101, 5)))
        # optimize_dpp(dataset_name, [i + 10 for i in range(15)])
        # optimize_cor_ratio(dataset_name, list(np.arange(0.65, 0.8, 0.01)))
        # optimize_cor_dpp_ratio(dataset_name, [i + 10 for i in range(15)], list(np.arange(0.65, 0.8, 0.01)))
        optimize_cor_given_dpp(dataset_name, list(np.arange(0.5, 0.8, 0.01)))
