'''
Author: your name
Date: 2021-03-16 17:49:23
LastEditTime: 2021-03-17 16:03:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /ours/dpp.py
'''
import time
import numpy as np
import math
from scipy.spatial.distance import cdist
from tqdm import tqdm


def dpp(kernel_matrix, max_length, item, epsilon=1E-10):
    """
    fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(kernel_matrix[item])
    # selected_item = item
    selected_items.append(selected_item)

    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    return selected_items


def dpp_sampling(feature, max_iter):
    sampling_set = []
    item_size = feature.shape[0]
    feature_dimension = feature.shape[1]
    max_length = max_iter

    scores = np.ones(item_size)
    feature_vectors = feature

    dis = cdist(feature_vectors, feature_vectors, metric='euclidean')
    # print('dis', dis)
    similarities = np.exp(-dis)
    # print(similarities)
    kernel_matrix = scores.reshape((item_size, 1)) * \
        similarities * scores.reshape((1, item_size))

    # t = time.time()
    for i in range(item_size):
        # kernel_matrix = similarities[i]*similarities*similarities[i].T
        result = dpp(kernel_matrix, max_length, i)
        sampling_set.append(result)
    return sampling_set
