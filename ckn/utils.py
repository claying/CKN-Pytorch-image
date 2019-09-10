# -*- coding: utf-8 -*-
import torch 
import math
import random
import numpy as np 

EPS = 1e-6


def gaussian_filter_1d(size, sigma=None):
    """Create 1D Gaussian filter
    """
    if size == 1:
        return torch.ones(1)
    if sigma is None:
        sigma = (size - 1.) / (2.*math.sqrt(2))
    m = (size - 1) / 2.
    filt = torch.arange(-m, m+1)
    filt = torch.exp(-filt.pow(2)/(2.*sigma*sigma))
    return filt/torch.sum(filt)

def spherical_kmeans(x, n_clusters, max_iters=100, block_size=None, verbose=True, init=None):
    """Spherical kmeans
    Args:
        x (Tensor n_samples x n_features): data points
        n_clusters (int): number of clusters
    """
    print(x.shape)
    use_cuda = x.is_cuda
    n_samples, n_features = x.size()
    if init is None:
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices.cuda()
        clusters = x[indices]

    prev_sim = np.inf
    tmp = x.new_empty(n_samples)
    assign = x.new_empty(n_samples, dtype=torch.long)
    if block_size is None or block_size == 0:
        block_size = x.shape[0]

    for n_iter in range(max_iters):
        # assign data points to clusters
        for i in range(0, n_samples, block_size):
            end_i = min(i + block_size, n_samples)
            cos_sim = x[i: end_i].mm(clusters.t())
            tmp[i: end_i], assign[i: end_i] = cos_sim.max(dim=-1)
        # cos_sim = x.mm(clusters.t())
        # tmp, assign = cos_sim.max(dim=-1)
        sim = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print("Spherical kmeans iter {}, objective value {}".format(
                n_iter + 1, sim))

        # update clusters
        for j in range(n_clusters):
            index = assign == j
            if index.sum().item() == 0:
                idx = tmp.argmin()
                clusters[j] = x[idx]
                tmp[idx] = 1.
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm().clamp(min=EPS)

        if torch.abs(prev_sim - sim)/(torch.abs(sim)+1e-20) < EPS:
            break
        prev_sim = sim
    return clusters

def normalize_(x, p=2, dim=-1):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    x.div_(norm.clamp(min=EPS))
    return x 

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res

def count_parameters(model):
    count = 0
    for param in model.parameters():
        count += np.prod(param.data.size())
    return count

if __name__ == "__main__":
    x = torch.rand(10000,50)
    x = normalize(x, dim=-1)
    print(x.norm(2, dim=-1))
    z = spherical_kmeans(x, 32)
    print(z)
    print(z.norm(2, dim=-1))
