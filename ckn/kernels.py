# -*- coding: utf-8 -*-
import torch 

def exp(x, alpha):
    """Element wise non-linearity
    kernel_exp is defined as k(x)=exp(alpha * (x-1))
    return:
        same shape tensor as x
    """
    return torch.exp(alpha*(x - 1.))

def poly(x, alpha=None):
    return x.pow(2)


kernels = {
    "exp": exp,
    "poly": poly
}