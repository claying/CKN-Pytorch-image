# -*- coding: utf-8 -*-
import math
import torch 
from torch import nn 
import torch.nn.functional as F
import numpy as np

from scipy import optimize
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin

from . import ops
from .kernels import kernels 
from .utils import spherical_kmeans, gaussian_filter_1d, normalize_, EPS


class CKNLayer(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
        padding="SAME", dilation=1, groups=1, subsampling=1, bias=False,
        kernel_func="exp", kernel_args=[0.5], kernel_args_trainable=False):
        """Define a CKN layer
        Args:
            kernel_args: an iterable object of paramters for kernel function
        """
        if padding == "SAME":
            padding = kernel_size // 2
        else:
            padding = 0
        super(CKNLayer, self).__init__(in_channels, out_channels, kernel_size, 
        stride=1, padding=padding, dilation=dilation, groups=groups, bias=False)
        self.normalize_()
        self.subsampling = subsampling
        self.patch_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        
        self._need_lintrans_computed = True 

        self.kernel_args_trainable = kernel_args_trainable
        self.kernel_func = kernel_func
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == "exp":
            kernel_args = [1./kernel_arg ** 2 for kernel_arg in kernel_args]
        self.kernel_args = kernel_args
        if kernel_args_trainable:
            self.kernel_args = nn.ParameterList(
                [nn.Parameter(torch.Tensor([kernel_arg])) for kernel_arg in kernel_args])

        kernel_func = kernels[kernel_func]
        self.kappa = lambda x: kernel_func(x, *self.kernel_args)

        self.register_buffer("ones",
            torch.ones(1, self.in_channels // self.groups, *self.kernel_size))
        self.init_pooling_filter()

        self.ckn_bias = None
        if bias:
            self.ckn_bias = nn.Parameter(
                torch.zeros(1, self.in_channels // self.groups, *self.kernel_size))

        self.register_buffer("lintrans",
            torch.Tensor(out_channels, out_channels))

    def init_pooling_filter(self):
        size = 2 * self.subsampling + 1
        pooling_filter = gaussian_filter_1d(size, self.subsampling/math.sqrt(2)).view(-1, 1)
        pooling_filter = pooling_filter.mm(pooling_filter.t())
        pooling_filter = pooling_filter.expand(self.out_channels, 1, size, size)
        self.register_buffer("pooling_filter", pooling_filter)

    def train(self, mode=True):
        super(CKNLayer, self).train(mode)
        self._need_lintrans_computed = True 

    def _compute_lintrans(self):
        """Compute the linear transformation factor kappa(ZtZ)^(-1/2)
        Returns:
            lintrans: out_channels x out_channels
        """
        if not self._need_lintrans_computed:
            return self.lintrans
        lintrans = self.weight.view(self.out_channels, -1)
        lintrans = lintrans.mm(lintrans.t())
        lintrans = self.kappa(lintrans)
        lintrans = ops.matrix_inverse_sqrt(lintrans)
        if not self.training:
            self._need_lintrans_computed = False 
            self.lintrans.data = lintrans.data 

        return lintrans

    def _conv_layer(self, x_in):
        """Convolution layer
        Compute x_out = ||x_in|| x kappa(Zt x_in/||x_in||)
        Args:
            x_in: batch_size x in_channels x H x W
            self.filters: out_channels x in_channels x *kernel_size
            x_out: batch_size x out_channels x (H - kernel_size + 1) x (W - kernel_size + 1)
        """
        if self.ckn_bias is not None:
            # compute || x - b ||
            patch_norm_x = F.conv2d(x_in.pow(2), self.ones, bias=None,
                                    stride=1, padding=self.padding,
                                    dilation=self.dilation, 
                                    groups=self.groups)
            patch_norm = patch_norm_x - 2 * F.conv2d(x_in, self.ckn_bias, bias=None,
                stride=1, padding=self.padding, dilation=self.dilation, 
                groups=self.groups)
            patch_norm = patch_norm + self.ckn_bias.pow(2).sum()
            patch_norm = torch.sqrt(patch_norm.clamp(min=EPS))

            x_out = super(CKNLayer, self).forward(x_in)
            bias = torch.sum(
                (self.weight * self.ckn_bias).view(self.out_channels, -1), dim=-1)
            bias = bias.view(1, self.out_channels, 1, 1)
            x_out = x_out - bias
            x_out = x_out / patch_norm.clamp(min=EPS)
            x_out = patch_norm * self.kappa(x_out)
            return x_out

        patch_norm = torch.sqrt(F.conv2d(x_in.pow(2), self.ones, bias=None,
            stride=1, padding=self.padding, dilation=self.dilation, 
            groups=self.groups).clamp(min=EPS))
        # patch_norm = patch_norm.clamp(EPS)

        x_out = super(CKNLayer, self).forward(x_in)
        x_out = x_out / patch_norm.clamp(min=EPS)
        x_out = patch_norm * self.kappa(x_out)
        return x_out

    def _mult_layer(self, x_in, lintrans):
        """Multiplication layer
        Compute x_out = kappa(ZtZ)^(-1/2) x x_in
        Args:
            x_in: batch_size x in_channels x H x W
            lintrans: in_channels x in_channels
            x_out: batch_size x in_channels x H x W
        """
        batch_size, in_c, H, W = x_in.size()
        x_out = torch.bmm(
            lintrans.expand(batch_size, in_c, in_c), x_in.view(batch_size, in_c, -1))
        return x_out.view(batch_size, in_c, H, W)

    def _pool_layer(self, x_in):
        """Pooling layer
        Compute I(z) = \sum_{z'} phi(z') x exp(-\beta_1 ||z'-z||_2^2)
        Args:
            x_in: batch_size x out_channels x H x W
        """
        if self.subsampling <= 1:
            return x_in
        x_out = F.conv2d(x_in, self.pooling_filter, bias=None, 
            stride=self.subsampling, padding=self.subsampling, 
            groups=self.out_channels)
        return x_out

    def forward(self, x_in):
        """Encode function for a CKN layer
        Args:
            x_in: batch_size x in_channels x H x W
        """
        x_out = self._conv_layer(x_in)
        #print(x_out.shape)
        x_out = self._pool_layer(x_out)
        lintrans = self._compute_lintrans()
        x_out = self._mult_layer(x_out, lintrans)
        #print(x_out.shape)
        return x_out

    def extract_2d_patches(self, x):
        """
        x: batch_size x C x H x W
        out: (batch_size * nH * nW) x (C * kernel_size)
        """
        h, w = self.kernel_size
        return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).contiguous().view(-1, self.patch_dim)

    def sample_patches(self, x_in, n_sampling_patches=1000):
        """Sample patches from the given Tensor
        Args:
            x_in (batch_size x in_channels x H x W)
            n_sampling_patches (int): number of patches to sample
        Returns:
            patches: (batch_size x (H - filter_size + 1)) x (in_channels x filter_size)
        """
        patches = self.extract_2d_patches(x_in)
        
        n_sampling_patches = min(patches.size(0), n_sampling_patches)
        patches = patches[:n_sampling_patches]
        return patches

    def unsup_train_(self, patches):
        """Unsupervised training for a CKN layer
        Args:
            patches: n x (in_channels x *kernel_size)
        Updates:
            filters: out_channels x in_channels x *kernel_size
        """
        if self.ckn_bias is not None:
            print("estimating bias")
            m_patches = patches.mean(0)
            self.ckn_bias.data.copy_(m_patches.view_as(self.ckn_bias.data))
            patches -= m_patches
        patches = normalize_(patches)
        block_size = None if self.patch_dim < 1000 else 10 * self.patch_dim
        weight = spherical_kmeans(patches, self.out_channels, block_size=block_size)
        weight = weight.view_as(self.weight.data)
        self.weight.data.copy_(weight)
        self._need_lintrans_computed = True 

    def normalize_(self):
        norm = self.weight.data.view(
            self.out_channels, -1).norm(p=2, dim=-1).view(-1, 1, 1, 1)
        self.weight.data.div_(norm.clamp_(min=EPS))

    def extra_repr(self):
        s = super(CKNLayer, self).extra_repr()
        s += ', subsampling={}'.format(self.subsampling)
        s += ', kernel=({}, {})'.format(self.kernel_func, self.kernel_args)
        return s

class Linear(nn.Linear, LinearModel, LinearClassifierMixin):
    def __init__(self, in_features, out_features, alpha=0.0, fit_bias=True,
                 penalty="l2", maxiter=1000):
        super(Linear, self).__init__(in_features, out_features, fit_bias)
        self.alpha = alpha
        self.fit_bias = fit_bias
        self.penalty = penalty
        self.maxiter = maxiter

    def forward(self, input, scale_bias=1.0):
        # out = super(Linear, self).forward(input)
        out = F.linear(input, self.weight, scale_bias * self.bias)
        return out

    def fit(self, x, y, criterion=None):
        # self.cuda()
        use_cuda = self.weight.is_cuda
        # print(use_cuda)
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        # reduction = criterion.reduction
        # criterion.reduction = 'sum'
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        alpha = self.alpha * x.shape[1] / x.shape[0]
        if self.bias is not None:
            scale_bias = (x ** 2).mean(-1).sqrt().mean().item()
            alpha *= scale_bias ** 2
        self.real_alpha = alpha
        self.scale_bias = scale_bias

        def eval_loss(w):
            w = w.reshape((self.out_features, -1))
            if self.weight.grad is not None:
                self.weight.grad = None
            if self.bias is None:
                self.weight.data.copy_(torch.from_numpy(w))
            else:
                if self.bias.grad is not None:
                    self.bias.grad = None
                self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
                self.bias.data.copy_(torch.from_numpy(w[:, -1]))
            y_pred = self(x, scale_bias=scale_bias).squeeze_(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            if alpha != 0.0:
                if self.penalty == "l2":
                    penalty = 0.5 * alpha * torch.norm(self.weight)**2
                elif self.penalty == "l1":
                    penalty = alpha * torch.norm(self.weight, p=1)
                    penalty.backward()
                loss = loss + penalty
            return loss.item()

        def eval_grad(w):
            dw = self.weight.grad.data
            if alpha != 0.0:
                if self.penalty == "l2":
                    dw.add_(alpha, self.weight.data)
            if self.bias is not None:
                db = self.bias.grad.data
                dw = torch.cat((dw, db.view(-1, 1)), dim=1)
            return dw.cpu().numpy().ravel().astype("float64")

        w_init = self.weight.data
        if self.bias is not None:
            w_init = torch.cat((w_init, 1./scale_bias * self.bias.data.view(-1, 1)), dim=1)
        w_init = w_init.cpu().numpy().astype("float64")

        w = optimize.fmin_l_bfgs_b(
            eval_loss, w_init, fprime=eval_grad, maxiter=self.maxiter, disp=0)
        if isinstance(w, tuple):
            w = w[0]

        w = w.reshape((self.out_features, -1))
        self.weight.grad.data.zero_()
        if self.bias is None:
            self.weight.data.copy_(torch.from_numpy(w))
        else:
            self.bias.grad.data.zero_()
            self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
            self.bias.data.copy_(scale_bias * torch.from_numpy(w[:, -1]))
        # criterion.reduction = reduction

    def fit2(self, x, y, criterion=None):
        from miso_svm import MisoClassifier
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
        scale_bias = np.sqrt((x ** 2).mean(-1)).mean()
        print(scale_bias)
        alpha = self.alpha * scale_bias ** 2 * x.shape[1]
        alpha /= x.shape[0]
        x = np.hstack([x, scale_bias * np.ones((x.shape[0], 1), dtype=x.dtype)])
        y = y.astype('float32')
        clf = MisoClassifier(Lambda=alpha, eps=1e-04, max_iterations=100 * x.shape[0], verbose=False)
        clf.fit(x, y)
        self.weight.data.copy_(torch.from_numpy(clf.W[:, :-1]))
        self.bias.data.copy_(scale_bias * torch.from_numpy(clf.W[:, -1]))

    def decision_function(self, x):
        x = torch.from_numpy(x)
        if self.weight.is_cuda:
            x = x.cuda()
        return self(x).data.cpu().numpy()

    def predict(self, x):
        return np.argmax(self.decision_function(x), axis=1)

    def predict_proba(self, x):
        return self._predict_proba_lr(x)

    @property
    def coef_(self):
        return self.weight.data.cpu().numpy()

    @property
    def intercept_(self):
        return self.bias.data.cpu().numpy()

class Preprocessor(nn.Module):
    def __init__(self):
        super(Preprocessor, self).__init__()
        self.fitted = True

    def forward(self, input):
        out = input - input.mean(dim=1, keepdim=True)
        return out / out.norm(dim=1, keepdim=True).clamp(min=EPS)

    def fit(self, input):
        pass

    def fit_transform(self, input):
        self.fit(input)
        return self(input)
