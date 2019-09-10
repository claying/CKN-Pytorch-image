# -*- coding: utf-8 -*-
import torch
from torch import nn 
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import cross_val_score

from timeit import default_timer as timer

from .layers import CKNLayer, Linear, Preprocessor
from miso_svm import MisoClassifier


class CKNSequential(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_sizes, 
                 subsamplings, kernel_funcs=None, kernel_args_list=None,
                 kernel_args_trainable=False, **kwargs):

        assert len(out_channels_list) == len(kernel_sizes) == len(subsamplings), "incompatible dimensions"
        super(CKNSequential, self).__init__()

        self.n_layers = len(out_channels_list)
        self.in_channels = in_channels
        self.out_channels = out_channels_list[-1]
        
        ckn_layers = []

        for i in range(self.n_layers):
            if kernel_funcs is None:
                kernel_func = "exp"
            else:
                kernel_func = kernel_funcs[i] 
            if kernel_args_list is None:
                kernel_args = 0.5
            else:
                kernel_args = kernel_args_list[i]
            
            ckn_layer = CKNLayer(in_channels, out_channels_list[i],
                                 kernel_sizes[i], subsampling=subsamplings[i],
                                 kernel_func=kernel_func, kernel_args=kernel_args,
                                 kernel_args_trainable=kernel_args_trainable, **kwargs)

            ckn_layers.append(ckn_layer)
            in_channels = out_channels_list[i]

        self.ckn_layers = nn.Sequential(*ckn_layers)

    def __getitem__(self, idx):
        return self.ckn_layers[idx]

    def __len__(self):
        return len(self.ckn_layers)

    def __iter__(self):
        return self.ckn_layers._modules.values().__iter__()

    def forward_at(self, x, i=0):
        assert x.size(1) == self.ckn_layers[i].in_channels, "bad dimension"
        return self.ckn_layers[i](x)

    def forward(self, x):
        return self.ckn_layers(x)

    def representation(self, x, n=0):
        if n == -1:
            n = self.n_layers
        for i in range(n):
            x = self.forward_at(x, i)
        return x 

    def normalize_(self):
        for module in self.ckn_layers:
            module.normalize_()

    def unsup_train_(self, data_loader, n_sampling_patches=100000, use_cuda=False, top_layers=None):
        """
        x: size x C x H x W 
        top_layers: module object represents layers before this layer
        """
        self.train(False)
        if use_cuda:
            self.cuda()
        with torch.no_grad():
            for i, ckn_layer in enumerate(self.ckn_layers):
                print()
                print('-------------------------------------')
                print('   TRAINING LAYER {}'.format(i + 1))
                print('-------------------------------------')
                n_patches = 0 
                try:
                    n_patches_per_batch = (n_sampling_patches + len(data_loader) - 1) // len(data_loader) 
                except:
                    n_patches_per_batch = 1000
                patches = torch.Tensor(n_sampling_patches, ckn_layer.patch_dim)
                if use_cuda:
                    patches = patches.cuda()

                for data, _ in data_loader:
                    if use_cuda:
                        data = data.cuda()
                    # data = Variable(data, volatile=True)
                    if top_layers is not None:
                        data = top_layers(data)
                    data = self.representation(data, i)
                    data_patches = ckn_layer.sample_patches(data.data, n_patches_per_batch)
                    size = data_patches.size(0)
                    if n_patches + size > n_sampling_patches:
                        size = n_sampling_patches - n_patches
                        data_patches = data_patches[:size]
                    patches[n_patches: n_patches + size] = data_patches
                    n_patches += size 
                    if n_patches >= n_sampling_patches:
                        break

                print("total number of patches: {}".format(n_patches))
                patches = patches[:n_patches]
                ckn_layer.unsup_train_(patches)

class CKNet(nn.Module):
    def __init__(self, nclass, in_channels, out_channels_list, kernel_sizes, 
                 subsamplings, kernel_funcs=None, kernel_args_list=None,
                 kernel_args_trainable=False, image_size=32,
                 fit_bias=True, alpha=0.0, maxiter=1000, **kwargs):
        super(CKNet, self).__init__()
        self.features = CKNSequential(
            in_channels, out_channels_list, kernel_sizes, 
            subsamplings, kernel_funcs, kernel_args_list,
            kernel_args_trainable, **kwargs)

        out_features = out_channels_list[-1]
        factor = 1
        for s in subsamplings:
            factor *= s
        factor = (image_size - 1) // factor + 1
        self.out_features = factor * factor * out_features
        self.nclass = nclass

        self.initialize_scaler()
        self.classifier = Linear(
            self.out_features, nclass, fit_bias=fit_bias, alpha=alpha, maxiter=maxiter)

    def initialize_scaler(self, scaler=None):
        pass

    def forward(self, input):
        features = self.representation(input)
        return self.classifier(features)

    def representation(self, input):
        features = self.features(input).view(input.shape[0], -1)
        if hasattr(self, 'scaler'):
            features = self.scaler(features)
        return features

    def unsup_train_ckn(self, data_loader, n_sampling_patches=1000000,
                        use_cuda=False):
        self.features.unsup_train_(data_loader, n_sampling_patches, use_cuda=use_cuda)

    def unsup_train_classifier(self, data_loader, criterion=None, use_cuda=False):
        encoded_train, encoded_target = self.predict(
            data_loader, only_representation=True, use_cuda=use_cuda)
        self.classifier.fit(encoded_train, encoded_target, criterion)

    def predict(self, data_loader, only_representation=False, use_cuda=False):
        self.eval()
        if use_cuda:
            self.cuda()
        n_samples = len(data_loader.dataset)
        batch_start = 0
        for i, (data, target) in enumerate(data_loader):
            batch_size = data.shape[0]
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                if only_representation:
                    batch_out = self.representation(data).data.cpu()
                else:
                    batch_out = self(data).data.cpu()
            if i == 0:
                output = batch_out.new_empty(n_samples, batch_out.shape[-1])
                target_output = target.new_empty(n_samples)
            output[batch_start:batch_start+batch_size] = batch_out
            target_output[batch_start:batch_start+batch_size] = target
            batch_start += batch_size
        return output, target_output

    def normalize_(self):
        self.features.normalize_()

    def print_norm(self):
        norms = []
        with torch.no_grad():
            for module in self.features:
                norms.append(module.weight.sum().item())
            norms.append(self.classifier.weight.sum().item())
        print(norms)

class UnsupCKNet(CKNet):
    def initialize_scaler(self):
        self.scaler = Preprocessor()

    def unsup_train(self, data_loader, n_sampling_patches=1000000,
                    use_cuda=False):
        self.train(False)
        print("Training CKN layers")
        tic = timer()
        self.unsup_train_ckn(data_loader, n_sampling_patches, use_cuda=use_cuda)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))
        print()
        print("Training classifier")
        tic = timer()
        self.unsup_train_classifier(data_loader, use_cuda=use_cuda)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))

    def unsup_cross_val(self, data_loader, test_loader=None, n_sampling_patches=500000,
                        alpha_grid=None, kfold=5, scoring='accuracy',
                        use_cuda=False):
        self.train(False)
        if alpha_grid is None:
            alpha_grid = np.arange(-15, 15)
        print("Training CKN layers")
        tic = timer()
        self.unsup_train_ckn(data_loader, n_sampling_patches, use_cuda=use_cuda)
        toc = timer()
        print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))
        print()
        print("Start cross-validation")
        best_score = -float('inf')
        best_alpha = 0
        tic = timer()
        encoded_train, encoded_target = self.predict(
            data_loader, only_representation=True, use_cuda=use_cuda)

        n_samples = len(encoded_target) * (1 - 1. / kfold)

        clf = self.classifier
        n_jobs = None if use_cuda else -1
        iter_since_best = 0
        print(encoded_train.shape)
        print(encoded_target.shape)

        if test_loader is not None:
            encoded_test, encoded_label = self.predict(
                test_loader, only_representation=True, use_cuda=use_cuda)

        encoded_train = encoded_train.numpy()
        encoded_target = encoded_target.numpy().astype('float32')
        encoded_test = encoded_test.numpy()
        encoded_label = encoded_label.numpy().astype('float32')

        for alpha in alpha_grid:
            alpha = 1. / (2. * n_samples  * 2.**alpha)
            #alpha = 1. / (2. * 2. ** alpha)
            print("lambda={}".format(alpha))
            clf = MisoClassifier(
                Lambda=alpha, max_iterations=int(1000*n_samples), verbose=True, seed=31, threads=0)
            if test_loader is None:
                score = cross_val_score(clf, encoded_train,
                                        encoded_target,
                                        cv=kfold, scoring=scoring, n_jobs=n_jobs)
                score = score.mean()
            else:
                clf.fit(encoded_train, encoded_target)
                score = clf.score(encoded_test, encoded_label)
            print("val score={}".format(score))
            if score > best_score:
                best_score = score
                best_alpha = alpha
                iter_since_best = 0
            else:
                iter_since_best += 1
                if iter_since_best >= 3:
                    break
        print("best lambda={}, best val score={}".format(best_alpha, best_score))
        if test_loader is None:
            clf = MisoClassifier(
                Lambda=best_alpha, max_iterations=int(1000*n_samples), verbose=True, seed=31, threads=0)
            clf.fit(encoded_train, encoded_target)
            toc = timer()
            #self.classifier.weight.data.copy_(torch.from_numpy(clf.coef_))
            self.classifier.weight.data.copy_(torch.from_numpy(clf.W))
            print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))
        return best_score

class UnsupCKNetCifar10(UnsupCKNet):
    def __init__(self, filters, kernel_sizes, subsamplings, sigma):
        super(UnsupCKNetCifar10, self).__init__(
            10, 3, filters, kernel_sizes, subsamplings,
            kernel_args_list=sigma, fit_bias=False, maxiter=5000)

class SupCKNetCifar10_5(CKNet):
    def __init__(self, alpha=0.0, **kwargs):
        kernel_sizes = [3, 1, 3, 1, 3]
        filters = [128, 128, 128, 128, 128]
        subsamplings = [2, 1, 2, 1, 3]
        kernel_funcs = ['exp', 'poly', 'exp', 'poly', 'exp']
        kernel_args_list = [0.5, 2, 0.5, 2, 0.5]
        super(SupCKNetCifar10_5, self).__init__(
            10, 3, filters, kernel_sizes, subsamplings, kernel_funcs=kernel_funcs,
            kernel_args_list=kernel_args_list, fit_bias=True, alpha=alpha, maxiter=5000, **kwargs)

class SupCKNetCifar10_14(CKNet):
    def __init__(self, alpha=0.0, **kwargs):
        kernel_sizes = [3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1]
        filters = [256, 128, 256, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        subsamplings = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2]
        kernel_funcs = ['exp', 'poly', 'exp', 'poly', 'exp', 'poly', 'exp', 'poly',
                        'exp', 'poly', 'exp', 'poly', 'exp', 'poly']
        kernel_args_list = [0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2]
        super(SupCKNetCifar10_14, self).__init__(
            10, 3, filters, kernel_sizes, subsamplings, kernel_funcs=kernel_funcs,
            kernel_args_list=kernel_args_list, fit_bias=True, alpha=alpha, maxiter=5000, **kwargs)

SUPMODELS = {
    'ckn14': SupCKNetCifar10_14,
    'ckn5': SupCKNetCifar10_5
}
