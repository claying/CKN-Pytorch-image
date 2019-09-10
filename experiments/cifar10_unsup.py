import os
import argparse
import torch
from torch.utils.data import DataLoader

from ckn.data import create_dataset
from ckn.utils import accuracy
from ckn.models import UnsupCKNetCifar10


def load_args():
    parser = argparse.ArgumentParser(
        description="unsup CKN for CIFAR10 image classification")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--datapath', type=str, default='../data/cifar-10/cifar_white.mat',
                        help='path to the dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--filters', nargs='+', type=int, help='number of filters')
    parser.add_argument('--subsamplings', nargs='+', type=int, help='sampling routine')
    parser.add_argument('--kernel-sizes', nargs='+', type=int, help='kernel sizes')
    parser.add_argument(
        '--sigma', nargs='+', type=float, default=None, help='parameters for dot-product kernel')
    parser.add_argument(
        '--sampling-patches', type=int, default=1000000, help='number of subsampled patches for K-means')
    parser.add_argument('--cv', action='store_true', 
        help='if True perform model select with cross validation, else on test set')
    args = parser.parse_args()
    args.gpu = torch.cuda.is_available()

    nlayers = len(args.filters)
    if args.sigma is None:
        args.sigma = [0.6] * nlayers
    
    return args


def main():
    args = load_args()
    print(args)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    train_dset = create_dataset(args.datapath)
    print(train_dset.train_data.shape)

    loader_args = {}
    if args.gpu:
        loader_args = {'pin_memory': True}
    data_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=False, num_workers=2, **loader_args)

    model = UnsupCKNetCifar10(
        args.filters, args.kernel_sizes, args.subsamplings, args.sigma)
    print(model)

    test_dset = create_dataset(args.datapath, train=False)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=2, **loader_args)
    # model.unsup_train(
    #     data_loader, n_sampling_patches=args.sampling_patches, use_cuda=args.gpu)
    if args.cv:
        model.unsup_cross_val(
            data_loader, test_loader=None, n_sampling_patches=args.sampling_patches, use_cuda=args.gpu)
        y_pred, y_true = model.predict(test_loader, use_cuda=args.gpu)
        score = accuracy(y_pred, y_true, (1,))
    else:
        score = model.unsup_cross_val(
            data_loader, test_loader=test_loader, n_sampling_patches=args.sampling_patches, use_cuda=args.gpu)
    print(score)

    #test_dset = create_dataset(args.datapath, train=False)
    #test_loader = DataLoader(
    #    test_dset, batch_size=args.batch_size, shuffle=False, num_workers=2, **loader_args)
    # y_pred, y_true = model.predict(test_loader, use_cuda=args.gpu)
    # scores = accuracy(y_pred, y_true, (1,))
    # print(scores)



if __name__ == '__main__':
    main()
