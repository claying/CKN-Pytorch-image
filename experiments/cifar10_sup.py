import os
import copy
import argparse
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ckn.data import create_dataset
from ckn.utils import accuracy, count_parameters
from ckn.models import SUPMODELS
from ckn.loss import LOSS

from timeit import default_timer as timer


def load_args():
    parser = argparse.ArgumentParser(
        description="CKN for CIFAR10 image classification")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--datapath', type=str, default='../data/cifar-10/cifar_white.mat',
                        help='path to the dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--model', default='ckn14', choices=list(SUPMODELS.keys()), help='which model to use')
    parser.add_argument(
        '--sampling-patches', type=int, default=150000, help='number of subsampled patches for initilization')
    parser.add_argument('--lr', default=1.0, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--alternating', action='store_true', help='use alternating opitmization')
    parser.add_argument('--alpha', default=0.1, type=float, help='regularization parameter')
    parser.add_argument('--loss', default='hinge', choices=list(LOSS.keys()), help='loss function')
    parser.add_argument('--outpath', type=str, default=None, help='output path')
    parser.add_argument('--augmentation', action='store_true', help='data augmentation')
    args = parser.parse_args()
    args.gpu = torch.cuda.is_available()
    
    return args

def sup_train(model, data_loader, args):
    criterion = LOSS[args.loss]()
    if args.alternating:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        alpha = args.alpha * model.out_features / args.batch_size
        optimizer = optim.SGD([
            {'params': model.features.parameters()},
            {'params': model.classifier.parameters(), 'weight_decay': alpha}], lr=args.lr, momentum=0.9)
    # lr_scheduler = None
    if args.model == 'ckn14':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [90, 100], gamma=0.1)
        if args.augmentation:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [90, 120], gamma=0.1)
    else:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 85, 100], gamma=0.1)
        if args.augmentation:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 100, 130], gamma=0.1)

    if args.gpu:
        model.cuda()
    print("Initialing CKN")
    tic = timer()
    model.unsup_train_ckn(
        data_loader['init'], args.sampling_patches, use_cuda=args.gpu)
    toc = timer()
    print("Finished, elapsed time: {:.2f}min".format((toc - tic)/60))

    epoch_loss = None
    best_loss = float('inf')
    best_acc = 0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)
        if args.alternating or epoch == 0:
            model.train(False)
            tic = timer()
            model.unsup_train_classifier(
                data_loader['train'], criterion=criterion, use_cuda=args.gpu)
            toc = timer()
            print('Last layer trained, elapsed time: {:.2f}s'.format(toc - tic))
            if not args.alternating:
                optimizer.param_groups[-1]['weight_decay'] = model.classifier.real_alpha

        for phase in ['train', 'val']:
            if phase == 'train':
                if lr_scheduler is not None and epoch > 0:
                    try:
                        lr_scheduler.step(metrics=epoch_loss)
                    except:
                        lr_scheduler.step()
                print("current LR: {}".format(
                            optimizer.param_groups[0]['lr']))
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_acc = 0

            tic = timer()
            for data, target in data_loader[phase]:
                size = data.size(0)
                if args.gpu:
                    data = data.cuda()
                    target = target.cuda()

                # forward
                if phase == 'train':
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    pred = output.data.argmax(dim=1)
                    loss.backward()
                    optimizer.step()
                    model.normalize_()
                else:
                    with torch.no_grad():
                        output = model(data)
                        loss = criterion(output, target)
                        pred = output.data.argmax(dim=1)
                
                running_loss += loss.item() * size
                running_acc += torch.sum(pred == target.data).item()
            toc = timer()

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_acc / len(data_loader[phase].dataset)
            
            print('{} Loss: {:.4f} Acc: {:.2f}% Elapsed time: {:.2f}s'.format(
                    phase, epoch_loss, epoch_acc * 100, toc - tic))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())
        print()

    print('Best epoch: {}'.format(best_epoch + 1))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_weights)

    return best_acc

def main():
    args = load_args()
    print(args)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)

    init_dset = create_dataset(args.datapath)
    train_dset = create_dataset(args.datapath, dataugmentation=args.augmentation)
    print(train_dset.train_data.shape)

    loader_args = {}
    if args.gpu:
        loader_args = {'pin_memory': True}
    init_loader = DataLoader(
        init_dset, batch_size=64, shuffle=False, num_workers=2, **loader_args)
    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, num_workers=4, **loader_args)

    model = SUPMODELS[args.model](alpha=args.alpha)
    print(model)
    nb_params = count_parameters(model)
    print('number of paramters: {}'.format(nb_params))

    test_dset = create_dataset(args.datapath, train=False)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=2, **loader_args)
    
    data_loader = {'init': init_loader, 'train': train_loader, 'val': test_loader}
    tic = timer()
    score = sup_train(model, data_loader, args)
    toc = timer()
    training_time = (toc - tic) / 60
    print("Final accuracy: {:6.2f}%, elapsed time: {:.2f}min".format(score * 100, training_time))

    # y_pred, y_true = model.predict(test_loader, use_cuda=args.gpu)
    # scores = accuracy(y_pred, y_true, (1,))
    # print(scores)
    if args.outpath is not None:
        import csv
        table = {'acc': score, 'training time': training_time}
        with open(args.outpath + '/metric.csv', 'w') as f:
            w = csv.DictWriter(f, table.keys())
            w.writeheader()
            w.writerow(table)

        torch.save({
            'args': args,
            'state_dict': model.state_dict()},
            args.outpath + '/model.pkl')



if __name__ == '__main__':
    main()
