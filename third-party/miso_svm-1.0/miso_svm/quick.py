"""Quick caller for Miso svm classifier"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Ghislain Durif, THOTH TEAM INRIA Grenoble Alpes"
__copyright__ = "INRIA"
__credits__ = ["Alberto Bietti", "Dexiong Chen", "Ghislain Durif",
                "Julien Mairal", "Daan Wynen"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Ghislain Durif"
__email__ = "ghislain.durif@inria.fr"
__status__ = "Development"
__date__ = "2017"

import logging
import numpy as np
from timeit import default_timer as timer

import miso_svm._miso as cmiso

EPS_NORM = 0.00001

def quick(features_tr, features_te,
          labels_tr, labels_te,
          eps=1e-4, threads=0, start_exp=-15, end_exp=15,
          add_iter=3, accelerated=True,
          verbose=True, seed=None):

    labels_tr = labels_tr.astype(np.float32).squeeze()
    labels_te = labels_te.astype(np.float32).squeeze()

    logging.info('doing normalization')
    features_tr -= features_tr.mean(axis=1, keepdims=True)
    features_te -= features_te.mean(axis=1, keepdims=True)
    features_tr /= np.maximum(EPS_NORM, np.linalg.norm(features_tr, axis=1, keepdims=True))
    features_te /= np.maximum(EPS_NORM, np.linalg.norm(features_te, axis=1, keepdims=True))

    Ctab = np.arange(start_exp, end_exp)
    N = labels_tr.shape[0]
    max_iterations=1000*N

    Lambdas = []
    accuracys = []

    best_acc = 0
    best_acc_i = -1

    for i,exp in enumerate(Ctab):
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max)

        Lambda = 1 / (2 * N * 2.0**exp)
        start = timer()
        W, iter_count, primals, losses =\
            cmiso.miso_one_vs_rest(features_tr, labels_tr,
                                   Lambda,
                                   max_iterations,
                                   eps=eps,
                                   accelerated=accelerated,
                                   threads=threads,
                                   verbose=verbose,
                                   seed=seed)
        end = timer()

        activations = W.dot(features_te.T)
        predictions = np.argmax(activations, axis=0)

        Lambdas.append(Lambda)
        accuracy = 1 - (np.count_nonzero(predictions - labels_te) / labels_te.shape[0])
        accuracys.append(accuracy)

        logging.info("Lambda = {} / acc = {:.4%} / training in {:.4f} sec"
                        .format(Lambda, accuracy, end-start))

        if accuracy > best_acc:
            best_acc = accuracy
            best_acc_i = i
        if i>=10 and best_acc_i <= i-4:
            break

    print("\n### Best accuracy = {:.4%} for Lambda = {}\n"
                .format(accuracys[best_acc_i], Lambdas[best_acc_i]))

    return accuracys[best_acc_i], Lambdas[best_acc_i]


if __name__=='__main__':
    import sklearn.datasets
    import sys

    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    def load_dataset():
        ds = sklearn.datasets.load_digits()
        X = ds.data.astype('float32')
        X -= X.mean(axis=1, keepdims=True)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y = ds.target.astype('float32')
        return X, Y

    X, Y = load_dataset()
    N = Y.size

    mask = np.random.choice([False, True], N, p=[0.8, 0.2])

    Xtr = X[np.logical_not(mask),]
    Xte = X[mask,]

    Ytr = Y[np.logical_not(mask),]
    Yte = Y[mask,]

    start = timer()
    acc, lamb = quick(Xtr.reshape(Xtr.shape[0], -1),
                      Xte.reshape(Xte.shape[0], -1),
                      Ytr, Yte,
                      eps=1e-4, threads=0, start_exp=-15, end_exp=15,
                      add_iter=3, accelerated=True,
                      verbose=False, seed=None)
    end = timer()
    logging.info("Training MISOS SVM in {:.4f} sec"
                    .format(end - start))
