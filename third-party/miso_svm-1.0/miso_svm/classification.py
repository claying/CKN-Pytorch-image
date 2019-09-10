#! /usr/bin/env python

"""run Miso svm classifier on features"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Daan Wynen, THOTH TEAM INRIA Grenoble Alpes"
__copyright__ = "INRIA"
__credits__ = ["Alberto Bietti", "Dexiong Chen", "Ghislain Durif",
                "Julien Mairal", "Daan Wynen"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Ghislain Durif"
__email__ = "ghislain.durif@inria.fr"
__status__ = "Development"
__date__ = "2017"


import numpy as np

import logging

from miso_svm.miso import MisoClassifier

from sklearn import model_selection, metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from datetime import datetime

EPS_NORM = 0.00001

def run(features_tr, features_te,
        labels_tr, labels_te,
        out_file=None, threads=0, start_exp=-15, end_exp=15,
        add_iter=3,
        confusion_matrix=True, do_cv=False,
        verbose=True, seed=None):
    """run miso svm classification on features and labels

    Args:
        features_tr (np.array): Matrix of features (observations in rows)
            in training set.
        features_te (np.array): Matrix of features (observations in rows)
            in test set.
        labels_tr (np.array): Matrix of labels (observations in rows)
            in training set.
        labels_te (np.array): Matrix of labels (observations in rows)
            in test set.
        out_file (string): File to store final classifier in. Should end
            in '.npz'
        threads (int): Number of OpenMP threads to use, default is 0.
        start_exp (int): Parameter search starting point, default is -15.
        end_exp (int): Parameter search end point, default is 15.
        add_iter (int): How many iterations to continue before
            accepting current best iteration, default is 3.
        confusion_matrix (bool): should confusion matrix be ploted or not.
        do_cv (bool): If False, do not do cross validation. Instead, use
            test accuracy to select model, default is False.
        verbose (int): 0 or 1, indicates verbosity in C++ code, default is 0.
        seed (int): Random seed for the SVM, default is None.

    """

    logging.info('Training feature map sparsity: {:5.2f}'.format(100*((features_tr==0).sum()/features_tr.size)))
    labels_tr = labels_tr.astype(np.float32).squeeze()
    logging.info('Test feature map sparsity: {:5.2f}'.format(100*((features_te==0).sum()/features_te.size)))
    labels_te = labels_te.astype(np.float32).squeeze()

    logging.info("Train shape: {}".format(features_tr.shape))
    logging.info("Test shape:  {}".format(features_te.shape))


    logging.info('doing normalization')
    features_tr -= features_tr.mean(axis=1, keepdims=True)
    features_te -= features_te.mean(axis=1, keepdims=True)
    features_tr /= np.maximum(EPS_NORM, np.linalg.norm(features_tr, axis=1, keepdims=True))
    features_te /= np.maximum(EPS_NORM, np.linalg.norm(features_te, axis=1, keepdims=True))

    logging.info('shuffling training data')
    shuffle_in_unison_scary(features_tr, labels_tr)

    start_time = datetime.now()

    logging.info('\n\n')
    logging.info('==================== START CLASSIFICATION ===================\n')
    logging.info('  Starting time: {0}\n'.format(start_time))
    logging.info('=============================================================\n')


    clf = cv_C_only(features_te, features_tr, labels_te, labels_tr,
                    start_exp, end_exp, seed,
                    add_iter, do_cv, verbose, threads)

    predictions_te = clf.predict(features_te)
    acc_te = clf.score(features_te, labels_te)
    logging.info("\n\n\tBest Acc_test: {:6.2f}%\n".format(100 * acc_te))

    end_time = datetime.now()
    logging.info('============================================================\n')
    logging.info('  End time: {0}\n'.format(end_time))
    logging.info('  Time taken: {0}\n'.format(end_time - start_time))
    logging.info('============================================================\n')
    logging.info('saving classifier to {}'.format(out_file))
    print("{:.2%}".format(acc_te))  # for scripts that expect the score in the last line

    if(out_file is not None):
        np.savez(out_file, clf=clf)

    if confusion_matrix:
        try:
            cm = confusion_matrix(labels_te, predictions_te)
            plot_confusion_matrix(cm, list(set(labels_tr)))
            plt.show(block=False)
            confmat_img_fname = 'confusion_matrix.png'
            plt.savefig(confmat_img_fname, bbox_inches='tight')
        except:
            pass


# shuffles two arrays along the first axis, with the same permutation
# taken from http://stackoverflow.com/q/4601373/393885
def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def cv_C_only(features_te, features_tr, labels_te, labels_tr, start_exp, end_exp, seed,
    add_iter=3, do_cv=True, verbose=0, threads=0, **kwargs):
    # manual grid search {{{

    best_score = -1
    best_acc_te = -1
    best_c = 0
    CV_FOLDS = 5 if do_cv else 1
    CV_TEST_PROPORTION = 0.2
    N = int(labels_tr.shape[0]*(1-CV_TEST_PROPORTION)) if do_cv else labels_tr.shape[0]
    np.random.seed(seed)
    splits = StratifiedShuffleSplit(CV_FOLDS, test_size=CV_TEST_PROPORTION)#, random_state=1)
    Ctab = np.arange(start_exp, end_exp)


    iter_since_best = 0
    start_time = datetime.now()
    last_time = start_time
    for exp in Ctab:
        Lambda = 1 / (2 * N * 2.0**exp)

        clf2 = MisoClassifier(Lambda=Lambda,
                              max_iterations=1000*N,
                              verbose=verbose,
                              threads=threads,
                              seed=seed)
        clf2.fit(features_tr, labels_tr)
        acc_tr = clf2.score(features_tr, labels_tr)
        acc_te = clf2.score(features_te, labels_te)

        if do_cv:
            clf = MisoClassifier(Lambda=Lambda,
                                  max_iterations=1000*N,
                                  verbose=verbose,
                                  threads=threads,
                                  seed=seed)
            cv_scores = model_selection.cross_val_score(clf, features_tr, labels_tr, cv=splits.split(features_tr, labels_tr), n_jobs=1)
            cv_i = np.mean(cv_scores)
            cv_i_std = np.std(cv_scores)
            now_time = datetime.now()
            logging.info("{:%H:%M:%S}\t{}\tLambda= {:<9.4e} cv={:6.2f}% (std={:6.2f})"
                    .format(now_time, str(now_time-last_time).split('.')[0], Lambda, cv_i * 100, cv_i_std * 100))
        else:
            now_time = datetime.now()
            logging.info("{:%H:%M:%S}\t{}\tLambda= {:<9.4e}\t"
                    .format(now_time, str(now_time-last_time).split('.')[0], Lambda))
        last_time = now_time

        logging.info("\tAcc_train: {:6.2f}% |".format(100 * acc_tr))
        logging.info("\tAcc_test: {:6.2f}% |".format(100 * acc_te))

        if do_cv:
            if cv_i > best_score:
                best_score = cv_i
                best_c = Lambda
                best_clf = clf2
                logging.info(' *')
            else:
                logging.info('  ')

        if acc_te > best_acc_te:
            best_acc_te = acc_te
            iter_since_best = 0
            # if we're not doing CV, use acc_te instead of CV score
            if not do_cv:
                best_score = acc_te
                best_c = Lambda
                best_clf = clf2
            logging.info(' +')
        else:
            iter_since_best += 1
            if iter_since_best >= add_iter:
                break
            logging.info('  ')

    # manual grid search }}}

    return best_clf


def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def clamp_values(arr, clampval = 800):
    """clamps array input in-place"""
    to_clamp = np.abs(arr) > clampval
    if (to_clamp).any():
        logging.info('clamping {} values to +- {}'.format(to_clamp.sum(), clampval))
    else:
        logging.info('no clamping necessary')
    np.clip(arr, -clampval, clampval, out=arr)


def heal_nans(arr):
    """remove NaNs if any and replace them by the average of the column"""
    if np.isnan(arr).any():
        logging.info('replacing {} NaN values in array.'.format(np.isnan(arr).sum()))
    else:
        logging.info('no NaNs in array')
        return

    for j in xrange(arr.shape[1]):
        nans = np.isnan(arr[:, j])
        if not nans.any():
            continue
        if nans.all():
            arr[:, j] = 0
            continue
        arr[:, j][nans] = np.mean(arr[:, j][np.logical_not(nans)])
