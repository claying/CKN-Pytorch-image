"""Miso svm classifier"""

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

import miso_svm._miso as cmiso
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

class MisoClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 Lambda=0.01,
                 eps=1e-4,
                 max_iterations=None,
                 accelerated=True,
                 threads=-1,
                 verbose=0,
                 seed=None):
        self.Lambda = Lambda
        self.eps = eps
        self.max_iterations = max_iterations
        self.accelerated = accelerated
        self.threads = threads
        self.verbose = verbose
        if seed is not None:
            self.seed = seed
        else:
            # set the seed, so that we can retrieve it later if needed
            self.seed = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max)

    def fit(self, X, y):
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        assert X.dtype == y.dtype
        assert X.dtype == np.float32    # TODO: might want to drop that later

        if self.max_iterations is None:
            self.max_iterations = 1000 * X.shape[0]
        self.W, self.iter_count, self.primals, self.losses =\
                cmiso.miso_one_vs_rest(X, y,
                                           self.Lambda, self.max_iterations,
                                           eps=self.eps,
                                           accelerated=self.accelerated,
                                           threads=self.threads,
                                           verbose=self.verbose,
                                           seed=self.seed)
        self.W = self.W.astype('float32')
        self.iter_count = self.iter_count.astype('intc')
        self.primals = self.primals.astype('float32')
        self.losses = self.losses.astype('float32')

    def predict(self, X):
        activations = self.W.dot(X.T)
        predictions = np.argmax(activations, axis=0)
        return predictions


def load_dataset():
    ds = sklearn.datasets.load_digits()
    X = ds.data.astype('float32')
    X -= X.mean(axis=1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = ds.target.astype('float32')
    return X, Y

if __name__=='__main__':
    import sklearn.datasets
    from sklearn import svm, model_selection, metrics
    from sklearn.model_selection import StratifiedShuffleSplit
    from datetime import datetime
    cv_folds = 20

    X, Y = load_dataset()
    N = Y.size
    splits = StratifiedShuffleSplit(cv_folds, test_size=0.2)#, random_state=1)

    def test_clf(clf, name):
        start_time = datetime.now()
        scores = sklearn.model_selection.cross_val_score(clf, X, Y, cv=splits.split(X,Y))
        time_taken = datetime.now() - start_time
        print('{}: {:6.2f}% in {:02}:{:07.4f} ({})'
                        .format(name,
                                100*np.mean(scores),
                                (time_taken.seconds//60)%60,
                                (time_taken.seconds + time_taken.microseconds/1000000)%60,
                                ', '.join(['{:6.2f}'.format(f) for f in scores])))

    for L in np.float(2)**np.arange(2, -15, -1):
        print('Lambda = {}'.format(L))
        clf_miso = MisoClassifier(verbose=0, Lambda=L, threads=1)
        clf_lsvc = svm.LinearSVC(loss='squared_hinge', C=1/(N * clf_miso.Lambda), fit_intercept=False)

        test_clf(clf_miso, 'MISO')
        test_clf(clf_lsvc, 'LSVC')
