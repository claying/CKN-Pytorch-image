# Convolutional kernel network with Pytorch

Re-implementation of Convolutional Kernel Network (CKN) from Mairal (2016)
in Python based on the [Pytorch][1] framework. 
The package is available under the **GPL-v3** license.

Author: Dexiong Chen

Credits: Ghislain Durif, Mathilde Caron, Alberto Bietti, Julien Mairal

The code is based on

>Mairal, Julien.
[End-to-end kernel learning with supervised convolutional kernel networks][5]. NIPS 2016.

If you have any issues, please contact dexiong.chen@inria.fr.

## Installation

We strongly recommend users to use [anaconda][2] to install the following packages

```
numpy
scipy
scikit-learn
pytorch=1.2.0
miso_svm
```
The Python package `miso_svm` can be installed with (original [repository][3])
```
cd third-party/miso_svm-1.0
python setup.py install
```

## Results

Reproduction of the results from [Mairal (2016)][5] with this package.
The results from the original paper (Mairal, 2016) were achieved using
cudnn-based Matlab code available [here][4]. To run the following experiments, please first download the [data][6], put into the folder `./data/cifar-10` and then do

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
cd experiments
```

#### Unsupervised CKN

Here is a summary of the results of **unsupervised** CKN on CIFAR10 image classification dataset with pre-whitening
and without data augmentation or model ensembling.

```python
# Code examples
python cifar10_unsup.py --filters 64 256 --subsamplings 2 6 --kernel-sizes 3 3
```

| #layers   | #filters      | filter size   | subsampling | sigma        | Accuracy |
|:---------:|:-------------:|:-------------:|:-----------:|:------------:|:--------:|
| 2         | 64, 256       | 3, 3          | 2, 6        | 0.6          |  77.5    |
| 2         | 256, 1024     | 3, 3          | 2, 6        | 0.6          |  82.0    |
| 2         | 512, 8192     | 3, 2          | 2, 6        | 0.6          |  84.0    |

#### Supervised CKN

Here is a summary of the results of **supervised** CKN on CIFAR10 image classification dataset with pre-whitening
and without data augmentation or model ensembling.

```python
# Code examples
python cifar10_sup.py --epochs 105 --lr 0.1 --alpha 0.001 --loss hinge --alternating --model ckn5
python cifar10_sup.py --epochs 105 --lr 0.1 --alpha 0.1 --loss hinge --alternating --model ckn14
```

| Architecture | Accuracy | training time (GTX1080\_ti) |
|:------------:|:--------:|:--------------------------:|
| CKN-5        | 86.1     | ~60 min                    |
| CKN-14       | 90.2     | ~260 min                   |


[1]: https://pytorch.org/
[2]: https://anaconda.org/
[3]: https://gitlab.inria.fr/gdurif/ckn-tf/tree/prod/miso_svm/
[4]: https://gitlab.inria.fr/mairal/ckn-cudnn-matlab/
[5]: http://papers.nips.cc/paper/6184-bayesian-latent-structure-discovery-from-multi-neuron-recordings.pdf
[6]: http://pascal.inrialpes.fr/data2/mairal/data/cifar_white.mat
