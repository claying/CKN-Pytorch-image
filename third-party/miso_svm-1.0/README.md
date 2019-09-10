# MISO SVM

This package implements a SVM (support vector machine) classification procedure
based on the MISO optimization algorithm, which is introduced in [1] (available
at <https://arxiv.org/abs/1506.02186v2> or
<http://papers.nips.cc/paper/5928-a-universal-catalyst-for-first-order-optimization>).

The 'miso_svm' package is based on C++ interfaced codes. All files included in
the 'miso_svm' package ([miso_svm/*] and in particular [miso_svm/miso_svm/*])
are released under the GPL-v3 license.

---

# Installation

This package requires the MKL from Intel (for Blas and OpenMP). You can get
the MKL by using the Python Anaconda distribution, or you can use your own
MKL license if you have one.


## Prerequisite when using Anaconda

You can get anaconda or miniconda from <https://conda.io/docs/user-guide/install/index.html>
or <https://conda.io/miniconda.html>.

Create a conda virtual environment and install dependencies within it:
```bash
conda create -n cknenv  # if not done yet
source activate cknenv
conda install numpy scipy scikit-learn matplotlib
```

## Install miso_svm

* On GNU/Linux and MacOS:

If using previously created conda environment:
```bash
source activate cknenv
```

then
```bash
git clone https://gitlab.inria.fr/thoth/ckn
cd ckn/miso_svm
python setup.py install
```

OR
```bash
wget http://pascal.inrialpes.fr/data2/gdurif/miso_svm-1.0.tar.gz
tar zxvf miso_svm-1.0.tar.gz
cd miso_svm-1.0
python setup.py install
```

To specify an installation directory:
```bash
inst=<your_install_directory>
PYV=$(python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)";)
export PYTHONPATH=$inst/lib/python${PYV}/site-packages:$PYTHONPATH
python setup.py install --prefix=$inst
```


## When using the official GitLab repository (for developpers)

(on GNU/Linux and MacOs only)

* To build/install/test the package, see:

```bash
./dev_command.sh help
```

## Example of use

See [classification.py](miso_svm/classification.py),
[quick.py](miso_svm/quick.py) or [miso.py](miso_svm/miso.py)


---

## References

[1] Lin, H., Mairal, J., Harchaoui, Z., 2015. A universal catalyst for first-order optimization, in: Advances in Neural Information Processing Systems. pp. 3384–3392.
