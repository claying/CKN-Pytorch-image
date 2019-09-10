#filters="64 256" # 77.6%
#filters="256 1024" # 82.0%
filters="512 8192" # 84.0% with kernel-sizes="3 2"
kernels="3 2"
. s ## activate virtual environment
cd experiments
python cifar10_unsup.py --filters $filters --subsamplings 2 6 --kernel-sizes $kernels
