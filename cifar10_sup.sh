. s ## activate virtual environment
cd experiments
python cifar10_sup.py --epochs 105 --lr 0.1 --alpha 0.001 --loss hinge --alternating --model ckn5
