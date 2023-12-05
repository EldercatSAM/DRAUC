import numpy as np
import subprocess
import time
num_gpus = 8
dataset = 'CIFAR10'

# DRAUC
lr = [0.01,0.02,0.05,0.1,0.2]
eps = [8,32,64,128]
lambda_lr = [0.01,0.02,0.1,0.2]
model = ['resnet20', 'resnet32'] # for MNIST, model=['resnet20', 'small_cnn']
scheduler = ['cos', 'step']
im_ratio = [0.01,0.05,0.1,0.2] # for Tiny-ImageNet-H, im_ratio = [0.02,0.03,0.06]

tests = []
for a in lr:
    for b in eps:
        for c in lambda_lr:
            for e in model:
                for f in scheduler:
                    for g in im_ratio:
                        tests.append([a,b,c,e,f,g])

print(len(tests))

for i in range(0,len(tests),num_gpus):
    processes = []
    for j in range(num_gpus):
        _lr, _eps,_llr, _m, _s, _i = tests[i+j]
        process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={j} python train.py --dataset {dataset} --model {_m} --lr-schedule {_s} --loss DRAUC --lr {_lr} --epsilon {_eps} --lambda-lr {_llr} --im-ratio-train {_i}', shell=True)
        processes.append(process)
        
    for process in processes:
        process.wait()

# CDRAUC
lr = [0.01,0.02,0.05,0.1,0.2]
eps = [8,32,64,128]
lambda_lr = [0.01,0.02,0.1,0.2]
k = [0.5,0.8,1.0,1.2,1.5]
model = ['resnet20', 'resnet32'] # for MNIST, model=['resnet20', 'small_cnn']
scheduler = ['cos', 'step']
im_ratio = [0.01,0.05,0.1,0.2] # for Tiny-ImageNet-H, im_ratio = [0.02,0.03,0.06]

tests = []
for a in lr:
    for b in eps:
        for c in lambda_lr:
            for d in k:
                for e in model:
                    for f in scheduler:
                        for g in im_ratio:
                            tests.append([a,b,c,d,e,f,g])

print(len(tests))

for i in range(0,len(tests),num_gpus):
    processes = []
    for j in range(num_gpus):
        _lr, _eps,_llr, _k, _m, _s, _i = tests[i+j]
        process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={j} python train.py --dataset {dataset} --model {_m} --lr-schedule {_s} --loss CDRAUC --k {_k} --lr {_lr} --epsilon {_eps} --lambda-lr {_llr} --im-ratio-train {_i}', shell=True)
        processes.append(process)
        
    for process in processes:
        process.wait()