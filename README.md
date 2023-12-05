## DRAUC

Official implementation of "DRAUC: An Instance-wise Distributionally Robust AUC Optimization Framework"
## MaxMatch: Semi-Supervised Learning with Worst-Case Consistency

This is a PyTorch implementation of [DRAUC: An Instance-wise Distributionally Robust AUC Optimization Framework](https://arxiv.org/pdf/2311.03055.pdf).

### Environments

* torch>=1.5.0
* torchvision>=0.6.0
* scikit-learn
* pillow


### Data Preparation
Download following datasets:
* CIFAR-10 and CIFAR-100 from https://www.cs.toronto.edu/~kriz/cifar.html
* MNIST from http://yann.lecun.com/exdb/mnist/
* Tiny-ImageNet from https://huggingface.co/datasets/zh-plus/tiny-imagenet
* CIFAR-10-C from https://zenodo.org/records/2535967
* CIFAR-100-C from https://zenodo.org/records/3555552
* MNIST-C from https://zenodo.org/records/3239543
* Tiny-ImageNet-C from https://zenodo.org/records/2469796

Preprocess dataset: run split_train_valid.py to generate train valid split for Tiny-ImageNet

  ```bash
  $ tree /path/to/your/datasets/
    ├── CIFAR10
    │   └── cifar-10-batches-py
    │       ├── batches.meta
    │       ├── data_batch_1
    │       ├── ...
    ├── CIFAR100
    │   └── cifar-100-python
    │       ├── cifar-100-python.tar.gz
    │       ├── file.txt~
    │       ├── meta
    │       ├── test
    │       └── train
    ├── CIFAR-100-C
    │   ├── brightness.npy
    │   ├── contrast.npy
    │   ├── ...
    ├── CIFAR-10-C
    │   ├── brightness.npy
    │   ├── contrast.npy
    │   ├── ...
    ├── MNIST
    │   ├── ImbalancedMNIST
    │   │   ├── processed
    │   │   └── raw
    ├── MNIST-C
    │   ├── brightness
    │   │   ├── test_images.npy
    │   │   ├── test_labels.npy
    │   │   ├── train_images.npy
    │   │   └── train_labels.npy
    │   ├── canny_edges
    │   │   ├── test_images.npy
    │   │   ├── ...
    |   ├── ...
    ├── TINYIMAGENET-H
    │   ├── test
    │   │   ├── n01443537
    │   │   ├── n01629819
    │   │   ├── ...
    │   ├── train
    │   │   ├── n01443537
    │   │   ├── n01629819
    │   │   ├── ...
    │   ├── valid
    │   │   ├── n01443537
    │   │   ├── n01629819
    │   │   ├── ...
    │   ├── wnids.txt
    │   └── words.txt
    ├── TINYIMAGENET-C
        ├── brightness
        ├── contrast
        ├── ...
    

  ```

### Training
```
python run.py
```
### Citation

If you use the code of this repository, please cite our paper:
```
@article{dai2023drauc,
      title={DRAUC: An Instance-wise Distributionally Robust AUC Optimization Framework}, 
      author={Siran Dai and Qianqian Xu and Zhiyong Yang and Xiaochun Cao and Qingming Huang},
      year={2023},
      eprint={2311.03055},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.