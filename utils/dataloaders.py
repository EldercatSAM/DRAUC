from .datasets import *
from .sampler import StratifiedSampler, make_balanced_sampler
import os
from torchvision import transforms
import torch
import torchvision

def get_multi_loader(args, seed):
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    if args.dataset == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set = ImbalancedCIFAR10(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'train',
                        transform=train_transform, target_transform=None,
                        download=True, train_imratio = args.train_valid_ratio)

        valid_set = ImbalancedCIFAR10(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'valid',
                        transform=test_transform, target_transform=None,
                        download=True, train_imratio = args.train_valid_ratio)# split = valid , use test transform
        
        
        sampler = StratifiedSampler(
            train_set.targets,
            args.batch_size,
            args.im_ratio_train,
            1-args.im_ratio_train
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            sampler = sampler,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=_init_fn
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        test_loaders = {
            'CIFAR10': None,
            'CIFAR10.1': None,
            'CIFAR10-C': None,
        }
        test_set = ImbalancedCIFAR10(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_test[0], seed=args.seed, split = 'test',
                        transform=test_transform, target_transform=None,
                        download=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['CIFAR10'] = test_loader

        test_set = CIFAR10_1(os.path.join(args.data_dir, 'CIFAR10.1_V6'), transform=test_transform)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['CIFAR10.1'] = test_loader

        test_set = CIFAR_C(root = os.path.join(args.data_dir, 'CIFAR-10-C'), transform=test_transform)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['CIFAR10-C'] = test_loader
    elif args.dataset == 'CIFAR100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set = ImbalancedCIFAR100(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'train',
                        transform=train_transform, target_transform=None,
                        download=True, train_imratio = args.train_valid_ratio)

        valid_set = ImbalancedCIFAR100(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'valid',
                        transform=test_transform, target_transform=None,
                        download=True, train_imratio = args.train_valid_ratio)# split = valid , use test transform
        sampler = StratifiedSampler(
            train_set.targets,
            args.batch_size,
            args.im_ratio_train,
            1-args.im_ratio_train
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            sampler = sampler,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=_init_fn
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        test_loaders = {
            'CIFAR100': None,
            'CIFAR100-C': None,
        }
        test_set = ImbalancedCIFAR100(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_test[0], seed=args.seed, split = 'test',
                        transform=test_transform, target_transform=None,
                        download=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['CIFAR100'] = test_loader

        test_set = CIFAR_C(root = os.path.join(args.data_dir, 'CIFAR-100-C'), transform=test_transform)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['CIFAR100-C'] = test_loader

    elif args.dataset == 'MNIST':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set = ImbalancedMNIST(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'train',
                        transform=train_transform, target_transform=None,
                        download=True, train_imratio = args.train_valid_ratio)

        valid_set = ImbalancedMNIST(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'valid',
                        transform=test_transform, target_transform=None,
                        download=True, train_imratio = args.train_valid_ratio)# split = valid , use test transform
        sampler = StratifiedSampler(
            train_set.targets,
            args.batch_size,
            args.im_ratio_train,
            1-args.im_ratio_train
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            sampler = sampler,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=_init_fn
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        test_loaders = {
            'MNIST': None,
            'MNIST-C': None,
        }
        test_set = ImbalancedMNIST(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_test[0], seed=args.seed, split = 'test',
                        transform=test_transform, target_transform=None,
                        download=True)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['MNIST'] = test_loader

        test_set = MNIST_C(root = os.path.join(args.data_dir, 'MNIST-C'), transform=test_transform)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['MNIST-C'] = test_loader
    
    elif args.dataset == 'TINYIMAGENET':
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set = ImbalancedTinyImagenet(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'train',
                        transform=train_transform, target_transform=None)

        valid_set = ImbalancedTinyImagenet(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'valid',
                        transform=test_transform, target_transform=None)# split = valid , use test transform
        sampler = StratifiedSampler(
            train_set.labels,
            args.batch_size,
            args.im_ratio_train,
            1-args.im_ratio_train
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            sampler = sampler,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=_init_fn
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        test_loaders = {
            'TINYIMAGENET': None,
            'TINYIMAGENET-C': None,
        }
        test_set = ImbalancedTinyImagenet(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_test[0], seed=args.seed, split = 'test',
                        transform=test_transform, target_transform=None)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['TINYIMAGENET'] = test_loader

        test_set = TinyImagenet_C(root = os.path.join(args.data_dir, 'TINYIMAGENET-C'), transform=test_transform)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['TINYIMAGENET-C'] = test_loader

    elif args.dataset == 'TINYIMAGENET-H':
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_set = ImbalancedTinyImagenet(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'train',
                        transform=train_transform, target_transform=None, use_hyper_class = True)

        valid_set = ImbalancedTinyImagenet(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'valid',
                        transform=test_transform, target_transform=None, use_hyper_class = True)# split = valid , use test transform
        sampler = StratifiedSampler(
            train_set.labels,
            args.batch_size,
            args.im_ratio_train,
            1-args.im_ratio_train
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            sampler = sampler,
            pin_memory=True,
            num_workers=args.num_workers,
            worker_init_fn=_init_fn
        )

        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )

        test_loaders = {
            'TINYIMAGENET': None,
            'TINYIMAGENET-C': None,
        }
        test_set = ImbalancedTinyImagenet(os.path.join(args.data_dir, args.dataset), 
                        im_ratio=args.im_ratio_train, seed=args.seed, split = 'test',
                        transform=test_transform, target_transform=None, use_hyper_class = True)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['TINYIMAGENET'] = test_loader

        test_set = TinyImagenet_C(root = os.path.join(args.data_dir, "TINYIMAGENET-C"), transform=test_transform, use_hyper_class = True, im_ratio = args.im_ratio_train)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        test_loaders['TINYIMAGENET-C'] = test_loader
   
    else:
        raise NotImplementedError

    return train_loader, valid_loader, test_loaders