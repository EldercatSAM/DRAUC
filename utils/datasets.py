import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
import os
from PIL import Image

from tqdm import tqdm
# from sampler import *


def check_imbalance_binary(Y):
    # numpy array
    num_samples = len(Y)
    pos_count = np.count_nonzero(Y == 1)
    neg_count = np.count_nonzero(Y == 0)
    pos_ratio = pos_count/ (pos_count + neg_count)


def ImbalanceGenerator(X, Y, imratio=0.5, shuffle=True, is_balanced=False, random_seed=123):
    '''
    Imbalanced Data Generator
    Reference: https://arxiv.org/abs/2012.03173
    '''
    
    assert isinstance(X, (np.ndarray, np.generic)), 'data needs to be numpy type!'
    assert isinstance(Y, (np.ndarray, np.generic)), 'data needs to be numpy type!'

    num_classes = np.unique(Y).size
    split_index = num_classes // 2 - 1
   
    id_list = list(range(X.shape[0]))
    np.random.seed(random_seed)
    np.random.shuffle(id_list)

    # print(id_list[:10])
    X = X[id_list]
    Y = Y[id_list]
    X_copy = X.copy()
    Y_copy = Y.copy()
    Y_copy[Y_copy<=split_index] = 0 # [0, ....]
    Y_copy[Y_copy>=split_index+1] = 1 # [0, ....]
    
    if is_balanced == False:
        num_neg = np.where(Y_copy==0)[0].shape[0]
        num_pos = np.where(Y_copy==1)[0].shape[0]
        keep_num_pos = int((imratio/(1-imratio))*num_neg )
        neg_id_list = np.where(Y_copy==0)[0] 
        pos_id_list = np.where(Y_copy==1)[0][:keep_num_pos] 
        X_copy = X_copy[neg_id_list.tolist() + pos_id_list.tolist() ] 
        Y_copy = Y_copy[neg_id_list.tolist() + pos_id_list.tolist() ] 
        #Y_copy[Y_copy==0] = 0
    
    if shuffle:
        # do shuffle in case batch prediction error
        id_list = list(range(X_copy.shape[0]))
        np.random.seed(random_seed)
        np.random.shuffle(id_list)
        X_copy = X_copy[id_list]
        Y_copy = Y_copy[id_list]
    
    num_samples = len(X_copy)
    pos_count = np.count_nonzero(Y_copy == 1)
    neg_count = np.count_nonzero(Y_copy == 0)
    pos_ratio = pos_count/ (pos_count + neg_count)
    print ('NUM_SAMPLES: [%d], POS:NEG: [%d : %d], POS_RATIO: %.4f'%(num_samples, pos_count, neg_count, pos_ratio) )

    return X_copy, Y_copy.reshape(-1, 1).astype(float)


class ImbalancedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, im_ratio=0.01, seed=0, split='train',
                 transform=None, target_transform=None,
                 download=False, train_imratio = 0.8, return_index = False):
        """
        Parmeters:
        root: CIFAR10 root dir
        imratio: pos_img_num / total_img_num in train set (0.5 in valid set)
        seed: the random seed
        split: train, valid or test
        tranform: transform on image
        target_tranform: transform on target
        download: whether download the dataset
        train_imratio: train_img_num / (train_img_num + valid_img_num)
        """

        if split != 'test':
            train = True
        else:
            train = False

        super(ImbalancedCIFAR10, self).__init__(root, train, transform, target_transform, download)
        #print(self.data.shape)
        # print(self.data.dtype)
        # raise ValueError
        self.return_index = return_index
        self.targets = np.array(self.targets)
        if split == 'train':
            img_num = self.data.shape[0]
            id_list = list(range(img_num))
            np.random.seed(seed)
            np.random.shuffle(id_list)
            train_split = id_list[:int(img_num*train_imratio)]
            self.data, self.targets = self.data[train_split], self.targets[train_split]
            self.data, self.targets = ImbalanceGenerator(self.data, self.targets, im_ratio, random_seed = seed)

        elif split == 'valid':
            img_num = self.data.shape[0]
            id_list = list(range(img_num))
            np.random.seed(seed)
            np.random.shuffle(id_list)
            val_split = id_list[int(img_num*train_imratio):]
            self.data, self.targets = self.data[val_split], self.targets[val_split]
            self.data, self.targets = ImbalanceGenerator(self.data, self.targets, im_ratio, random_seed = seed)
        else:
            self.data, self.targets = ImbalanceGenerator(self.data, self.targets, im_ratio, random_seed = seed)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return img, target, index
        else:
            return img, target

class ImbalancedMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, im_ratio=0.01, seed=0, split='train',
                 transform=None, target_transform=None,
                 download=False, train_imratio = 0.8, return_index = False):
        """
        Parmeters:
        root: MNIST root dir
        imratio: pos_img_num / total_img_num in train set (0.5 in valid set)
        seed: the random seed
        split: train, valid or test
        tranform: transform on image
        target_tranform: transform on target
        download: whether download the dataset
        train_imratio: train_img_num / (train_img_num + valid_img_num)
        """

        if split != 'test':
            train = True
        else:
            train = False
        # print(root)
        super(ImbalancedMNIST, self).__init__(root, train, transform, target_transform, download)
        self.return_index = return_index
        self.data = self.data.numpy()
        self.targets = np.array(self.targets)
        if split == 'train':
            img_num = self.data.shape[0]
            id_list = list(range(img_num))
            np.random.seed(seed)
            np.random.shuffle(id_list)
            train_split = id_list[:int(img_num*train_imratio)]
            self.data, self.targets = self.data[train_split], self.targets[train_split]
            self.data, self.targets = ImbalanceGenerator(self.data, self.targets, im_ratio, random_seed = seed)

        elif split == 'valid':
            img_num = self.data.shape[0]
            id_list = list(range(img_num))
            np.random.seed(seed)
            np.random.shuffle(id_list)
            val_split = id_list[int(img_num*train_imratio):]
            self.data, self.targets = self.data[val_split], self.targets[val_split]
            self.data, self.targets = ImbalanceGenerator(self.data, self.targets, im_ratio, random_seed = seed)
        else:
            self.data, self.targets = ImbalanceGenerator(self.data, self.targets, im_ratio, random_seed = seed)

        self.data = torch.from_numpy(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return img, target, index
        else:
            return img, target
            
class ImbalancedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, im_ratio=0.01, seed=0, split='train',
                 transform=None, target_transform=None,
                 download=False, train_imratio = 0.8, return_index = False):
        """
        Parmeters:
        root: CIFAR100 root dir
        imratio: pos_img_num / total_img_num in train set (0.5 in valid set)
        seed: the random seed
        split: train, valid or test
        tranform: transform on image
        target_tranform: transform on target
        download: whether download the dataset
        train_imratio: train_img_num / (train_img_num + valid_img_num)
        """

        if split != 'test':
            train = True
        else:
            train = False

        super(ImbalancedCIFAR100, self).__init__(root, train, transform, target_transform, download)
        self.targets = np.array(self.targets)
        self.return_index = return_index
        
        if split == 'train':
            img_num = self.data.shape[0]
            id_list = list(range(img_num))
            np.random.seed(seed)
            np.random.shuffle(id_list)
            train_split = id_list[:int(img_num*train_imratio)]
            self.data, self.targets = self.data[train_split], self.targets[train_split]
            self.data, self.targets = ImbalanceGenerator(self.data, self.targets, im_ratio, random_seed = seed)

        elif split == 'valid':
            img_num = self.data.shape[0]
            id_list = list(range(img_num))
            np.random.seed(seed)
            np.random.shuffle(id_list)
            val_split = id_list[int(img_num*train_imratio):]
            self.data, self.targets = self.data[val_split], self.targets[val_split]
            self.data, self.targets = ImbalanceGenerator(self.data, self.targets, im_ratio, random_seed = seed)
        else:
            self.data, self.targets = ImbalanceGenerator(self.data, self.targets, im_ratio, random_seed = seed)
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.return_index:
            return img, target, index
        else:
            return img, target

class CIFAR10_1():
    def __init__(self, root = "" , transform = None, return_index = False, binary = True):
        images = np.load(os.path.join(root, 'data.npy'), allow_pickle=True)
        labels = np.load(os.path.join(root, 'label.npy'), allow_pickle=True)
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        self.return_index = return_index

        if binary:
            num_classes = np.unique(labels).size
            split_index = num_classes // 2 - 1
            
            labels[labels<=split_index] = 0 # [0, ....]
            labels[labels>=split_index+1] = 1 # [0, ....]
        self.images, self.labels = images, labels
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        if self.return_index:
            return image, label, index
        else:
            return image, label
    def __len__(self):
        return len(self.labels)

class CIFAR_C():
    def __init__(self, root = "" , transform = None, return_index = False, binary = True):
        """
        Binary CIFAR-10/100-C
        """
        self.return_index = return_index
        all_files = os.listdir(root)
        images, labels = [], []
        for f in all_files:
            if f != 'labels.npy':
                images.append(np.load(os.path.join(root, f), allow_pickle=True))
                labels.append(np.load(os.path.join(root, 'labels.npy'), allow_pickle=True))
        images = np.concatenate(np.array(images))
        labels = np.concatenate(np.array(labels))
        
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        
        if binary:
            num_classes = np.unique(labels).size
            split_index = num_classes // 2 - 1
            
            labels[labels<=split_index] = 0 # [0, ....]
            labels[labels>=split_index+1] = 1 # [0, ....]
        self.images, self.labels = images, labels
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        if self.return_index:
            return image, label, index
        else:
            return image, label
    def __len__(self):
        return len(self.labels)

class MNIST_C():
    def __init__(self, root = "" , transform = None, return_index = False):
        """
        Binary CIFAR-10/100-C
        """
        self.return_index = return_index
        all_corrupts = os.listdir(root)
        images, labels, corruptions = [], [], []
        for f in all_corrupts:
            corruptions.append(f)
            images.append(np.load(os.path.join(root, f, 'test_images.npy'), allow_pickle=True))
            labels.append(np.load(os.path.join(root, f, 'test_labels.npy'), allow_pickle=True))
        images = np.concatenate(np.array(images))
        labels = np.concatenate(np.array(labels))
        
        self.corruptions = corruptions
        assert labels.min() >= 0
        assert images.dtype == np.uint8
        
        num_classes = np.unique(labels).size
        split_index = num_classes // 2 - 1
        
        labels[labels<=split_index] = 0 # [0, ....]
        labels[labels>=split_index+1] = 1 # [0, ....]
        self.images, self.labels = images, labels
        self.transform = transform
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        if self.return_index:
            return image, label, index
        else:
            return image, label
    def __len__(self):
        return len(self.labels)

def pil_loader(filename, label=False):
    ext = (os.path.splitext(filename)[-1]).lower()
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        img = Image.open(filename)
        if not label:
            img = img.convert('RGB')
            img = np.array(img).astype(dtype=np.uint8)
            img = img[:,:,::-1]  #convert to BGR
        else:
            if img.mode != 'L' and img.mode != 'P':
                img = img.convert('L')
            img = np.array(img).astype(dtype=np.uint8)
    elif ext == '.mat':
        img = scio.loadmat(filename)
    elif ext == '.npy':
        img = np.load(filename, allow_pickle=True)
    else:
        raise NotImplementedError('Unsupported file type %s'%ext)

    return img
    
class ImbalancedTinyImagenet():
    def __init__(self, root, im_ratio=0.01, seed=0, split='train',
                 transform=None, target_transform=None, return_index = False, use_hyper_class = False):
        assert split in ['train', 'valid', 'test']
        if use_hyper_class:
            assert im_ratio in [0.02, 0.03, 0.06]
        self.return_index = return_index
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.generate_classid()
        self.read_dataset()

        if not use_hyper_class:
            self.images, self.labels = ImbalanceGenerator(self.images, self.labels, im_ratio, random_seed = seed)
        else:
            if im_ratio == 0.02:
                self.pos_class = (67, 115, 41, 35) # bird
            elif im_ratio == 0.03:
                self.pos_class = (182, 135, 78, 39, 11, 194) # dog
            elif im_ratio == 0.06: 
                self.pos_class = (147, 69, 157, 163, 108, 114, 64, 90, 15, 117, 75, 152) # vehicle
            self.labels = np.where(np.isin(self.labels, self.pos_class), 1, 0)
            

    def generate_classid(self):
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as f:
            class_ids = f.readlines()

        # print(class_ids)
        tmp = {}
        for i, ids in enumerate(class_ids):
            tmp[ids.strip()] = i

        self.class_ids = tmp
    
    def read_dataset(self):
        self.images, self.labels = [], []
        image_folders = os.listdir(os.path.join(self.root, self.split))
        
        for folder in image_folders:
            if 'txt' in folder:
                continue
            imgs = os.listdir(os.path.join(self.root, self.split, folder, 'images'))
            for img in imgs:
                img = Image.open(os.path.join(self.root, self.split, folder, 'images', img)).convert('RGB')
                img = np.array(img).astype(dtype=np.uint8)
                self.images.append(img)
                self.labels.append(self.class_ids[folder])
            
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        image = self.transform(image)
        if self.return_index:
            return image, label, index
        else:
            return image, label
    
    def __len__(self):
        return len(self.images)

class TinyImagenet_C():
    def __init__(self, root, transform=None, target_transform=None, return_index = False, use_hyper_class = False, im_ratio = 0.01):
        if use_hyper_class:
            assert im_ratio in [0.02, 0.03, 0.06]

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.generate_classid()
        self.return_index = return_index
        self.read_dataset()

        self.labels = np.array(self.labels)
        
        if not use_hyper_class:
            num_classes = np.unique(self.labels).size
            split_index = num_classes // 2 - 1
            self.labels[self.labels<=split_index] = 0 # [0, ....]
            self.labels[self.labels>=split_index+1] = 1 # [0, ....]
        else:
            if im_ratio == 0.02:
                self.pos_class = (67, 115, 41, 35) # bird
            elif im_ratio == 0.03:
                self.pos_class = (182, 135, 78, 39, 11, 194) # dog
            elif im_ratio == 0.06: 
                self.pos_class = (147, 69, 157, 163, 108, 114, 64, 90, 15, 117, 75, 152) # vehicle
            else:
                raise ValueError
            self.labels = np.where(np.isin(self.labels, self.pos_class), 1, 0)

    def generate_classid(self):
        with open(os.path.join(self.root, 'wnids.txt'), 'r') as f:
            class_ids = f.readlines()

        tmp = {}
        for i, ids in enumerate(class_ids):
            tmp[ids.strip()] = i

        self.class_ids = tmp
    
    def read_dataset(self):
        self.images, self.labels = [], []
        
        for aug in os.listdir(self.root):
            if '.txt' in aug:
                continue
            for level in ['1', '2', '3', '4', '5']:
                image_folders = os.listdir(os.path.join(self.root, aug, level))
        
                for folder in image_folders:
                    if folder not in self.class_ids:
                        continue
                    imgs = os.listdir(os.path.join(self.root, aug, level, folder))
                    # print(imgs)
                    for img in imgs:
                        # print(img)
                        path = os.path.join(self.root, aug, level, folder, img)
                        self.images.append(path)
                        self.labels.append(self.class_ids[folder])
            
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        image = self.transform(img)
        if self.return_index:
            return image, label, index
        else:
            return image, label
    
    def __len__(self):
        return len(self.images)

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, imb_type='exp', imb_factor=1, seed=0, split='train',
                 transform=None, target_transform=None, train_imratio = 0.8, 
                 download=False):
        
        if split != 'test':
            train = True
        else:
            train = False
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)

        self.cls_num = 10
        self.targets = np.array(self.targets)
        if split == 'train':
            train_split = []
            for i in range(self.cls_num):
                id_list = np.where(self.targets == i)[0].tolist()
                img_num = len(id_list)
                np.random.seed(seed)
                np.random.shuffle(id_list)
                id_list = id_list[:int(img_num*train_imratio)]

                train_split += id_list
            self.data, self.targets = self.data[train_split], self.targets[train_split]

        elif split == 'valid':
            val_split = []
            for i in range(self.cls_num):
                id_list = np.where(self.targets == i)[0].tolist()
                img_num = len(id_list)
                np.random.seed(seed)
                np.random.shuffle(id_list)
                id_list = id_list[int(img_num*train_imratio):]

                val_split += id_list
            self.data, self.targets = self.data[val_split], self.targets[val_split]
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.phat = 0.1
        self.gen_imbalanced_data(self.img_num_list)
        self.targets = np.array(self.targets)
        
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            assert the_img_num <= len(idx), print(the_img_num, len(idx))
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_two_class(self, Y):
        Y = np.array(Y)
        loc_0 = np.where(Y <= (self.cls_num/2-1))[0]
        loc_1 = np.where(Y > (self.cls_num/2-1))[0]
        Y[loc_1] = 1
        Y[loc_0] = 0
        self.phat = len(np.where(Y == 1)[0])/len(Y)
        return Y.tolist()

def get_dataset_info(args):
    cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
    cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

    cifar100_mean = (0.5071, 0.4865, 0.4409)
    cifar100_std = (0.2673, 0.2564, 0.2762)

    svhn_mean = (0.4377, 0.4438, 0.4728)
    svhn_std = (0.1980, 0.2010, 0.1970)

    tinyimagenet_mean = (0.4802, 0.4481, 0.3975)
    tinyimagenet_std = (0.2764, 0.2689, 0.2816)

    mnist_mean = (0.1307)
    mnist_std = (0.3081)
    if args.dataset == "CIFAR10":
        mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar10_std).view(3,1,1).cuda()
        classnum = 10
    elif args.dataset == "CIFAR100":
        mu = torch.tensor(cifar100_mean).view(3,1,1).cuda()
        std = torch.tensor(cifar100_std).view(3,1,1).cuda()
        classnum = 100
    elif args.dataset == "MNIST":
        mu = torch.tensor(mnist_mean).view(1,1,1).cuda()
        std = torch.tensor(mnist_std).view(1,1,1).cuda()
        classnum = 10
    elif args.dataset == "SVHN":
        mu = torch.tensor(svhn_mean).view(3,1,1).cuda()
        std = torch.tensor(svhn_std).view(3,1,1).cuda()
        classnum = 10
    elif args.dataset in ['TINYIMAGENET','TINYIMAGENET-H']:
        mu = torch.tensor(tinyimagenet_mean).view(3,1,1).cuda()
        std = torch.tensor(tinyimagenet_std).view(3,1,1).cuda()
        classnum = 200
    return mu, std, classnum
