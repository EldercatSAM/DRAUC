import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np
import _pickle as pk
from collections import Counter
from sklearn.utils import shuffle
import math
import pandas as pd

import pdb 
import torchvision

# import torch
from torch.utils.data import WeightedRandomSampler

def make_balanced_sampler(labels, samples_per_class):
    # Make sure we have enough samples for all classes
    assert all(n > 0 for n in samples_per_class), "Not all classes have samples."

    # Generate a list of indices (repeated according to each class's required samples)
    indices = []
    for class_id, n_samples in enumerate(samples_per_class):
        class_indices = np.where(labels == class_id)[0]
        # Make sure there are samples for the current class
        if len(class_indices) > 0:
            indices += list(np.random.choice(class_indices, size=n_samples, replace=len(class_indices) < n_samples))
        else:
            print(f'Warning: Class {class_id} has no samples. Consider checking your dataset.')
        
    # Use these indices with the WeightedRandomSampler
    sampler = WeightedRandomSampler(indices, len(indices))

    return sampler

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, 
                class_vector, 
                batch_size, 
                rpos=1, 
                rneg=4
                ):
        self.class_vector = class_vector
        # self.n_splits = int(class_vector.size(0) / batch_size)
        self.batch_size = batch_size
        
        assert rpos !=0 and rneg !=0
        tmp = min(rpos, rneg)
        self.rpos = rpos /tmp
        self.rneg = rneg /tmp
        print(self.rpos, self.rneg)
        if isinstance(class_vector, torch.Tensor):
            y = class_vector.cpu().numpy()
        elif isinstance(class_vector, list):
            y = np.array(class_vector)
        else:
            y = class_vector
        y = y.squeeze().astype('long')
        y_counter = Counter(y)
        self.data = pd.DataFrame({'y': y})
        if len(y_counter.keys()) == 2:
            ratio = (rneg, rpos)
            
            self.class_batch_size = {
                k: round(batch_size * ratio[k] / sum(ratio))
                for k in y_counter.keys()
            }

            # print(self.class_batch_size)
        
            if rpos / rneg > y_counter[1] / y_counter[0]:
                add_pos = round(rpos / rneg * y_counter[0]) - y_counter[1]

                print("-" * 50)
                print("To balance ratio, add %d pos imgs (with replace = True)" % add_pos)
                print("-" * 50)

                pos_samples = self.data[self.data.y == 1].sample(add_pos, replace=True)
                assert pos_samples.shape[0] == add_pos

                self.data = self.data.append(pos_samples, ignore_index=False)

            else:
                add_neg = round(rneg / rpos * y_counter[1]) - y_counter[0]

                print("-" * 50)
                print("To balance ratio, add %d neg imgs repeatly" % add_neg)
                print("-" * 50)
                neg_samples = self.data[self.data.y == 0].sample(add_neg, replace=True)
                assert neg_samples.shape[0] == add_neg
                self.data = self.data.append(neg_samples, ignore_index=False)

        print("-" * 50)
        print("after complementary the ratio, having %d images" % self.data.shape[0])
        print(self.class_batch_size)
        print("-" * 50)

        self.real_batch_size = int(sum(self.class_batch_size.values()))

    def gen_sample_array(self):
        # sampling for each class
        def sample_class(group):
            n = self.class_batch_size[group.name]
            return group.sample(n) if group.shape[0] >= n else group.sample(n, replace=True)

        data = self.data.copy()
        data['idx'] = data.index
        data = data.reset_index()

        result = []
        while True:
            try:
                batch = data.groupby('y', group_keys=False).apply(sample_class)
                assert len(batch) == self.real_batch_size#, f'not enough instances {self.real_batch_size}!'
            except (ValueError, AssertionError) as e:
                print(e)
                break
            result.extend(shuffle(batch.idx))
            # pdb.set_trace()

            data.drop(index=batch.index, inplace=True)
        return result

    def __iter__(self):
        self.index_list = self.gen_sample_array()
        return iter(self.index_list)

    def __len__(self):
        try:
            l = len(self.index_list)
        except:
            l = len(self.class_vector)
        return l

class ControlledDataSampler(Sampler):
    r""" Base class for Controlled Data Sampler."""
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 labels=None, 
                 shuffle=True, 
                 num_pos=None, 
                 num_sampled_tasks=None, 
                 sampling_rate=0.5,
                 random_seed=2023): 

        assert batch_size is not None, 'batch_size can not be None!'
        assert (num_pos is None) or (sampling_rate is None), 'only one of {pos_num} and {sampling_rate} is needed!'
        
        if sampling_rate:
           assert sampling_rate>0.0 and sampling_rate<1.0, 'sampling rate is not a valid number!'
        if labels is None:
           labels = self._get_labels(dataset)
        self.labels = self._check_labels(labels) # return: (N, ) or (N, T)
        
        self.random_seed = random_seed
        self.shuffle = shuffle
       
        self.num_samples = int(len(labels))
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
       
        np.random.seed(self.random_seed)
        
        total_tasks = 0
        if num_sampled_tasks is None:
           total_tasks = self._get_num_tasks(self.labels) 
        self.total_tasks = total_tasks
        self.num_sampled_tasks = num_sampled_tasks
        self.pos_indices, self.neg_indices = self._get_sample_class_indices(self.labels) # task_id: 0, 1, 2, 3, ...
        self.class_counts = self._get_sample_class_counts(self.labels)                   # pos_len & neg_len 
 
        if self.sampling_rate:
           self.num_pos = int(self.sampling_rate*batch_size) 
           if self.num_pos == 0:
              self.num_pos = 1
           self.num_neg = batch_size - self.num_pos
        elif num_pos:
            self.num_pos = num_pos
            self.num_neg = batch_size - num_pos
        else:
            NotImplementedError

        self.num_batches = len(labels)//batch_size 
        self.sampled = []
        
    def _check_array(self, data, squeeze=True):
        if not isinstance(data, (np.ndarray, np.generic)):
           data = np.array(data)
        if squeeze:
           data = np.squeeze(data)
        return data

    def _get_labels(self, dataset):
        r"""Extract labels from given any dataset object."""
        if isinstance(dataset, torch.utils.data.Dataset):
           return np.array(dataset.targets)       
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
           return np.array(dataset.targets)    
        else:
           return np.array(dataset.labels) # TODO: support more Dataset types
    
    def _check_labels(self, labels): 
        r"""Validate labels on three cases: nan, negative, one-hot."""
        if np.isnan(labels).sum()>0:
           raise ValueError('labels contain NaN value!') 
        labels = self._check_array(labels, squeeze=True)
        if (labels<0).sum() > 0 :
           raise ValueError('labels contain negative value!') 
        if len(labels.shape) == 1:
           num_classes = np.unique(labels).size
           assert num_classes > 1, 'labels must have >= 2 classes!'
           if num_classes > 2: # format multi-class to multi-label
              num_samples = len(labels)
              new_labels = np.eye(num_classes)[labels]  
              return new_labels
        return labels

    def _get_num_tasks(self, labels):
        r"""Compute number of unique labels for binary and multi-label datasets."""
        if len(labels.shape) == 1:
            return len(np.unique(labels)) 
        else: 
            return labels.shape[-1] 
            
    def _get_unique_labels(self, labels):
        r"""Extract unique labels for binary and multi-label (task) datasets."""
        unique_labels = np.unique(labels) if len(labels.shape)==1 else np.arange(labels.squeeze().shape[-1])
        assert len(unique_labels) > 1, 'labels must have >=2 classes!'
        return unique_labels

    def _get_sample_class_counts(self, labels):
       r"""Compute number of postives and negatives per label (task). """
       num_sampled_task = self._get_num_tasks(labels)
       dict = {}
       if num_sampled_task == 2: 
           task_id = 0   # binary data, i.e. num_sampled_task == 1
           dict[task_id] = (np.count_nonzero(labels == 1), np.count_nonzero(labels == 0) )
       else:
           task_ids = np.arange(num_sampled_task)              
           for task_id in task_ids:
               dict[task_id] = (np.count_nonzero(labels[:, task_id] > 0), np.count_nonzero(labels[:, task_id] == 0) )
       return dict

    def _get_sample_class_indices(self, labels, num_sampled_task=None):
        r"""Extract sample indices for postives and negatives per label (task)."""
        if not num_sampled_task:
           num_sampled_task = self._get_num_tasks(labels)
        num_sampled_task = num_sampled_task - 1 if num_sampled_task == 2 else num_sampled_task    
        pos_indices, neg_indices = {}, {}
        for task_id in range(num_sampled_task):
             label_t = labels[:, task_id] if num_sampled_task > 2 else labels
             pos_idx = np.flatnonzero(label_t>0)
             neg_idx = np.flatnonzero(label_t==0)
             if self.shuffle:
                np.random.shuffle(pos_idx), np.random.shuffle(neg_idx)
             pos_indices[task_id] = pos_idx
             neg_indices[task_id] = neg_idx
        return pos_indices, neg_indices
    
    def __iter__(self):
        r"""Naive implementation for Controlled Data Sampler."""
        pos_id = 0
        neg_id = 0
        if self.shuffle:
           np.random.shuffle(self.pos_pool)
           np.random.shuffle(self.neg_pool)
        for i in range(self.num_batches):
            for j in range(self.num_pos):
                self.sampled.append(self.pos_indices[pos_id % self.pos_len])
                pos_id += 1
            for j in range(self.num_neg):
                self.sampled.append(self.neg_indices[neg_id % self.neg_len])
                neg_id += 1    
        return iter(self.sampled)

    def __len__ (self):
        return len(self.sampled)