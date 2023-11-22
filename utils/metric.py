from sklearn.metrics import roc_auc_score
import sklearn
import numpy as np
import torch

def check_tensor_shape(tensor, shape):
    # check tensor shape 
    if not torch.is_tensor(tensor):
        raise ValueError('Input is not a valid torch tensor!')
    if not isinstance(shape, (tuple, list, int)):
        raise ValueError("Shape must be a tuple, an integer or a list!")
    if isinstance(shape, int):
        shape = torch.Size([shape])
    tensor_shape = tensor.shape
    if len(tensor_shape) != len(shape):
        tensor = tensor.reshape(shape)
    return tensor

def check_array_shape(array, shape):
    # check array shape
    array = check_array_type(array)  
    if array.size == 0:
        raise ValueError("Array is empty.")
    if array.shape != shape and len(array.shape) != 1:
        try:
            array = array.reshape(shape)
        except ValueError as e:
            raise ValueError(f"Could not reshape array of shape {array.shape} to {shape}.") from e
    return array
def check_array_type(array):
    # convert to array type 
    if not isinstance(array, (np.ndarray, np.generic)):
        array = np.array(array)
    return array
def select_mean(array, threshold=0):
    # select elements for average based on threshold
    array = check_array_type(array)  
    select_array = array[array >= threshold] 
    if len(select_array) != 0: 
        return np.mean(select_array)
    else:
        return None 
def auc_roc_score(y_true, y_pred, reduction='mean', **kwargs):
    r"""Evaluation function of AUROC"""
    y_true = check_array_type(y_true)
    y_pred = check_array_type(y_pred)
    num_labels = y_true.shape[-1] if len(y_true) == 2 else 1
    y_true = check_array_shape(y_true, (-1, num_labels)) 
    y_pred = check_array_shape(y_pred, (-1, num_labels))
    assert reduction in ['mean', None, 'None'], 'Input is not valid!'
    if y_pred.shape[-1] != 1 and len(y_pred.shape) > 1:
        class_auc_list = []
        for i in range(y_pred.shape[-1]):
            try:
                local_auc = roc_auc_score(y_true[:, i], y_pred[:, i],  **kwargs)
                class_auc_list.append(local_auc)
            except: 
                # edge case: no positive samples in the data set
                class_auc_list.append(-1.0) # if only one class
        if reduction == 'mean':
            return select_mean(class_auc_list, threshold=0) # return non-negative mean
        return class_auc_list
    return roc_auc_score(y_true, y_pred, **kwargs)


def AUROC(y_true, y_pred, multi_type='ova', acc=True):
    """
        Compute Area Under the Receiver Operating Characteristic Curve (AUROC).
        Note:
            This function can be only used with binary, multiclass AUC (either 'ova' or 'ovo').

    """
    # y_true = y_true.flatten()
    if not isinstance(y_true, np.ndarray):
        warnings.warn("The type of y_ture must be np.ndarray")
        y_true = np.asarray(y_true)
    
    if not isinstance(y_pred, np.ndarray):
        warnings.warn("The type of y_pred must be np.ndarray")
        y_pred = np.asarray(y_pred)
    
    # if len(np.unique(y_true)) == 2:
    #     assert len(y_pred) == len(y_true), 'prediction and ground-truth must be the same length!'
    #     return roc_auc_score(y_true=y_true, y_score=y_pred)
    # elif len(np.unique(y_true)) > 2:
    if multi_type == 'ova':
        return sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred, multi_class='ovr')
    elif multi_type == 'ovo':
        if acc:
            return fast_multi_class_auc_score(y_true=y_true, y_pred=y_pred)
        else:
            auc = multi_class_auc_score(y_true=y_true, y_pred=y_pred)
        return auc
    else:
        raise ValueError('multiclass only supports ova and ovo regime!')
    # else:
    #     raise ValueError('AUROC must have at least two classes!')

def multi_class_auc_score(y_true, y_pred, **kwargs):
    n = y_true.max() + 1

    def bin_auc(label, pred, i, j):
        msk1 = (label == i)
        msk2 = (label == j)
        y1 = pred[msk1, i]
        y2 = pred[msk2, i]
        return np.mean([ix > jx for ix in y1 for jx in y2])

    return np.mean([
        bin_auc(y_true, y_pred, i, j) for i in range(n) for j in range(n)
        if i != j
    ])

def fast_multi_class_auc_score(y_true, y_pred, **kwargs):
    classes = np.unique(y_true)
    num_class = len(classes)
    sum_cls = np.array([(y_true == i).sum() for i in range(num_class)])

    def bin_auc(label, pred, idx, sum_cls):
        pd = pred[:, idx]
        lbl = label
        r = np.argsort(pd)
        lbl = lbl[r]
        sum_cls = sum_cls[lbl]

        loc_idx = np.where(lbl == idx)[0][::-1]
        weight = np.zeros((len(lbl)))
        for i in range(len(loc_idx) - 1):
            weight[loc_idx[i+1] + 1:loc_idx[i]] = i + 1
        weight[:loc_idx[-1]] = len(loc_idx)

        res = (weight / sum_cls).sum() / len(loc_idx) / (num_class - 1)

        return res

    return np.mean([
        bin_auc(y_true, y_pred, idx, sum_cls) for idx in range(num_class)
    ])