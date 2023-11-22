import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from abc import abstractmethod

def squared_loss(margin, t):
    return (margin - t)** 2

def squared_hinge_loss(margin, t):
    return torch.max(margin - t, torch.zeros_like(t)) ** 2

def logistic_loss(margin, t):
    return torch.log(1+torch.log(-margin*t))

def _check_tensor_shape(inputs, shape=(-1, 1)):
    input_shape = inputs.shape
    target_shape = shape
    if len(input_shape) != len(target_shape):
        inputs = inputs.reshape(target_shape)
    return inputs

def _get_surrogate_loss(backend='squared_hinge'):
    if backend == 'squared_hinge':
       surr_loss = squared_hinge_loss
    elif backend == 'squared':
       surr_loss = squared_loss
    elif backend == 'logistic':
       surr_loss = logistic_loss
    else:
        raise ValueError('Out of options!')
    return surr_loss

class DRAUCLoss(torch.nn.Module): # AUC margin loss with extra k to gurantee convexity of alpha
    def __init__(self, margin=1.0, k = 1, _lambda = 1., device=None):
        super(DRAUCLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.k = k
        assert self.k > 0
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device) 
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self._lambda = torch.tensor(_lambda, dtype=torch.float32, device=self.device, requires_grad=False).to(self.device)

    def mean(self, tensor):
        return torch.sum(tensor)/(torch.count_nonzero(tensor) + 1e-6)

    def stop_grad(self):
        self.a.requires_grad = False
        self.b.requires_grad = False
        self.alpha.requires_grad = False
    
    def start_grad(self):
        self.a.requires_grad = True
        self.b.requires_grad = True
        self.alpha.requires_grad = True

    def forward(self, y_pred, y_true):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        pos_mask = (1==y_true).float()
        neg_mask = (0==y_true).float()
        loss = self.mean((y_pred - self.a)**2*pos_mask) + \
               self.mean((y_pred - self.b)**2*neg_mask) + \
               2*self.alpha*(self.margin + self.mean(y_pred*neg_mask) - self.mean(y_pred*pos_mask)) - \
               self.k * self.alpha**2
        if torch.isnan(loss):
            raise ValueError            
        return loss   

class AUCMLoss_V2(torch.nn.Module):
    """AUC-Margin Loss: a novel loss function to optimize AUROC
    
    Args:
        margin: margin for AUCM loss, e.g., m in [0, 1]
        
    Return:
        loss value (scalar) 
        
    Reference: 
            Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification,
            Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao,
            Proceedings of the IEEE/CVF International Conference on Computer Vision 2021.

    """
    def __init__(self, margin=1.0, device=None):
        super(AUCMLoss_V2, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device) 
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device) 

    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true):
        # print(y_pred, y_true)
        
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        pos_mask = (1==y_true).float()
        neg_mask = (0==y_true).float()
        loss = self.mean((y_pred - self.a)**2*pos_mask) + \
               self.mean((y_pred - self.b)**2*neg_mask) + \
               2*self.alpha*(self.margin + self.mean(y_pred*neg_mask) - self.mean(y_pred*pos_mask)) - \
               self.alpha**2  

        return loss   


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.binary_cross_entropy(input, target,reduction=self.reduction) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class DROLTLoss(nn.Module):
    def __init__(self, temperature=5., base_temperature=1., class_weights=None, epsilon=None):
        super(DROLTLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.class_weights = class_weights
        self.epsilons = torch.tensor(epsilon, requires_grad = False)

    def pairwise_euaclidean_distance(self, x, y):
        return torch.cdist(x, y)

    def pairwise_cosine_sim(self, x, y):
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        y = y / (y.norm(dim=1, keepdim=True) + 1e-8)
        return torch.matmul(x, y.T)

    def forward(self, batch_feats, batch_targets, centroid_feats, centroid_targets):
        device = (torch.device('cuda')
                  if centroid_feats.is_cuda
                  else torch.device('cpu'))
        classes, positive_counts = torch.unique(batch_targets, return_counts=True)
        centroid_classes = torch.unique(centroid_targets)
        train_prototypes = torch.stack([centroid_feats[torch.where(centroid_targets == c)[0]].mean(0)
                                        for c in centroid_classes])

        pairwise = -1 * self.pairwise_euaclidean_distance(train_prototypes, batch_feats)
        if self.epsilons is not None:
            mask = torch.eq(centroid_classes.contiguous().view(-1, 1), batch_targets.contiguous().view(-1, 1).T).to(
                device)
            a = pairwise.clone()
            pairwise[mask] = a[mask] - self.epsilons.to(device)

        logits = torch.div(pairwise, self.temperature)
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
        log_prob = torch.stack([log_prob[:, torch.where(batch_targets == c)[0]].mean(1) for c in classes], dim=1)
        mask = torch.eq(centroid_classes.contiguous().view(-1, 1), classes.contiguous().view(-1, 1).T).float().to(
            device)
        log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob_pos
        if self.class_weights is not None:
            weights = self.class_weights[centroid_classes]
            weighted_loss = loss * weights
            loss = weighted_loss.sum() / weights.sum()
        else:
            loss = loss.sum() / len(classes)

        return loss

# copy from libauc 
def get_surrogate_loss(loss_name='squared_hinge'):
    r"""
        A wrapper to call a specific surrogate loss function.
    
        Args:
            loss_name (str): type of surrogate loss function to fetch, including 'squared_hinge', 'squared', 'logistic', 'barrier_hinge' (default: ``'squared_hinge'``).
    """
    # assert f'{loss_name}_loss' in __all__, f'{loss_name} is not implemented'
    if loss_name == 'squared_hinge':
       surr_loss = squared_hinge_loss
    elif loss_name == 'squared':
       surr_loss = squared_loss
    elif loss_name == 'logistic':
       surr_loss = logistic_loss
    elif loss_name == 'barrier_hinge':
       surr_loss = barrier_hinge_loss
    else:
        raise ValueError('Out of options!')
    return surr_loss


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

class pAUC_DRO_Loss(torch.nn.Module):
    r"""
        Partial AUC loss based on KL-DRO to optimize One-way Partial AUROC (OPAUC). In contrast to conventional AUC, partial AUC pays more attention to partial difficult samples. By leveraging the Distributionally Robust Optimization (DRO), the objective is defined as

            .. math::
               \min_{\mathbf{w}}\frac{1}{n_+}\sum_{\mathbf{x}_i\in\mathbf{S}_+} \max_{\mathbf{p}\in\Delta} \sum_j p_j L(\mathbf{w}; \mathbf{x}_i, \mathbf{x}_j) - \lambda \text{KL}(\mathbf{p}, 1/n)

        Then the objective is reformulated as follows to develop an algorithm.

            .. math::
               \min_{\mathbf{w}}\frac{1}{n_+}\sum_{\mathbf{x}_i \in \mathbf{S}_+}\lambda \log \frac{1}{n_-}\sum_{\mathbf{x}_j \in \mathbf{S}_-}\exp\left(\frac{L(\mathbf{w}; \mathbf{x}_i, \mathbf{x}_j)}{\lambda}\right)

        where :math:`L(\mathbf{w}; \mathbf{x_i}, \mathbf{x_j})` is the surrogate pairwise loss function for one positive data and one negative data, e.g., squared hinge loss, :math:`\mathbf{S}_+` and :math:`\mathbf{S}_-` denote the subsets of the dataset which contain only positive samples and negative samples, respectively.

        The optimization algorithm for solving the above objective is implemented as :obj:`~libauc.optimizers.SOAPs`. For the derivation of the above formulation, please refer to the original paper [4]_.


        Args:
            data_len (int):  total number of samples in the training dataset.
            gamma (float): parameter for moving average estimator (default: ``0.9``).
            surr_loss (string, optional): surrogate loss used in the problem formulation (default: ``'squared_hinge'``).
            margin (float, optional): margin for squared-hinge surrogate loss (default: ``1.0``).
            Lambda (float, optional): weight for KL divergence regularization, e.g., 0.1, 1.0, 10.0 (default: ``1.0``).

        Example:
            >>> loss_fn = libauc.losses.pAUC_DRO_Loss(data_len=data_length, gamma=0.9, Lambda=1.0)
            >>> preds  = torch.randn(32, 1, requires_grad=True)
            >>> target = torch.empty(32, dtype=torch.long).random_(1)
            >>> index = torch.randint(32, (32,), requires_grad=False)
            >>> loss = loss_fn(preds, target, index)
            >>> loss.backward()

        .. note::
           To use :class:`~libauc.losses.pAUC_DRO_Loss`, we need to track index for each sample in the training dataset. To do so, see the example below:

           .. code-block:: python

               class SampleDataset (torch.utils.data.Dataset):
                    def __init__(self, inputs, targets):
                        self.inputs = inputs
                        self.targets = targets
                    def __len__ (self) :
                        return len(self.inputs)
                    def __getitem__ (self, index):
                        data = self.inputs[index]
                        target = self.targets[index]
                        return data, target, index

        .. note::
            Practical tips: 
            
            - ``gamma`` is a parameter which is better to be tuned in the range (0, 1) for better performance. Some suggested values are ``{0.1, 0.3, 0.5, 0.7, 0.9}``.
            - ``margin`` can be tuned in ``{0.1, 0.3, 0.5, 0.7, 0.9, 1.0}`` for better performance.
            - ``Lambda`` can be tuned in the range (0.1, 10) for better performance. 

    """                        
    def __init__(self, 
                 data_len, 
                 gamma=0.9,
                 margin=1.0,
                 Lambda=1.0, 
                 surr_loss='squared_hinge', 
                 device=None):
        super(pAUC_DRO_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device     
        self.data_len = data_len      
        self.u_pos = torch.tensor([0.0]*data_len).view(-1, 1).to(self.device)
        self.margin = margin
        self.gamma = gamma 
        self.Lambda = Lambda                           
        self.surrogate_loss = get_surrogate_loss(surr_loss)
    
    def forward(self, y_pred, y_true, index, **kwargs):
        y_pred = check_tensor_shape(y_pred, (-1, 1))
        y_true = check_tensor_shape(y_true, (-1, 1))
        index  = check_tensor_shape(index, (-1,))  
        pos_mask = (y_true == 1).squeeze() 
        neg_mask = (y_true == 0).squeeze() 
        assert sum(pos_mask) > 0, "Input data has no positive sample! Please use 'libauc.sampler.DualSampler' for data resampling!"
        if len(index) == len(y_pred): 
            index = index[pos_mask]   # indices for positive samples only       
        f_ps = y_pred[pos_mask]            # shape: (len(f_ps), 1) 
        f_ns = y_pred[neg_mask].squeeze()  # shape: (len(f_ns), ) 
        surr_loss = self.surrogate_loss(self.margin, (f_ps - f_ns))  # shape: (len(f_ps), len(f_ns))                       
        exp_loss = torch.exp(surr_loss/self.Lambda)
        self.u_pos[index] = (1 - self.gamma) * self.u_pos[index] + self.gamma * (exp_loss.mean(1, keepdim=True).detach())
        p = exp_loss/self.u_pos[index]    # shape: (len(f_ps), len(f_ns))                       
        p.detach_()
        loss = torch.mean(p * surr_loss)
        return loss

class AUCLoss(nn.Module):
    
    ''' 
        args:
            num_classes: number of classes (mush include params)

            gamma: safe margin in pairwise loss (default=1.0) 

            transform: manner to compute the multi-classes AUROC Metric, either 'ovo' or 'ova' (default as 'ovo' in our paper)
    
    '''
    def __init__(self,
                 num_classes,
                 gamma=1,
                 transform='ovo', *kwargs):
        super(AUCLoss, self).__init__()

        if transform != 'ovo' and transform != 'ova':
            raise Exception("type should be either ova or ovo")
        self.num_classes = num_classes
        self.gamma = gamma
        self.transform = transform

        if kwargs is not None:
            self.__dict__.update(kwargs)
    
    def _check_input(self, pred, target):
        assert pred.max() <= 1 and pred.min() >= 0
        assert target.min() >= 0
        assert pred.shape[0] == target.shape[0]

    def forward(self, pred, target, **kwargs):
        '''
        args:
            pred: score of samples residing in [0,1]. 
            For examples, with respect to binary classification tasks, pred = torch.Sigmoid(...)
            o.w. pred = torch.Softmax(...) 

            target: index of classes. In particular, w.r.t. binary classification tasks, we regard y=1 as pos. instances.

        '''
        self._check_input(pred, target)

        if self.num_classes == 2:
            Y = target.float().squeeze()
            numPos = torch.sum(Y.eq(1))
            numNeg = torch.sum(Y.eq(0))
            Di = 1.0 / numPos / numNeg
            return self.calLossPerCLass(pred.squeeze(1), Y, Di, numPos)
        else:
            if self.transform == 'ovo':
                factor = self.num_classes * (self.num_classes - 1)
            else:
                factor = 1

            Y = torch.stack(
                [target.eq(i).float() for i in range(self.num_classes)],
                1).squeeze()

            N = Y.sum(0)  
            D = 1 / N[target.squeeze().long()]  

            loss = torch.Tensor([0.]).cuda()
            if self.transform == 'ova':
                ones_vec = torch.ones_like(D).cuda()
            
            for i in range(self.num_classes):
                if self.transform == 'ovo':
                    Di = D / N[i]
                else:
                    fac = torch.tensor([1.0]).cuda() / (N[i] * (N.sum() - N[i]))
                    Di = fac * ones_vec
                Yi, predi = Y[:, i], pred[:, i]
                # print(Yi.shape)
                loss += self.calLossPerCLass(predi, Yi, Di, N[i])

            return loss / factor

    def calLossPerCLass(self, predi, Yi, Di, Ni):
        
        return self.calLossPerCLassNaive(predi, Yi, Di, Ni)

    @abstractmethod
    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        pass


class SquareAUCLoss(AUCLoss):
    def __init__(self, num_classes, gamma=1, transform='ovo', **kwargs):
        super(SquareAUCLoss, self).__init__(num_classes, gamma, transform)

        # self.num_classes = num_classes
        # self.gamma = gamma

        if kwargs is not None:
            self.__dict__.update(kwargs)

    def calLossPerCLassNaive(self, predi, Yi, Di, Ni):
        diff = predi - self.gamma * Yi

        # print(predi.shape, Yi.shape)
        nD = Di.mul(1 - Yi)
        fac = (self.num_classes -
               1) if self.transform == 'ovo' else torch.tensor(1.0).cuda()
        S = Ni * nD + (fac * Yi / Ni)
        # print(diff.shape)
        diff = diff.reshape((-1, ))
        # print(diff.shape)
        S = S.reshape((-1, ))
        # print(S.shape)
        A = diff.mul(S).dot(diff)
        nD= nD.reshape((-1, ))
        Yi= Yi.reshape((-1, ))
        B = diff.dot(nD) * Yi.dot(diff)
        return 0.5 * A - B