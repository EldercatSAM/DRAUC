from models import resnet20, resnet20_f, resnet32, resnet32_f, small_cnn, small_cnn_f, \
        efficientnetb1, efficientnetb0, efficientnetb1_f, efficientnetb0_f, densenet121, densenet121_f
import argparse
from optimizers import DRAUCOptim, PESG
from losses import DRAUCLoss,AUCMLoss_V2,FocalLoss,DROLTLoss

from utils.datasets import ImbalancedCIFAR10, get_dataset_info
from utils.dataloaders import get_multi_loader
from utils.modules import CompleteLogger, AverageMeter, ProgressMeter
import os
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
import time
from utils.metric import auc_roc_score
import torch 
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score

mu = None
std = None

upper_limit, lower_limit = 1,0

def config(args):
    if args.dataset != 'MNIST':
        in_channels = 3
    else:
        in_channels = 1
    if args.model == 'resnet20':
        if args.loss != 'DROLT':
            model = resnet20(in_channels = in_channels)
        else:
            model = resnet20_f(in_channels = in_channels)
    elif args.model == 'resnet32':
        if args.loss != 'DROLT':
            model = resnet32(in_channels = in_channels)
        else:
            model = resnet32_f(in_channels = in_channels)
    elif args.model == 'small_cnn':
        if args.loss != 'DROLT':
            model = small_cnn(in_channels = in_channels)
        else:
            model = small_cnn_f(in_channels = in_channels)
    elif args.model == 'efficientnetb0':
        if args.loss != 'DROLT':
            model = efficientnetb0(in_channels = in_channels)
        else:
            model = efficientnetb1_f(in_channels = in_channels)

    elif args.model == 'efficientnetb1':
        if args.loss != 'DROLT':
            model = efficientnetb1(in_channels = in_channels)
        else:
            model = efficientnetb1_f(in_channels = in_channels)
    
    elif args.model == 'densenet121':
        if args.loss != 'DROLT':
            model = densenet121(in_channels = in_channels)
        else:
            model = densenet121_f(in_channels = in_channels)
    else:
        raise NotImplementedError
    model.to(device)
    if args.loss == 'DRAUC':
        criterion = DRAUCLoss(margin = args.margin, k = 1., _lambda = args.lambda_)

        optimizer = DRAUCOptim(
            model, 
            a = criterion.a, 
            b = criterion.b, 
            alpha = criterion.alpha,
            lr = args.lr,
            momentum=args.momentum,
            weight_decay = args.wd, 
            epoch_to_opt=args.warmup_epochs,
        )

    elif args.loss == 'CDRAUC':
        criterion = DRAUCLoss(margin = args.margin, k = 1., _lambda = args.lambda_)
        criterion._lambda1 = torch.tensor(args.lambda_, dtype=torch.float32, device=device, requires_grad=False).to(device)

        _k = (args.epsilon - args.im_ratio_train * args.k * args.epsilon) / (1-args.im_ratio_train)
        criterion.eps = torch.tensor(args.epsilon * _k, dtype=torch.float32, device=device, requires_grad=False).to(device)
        criterion.eps1 = torch.tensor(args.epsilon * args.k, dtype=torch.float32, device=device, requires_grad=False).to(device)
        optimizer = DRAUCOptim(
            model, 
            a = criterion.a, 
            b = criterion.b, 
            alpha = criterion.alpha,
            lr = args.lr,
            momentum=args.momentum,
            weight_decay = args.wd, 
            epoch_to_opt=args.warmup_epochs,
        )
        args.epsilon = torch.tensor(args.epsilon, dtype=torch.float32, device=device, requires_grad=False).to(device)
   
    elif args.loss == 'CE':
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)

    elif args.loss == 'AUCMLoss':
        criterion = AUCMLoss_V2()
        optimizer = PESG(model, 
                loss_fn=criterion,
                lr=args.lr, 
                momentum=args.momentum,
                margin=args.margin, 
                epoch_decay=args.epoch_decay, 
                weight_decay=args.wd)
    elif args.loss == 'FocalLoss':
        criterion = FocalLoss()
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)

    elif args.loss == 'WDRO':
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)

    elif args.loss == 'DROLT':
        criterion = DROLTLoss(epsilon = args.DROLT_epsilon)
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)

    else:
        print(args.loss)
        raise NotImplementedError

    return model, criterion, optimizer

def normalize(X):
    return (X - mu)/std

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def mynorm(x, order): 
    """
    Custom norm, given x is 2D tensor [b, d]. always calculate norm on the dim=1  
        L1(x) = 1/d * sum(abs(x_i))
        L2(x) = sqrt(1/d * sum(square(x)))
        Linf(x) = max(abs(x_i))
    """
    x = torch.reshape(x, [x.shape[0], -1])
    b, d = x.shape 
    if order == 1: 
        return 1./d * torch.sum(torch.abs(x), dim=1) # [b,]
    elif order == 2: 
        return torch.sqrt(1./d * torch.sum(torch.square(x), dim=1)) # [b,]
    elif order == np.inf:
        return torch.max(torch.abs(x), dim=1)[0] # [b,]
    else: 
        raise 

def attack_DRO(model, f, x, y, _lambda, attack_lr, \
                 epsilon = 0.08, iters = 10, projection = False, p = np.inf, constrained = False, _lambda1 = None, epsilon1 = None):
    '''
    model: Classifier
    f: Loss function, e.g., AUC Loss
    x: Original input image
    y: Label
    _lambda: Regularization parmeter lambda
    attack_lr: Learning rate for gradient ascent
    epsilon: maximum attack distance
    iters: Iteration nums for gradient ascent for maximizaiton
    projection: Weather to restrict delta in epsilon Lp norm ball
    p: Use Lp norm to evaluate the distance
    '''
    if isinstance(f, DRAUCLoss):
        f.stop_grad()

    model.eval()
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(x).cuda()
    
    delta = torch.zeros_like(x).cuda() #init delta
    delta.normal_()

    if constrained:
        y = y.squeeze(dim = -1)
        assert _lambda1 is not None and epsilon1 is not None
        # print(delta[y==1].shape, delta[y==0].shape)
        if p==2: # restrict adv samples in epsilon p-norm ball
            d_flat = delta[y==0].view(delta[y==0].size(0),-1)
            d_flat1 = delta[y==1].view(delta[y==1].size(0),-1)
            n = d_flat.norm(p=p,dim=1).view(delta[y==0].size(0),1,1,1)
            n1 = d_flat1.norm(p=p,dim=1).view(delta[y==1].size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            r1 = torch.zeros_like(n1).uniform_(0, 1)
            delta[y==1] *= r1/n1*epsilon1
            delta[y==0] *= r/n*epsilon
        elif p == np.inf:
            delta[y==0].uniform_(-epsilon, epsilon)
            delta[y==1].uniform_(-epsilon1, epsilon1)
        else:
            raise ValueError

        _lambda = y * _lambda1 + (1-y) * _lambda
        y = y.unsqueeze(dim = -1)
    else:
        if p==2: # restrict adv samples in epsilon p-norm ball
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=p,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        elif p == np.inf:
            delta.uniform_(-epsilon, epsilon)
        else:
            raise ValueError

    delta = clamp(delta, -x, 1-x)
    delta.requires_grad = True

    for _ in range(iters): # generate DRO adv samples
        loss = f(torch.sigmoid(model(normalize(x+delta))), y) - (_lambda * torch.pow(torch.norm(delta, p=p),2)).mean()
        loss.backward()

        grad = delta.grad.detach()

        d = delta
        if p == 2:
            g_norm = torch.norm(grad.view(grad.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = grad/(g_norm + 1e-10)
            
            if projection:
                d = (d + scaled_g * attack_lr).view(delta.size(0),-1).renorm(p=p,dim=0,maxnorm=epsilon).view_as(delta)
            else:
                d = d + scaled_g * attack_lr
        elif p==np.inf:
            d = d + attack_lr * torch.sign(grad)

        d = clamp(d, -x, 1-x)
        delta.data = d
        delta.grad.zero_()

        all_loss = F.binary_cross_entropy(torch.sigmoid(model(normalize(x+delta))), y, reduction='none').squeeze()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    model.train()
    if isinstance(f, DRAUCLoss):
        f.start_grad()

    return max_delta

def get_fname(args):
    names = args.model + '_' + args.lr_schedule +  '_bs' + str(args.batch_size) + '_lr' + str(args.lr) 
    names = names + args.loss
    names = names + f"_imratio{args.im_ratio_train}"
    if args.loss == 'DRAUC' or args.loss == 'DROCE' or args.loss == 'CDRAUC':
        names = names + f'_norm_{args.norm}_eps_{args.epsilon}_lambdalr_{args.lambda_lr}_lambda_{args.lambda_}'
        if args.loss == 'CDRAUC':
            names = names + f'_k_{args.k}'
    if args.loss == 'AUCMLoss':
        names = names + f'_margin_{args.margin}'
    if args.loss == 'WDRO':
        names = names + f'_lambdaGrad_{args.lambda_grad}'
    if args.loss == 'DROLT':
        names = names + f'_lambdaDROLT_{args.DROLT_lambda}_epsDROLT_{args.DROLT_epsilon}'
    print('File name: ', names)
    return names

def main(args):
    global device, mu, std
    args.lr *= args.batch_size / 128 
    print(f"lr: {args.lr}, lambda: {args.lambda_}")

    mu, std, class_num = get_dataset_info(args)
    args.class_num = class_num
    device = torch.device("cuda:{gpu}".format(gpu=args.gpu) if torch.cuda.is_available() else "cpu")
    
    fname = get_fname(args)
    args.attack_lr /= 255.
    args.epsilon /= 255.
    if not isinstance(args.seed, list):
        args.seed = [args.seed]
    
    total_auc = []
    for random_seed in args.seed:
        
        base_path = os.path.join(
                            args.save_dir, 
                            args.dataset, 
                            args.loss,
                            fname,
                            'seed' + str(random_seed))

        if not os.path.exists(base_path):
            os.makedirs(base_path)
        else:
            if os.path.exists(os.path.join(base_path, "test_auc_record.txt")):
                print("Existed experiment!")
                return

        logger = CompleteLogger(os.path.join(base_path, 'saved_model'))
        logger.info(args)
        logger.info("======> Training with random seed: {}".format(random_seed))
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

        cudnn.benchmark = True

        train_loader, valid_loader, test_loaders = get_multi_loader(args, random_seed)

        model, criterion, optimizer = config(args)

        if args.lr_schedule == "step":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.decay_epochs,gamma=0.1,last_epoch=-1)
        elif args.lr_schedule == 'cos':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        elif args.lr_schedule == "None":
            lr_scheduler = None
        else:
            raise NotImplementedError
        best_auc = 0
        best_epoch = -1

        train_aucs = []
        valid_aucs = []
        test_aucs = []
        disp_all, disn_all = [],[]
        for epoch in range(args.epochs):
            train_auc = train(args, train_loader, model, criterion, optimizer, epoch)

            valid_auc = validate(args, valid_loader, model, criterion)

            train_aucs.append(train_auc)
            valid_aucs.append(valid_auc)
            # remember best acc@1 and save checkpoint

            cur_checkpoint = {
                'model': model.state_dict(),
                'schedular': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'epoch': epoch
            }
            torch.save(cur_checkpoint, logger.get_checkpoint_path('lr_{}_latest_seed_{}'.format(args.lr, random_seed)))

            if valid_auc > best_auc:
                shutil.copy(logger.get_checkpoint_path('lr_{}_latest_seed_{}'.format(args.lr, random_seed)), 
                            logger.get_checkpoint_path('lr_{}_best_seed_{}'.format(args.lr, random_seed)))
                best_epoch = epoch
            
            best_auc = max(best_auc, valid_auc)

            if lr_scheduler is not None:
                lr_scheduler.step()
            logger.info("Epoch: {}/{}, AUC: {}, cur_best_auc: {} at Epoch: {}".format(epoch, 
                                                                                        args.epochs, 
                                                                                        valid_auc,
                                                                                        best_auc,
                                                                                        best_epoch))
        np.savetxt(base_path + '/train_auc_record.txt', np.array(train_aucs))
        np.savetxt(base_path + '/valid_auc_record.txt', np.array(valid_aucs))
        state_dict = torch.load(logger.get_checkpoint_path('lr_{}_best_seed_{}'.format(args.lr, random_seed)))['model']
        model.load_state_dict(state_dict, strict=False)

        test_aucs = test(args, test_loaders, model)
        with open(base_path + f'/test_auc_record.txt', "w") as f:
            f.write(repr(test_aucs))
        
        logger.info("======> Random seed: {}".format(random_seed))
        logger.info("best_val_auc = {:.4f}".format(best_auc))
        logger.info("======>")

        total_auc.append(best_auc)
    
    logger.info("======> Average performance over {} experiments: {}".format(len(args.seed), np.mean(total_auc)))
    logger.info("======> Std performance over {} experiments: {}".format(len(args.seed), np.std(total_auc)))
    
    return best_auc

def train(args, train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    
    progress = ProgressMeter(
            args.epochs,
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    
    model.train()
    if epoch in args.decay_epochs and 'AUC' in args.loss:
        optimizer.update_regularizer(decay_factor=10) # decrease learning rate by 10x & update regularizer

    pred_list, target_list = [], []

    dis_cum = []
    dis_cum1 = []

    disps, disns = [],[]
    end = time.time()

    if args.loss == 'DROLT' and epoch > args.DROLT_warmup_epochs:
        #Get train_centroid
        model.eval()
        if args.model == 'resnet20' or args.model == 'resnet32':
            features = torch.empty((0, 64)).to(device) # 0, feature_dim
        elif args.model == 'efficientnetb0':
            features = torch.empty((0, 1280)).to(device)
        elif args.model == 'densenet121':
            features = torch.empty((0, 1024)).to(device)
        else:
            raise NotImplementedError
        labels = torch.empty(0, dtype=torch.float).to(device)
        with torch.no_grad():
            for i, (images, targets) in enumerate(train_loader):
                images, targets = images.float().to(device), targets.float().to(device)
                feat, _ = model(normalize(images))
                features = torch.cat((features, feat))
                labels = torch.cat((labels, targets))

        model.train()

    lambda_grad = torch.tensor(args.lambda_grad, requires_grad = True)
    for i, (images, targets) in enumerate(train_loader):
        # print(i)
        # print(criterion._lambda)
        images, targets = images.float().to(device), targets.float().to(device)
        if args.dataset == 'MELANOMA' or args.dataset == 'TINYIMAGENET-H':
            targets = targets.unsqueeze(dim = -1)
        if args.loss == 'WDRO':
            images.requires_grad = True
        if args.loss == 'DRAUC' or args.loss == 'DROCE':
            delta = attack_DRO(model, criterion, images, targets, criterion._lambda, args.attack_lr, 
                 args.epsilon , args.attack_iters, args.projection, args.norm)
            images = torch.clamp(images + delta[:images.size(0)], 0, 1)
        if args.loss == 'CDRAUC':
            delta = attack_DRO(model, criterion, images, targets, _lambda = criterion._lambda, attack_lr = args.attack_lr, 
                  epsilon = criterion.eps , iters = args.attack_iters, projection = args.projection, p = args.norm,
                  constrained = True, _lambda1 = criterion._lambda1, epsilon1 = criterion.eps1)
            images = torch.clamp(images + delta[:images.size(0)], 0, 1)
        data_time.update(time.time() - end) # time cost of loading data

        if args.loss == 'DROLT':
            feats, preds = model(normalize(images))
            preds = torch.sigmoid(preds)
            if epoch > args.DROLT_warmup_epochs:
                loss = args.DROLT_lambda * criterion(feats, targets, features, labels) + (1 - args.DROLT_lambda) * F.binary_cross_entropy(preds, targets)
            else:
                loss = F.binary_cross_entropy(preds, targets)
        else:
            preds = torch.sigmoid(model(normalize(images)))
            loss = criterion(preds, targets)

        pred_list.append(preds.detach().cpu().numpy())
        target_list.append(targets.detach().cpu().numpy())
        
        if args.loss == 'WDRO':
            grad_input = torch.autograd.grad(loss, images, retain_graph = True)[0]
            loss = loss + lambda_grad * torch.sum(torch.square(grad_input))
    
        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.loss == 'DRAUC' or args.loss == 'DROCE':
            dis = torch.mean(mynorm(delta.detach(), order=1), dim=0)

            dis_cum.append(dis)
            if i % args.lambda_period == 0: 
                criterion._lambda = criterion._lambda - args.lambda_lr * (args.epsilon - sum(dis_cum)/len(dis_cum))
                dis_cum = []

            targets = targets.squeeze(dim = -1)

        if args.loss == 'CDRAUC':
            targets = targets.squeeze(dim = -1)
            dis = torch.mean(mynorm(delta[targets==0].detach(), order=1), dim=0)
            dis1 = torch.mean(mynorm(delta[targets==1].detach(), order=1), dim=0)
            dis_cum.append(dis)
            dis_cum1.append(dis1)
            if i % args.lambda_period == 0: 
                criterion._lambda = criterion._lambda - args.lambda_lr * (criterion.eps - sum(dis_cum)/len(dis_cum))
                criterion._lambda1 = criterion._lambda1 - args.lambda_lr * (criterion.eps1 - sum(dis_cum1)/len(dis_cum1))
                dis_cum = []
                dis_cum1 = []

            targets = targets.squeeze(dim = -1)

        batch_time.update(time.time() - end)
        end = time.time()


        
    return auc_roc_score(np.concatenate(target_list), np.concatenate(pred_list))
   
def validate(args, valid_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    model.eval()
    aucs = []
        
    pred_list, target_list = [], []
    end = time.time()
    for i, (images, targets) in enumerate(valid_loader):
        
        images, targets = images.to(device), targets.to(device)
        if args.loss == 'DROLT':
            feats, preds = model(normalize(images))
            preds = torch.sigmoid(preds)
        else:
            preds = torch.sigmoid(model(normalize(images)))

        pred_list.append(preds.detach().cpu().numpy())
        target_list.append(targets.detach().cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

    return auc_roc_score(np.concatenate(target_list), np.concatenate(pred_list))


def test(args, test_loaders, model):
    batch_time = AverageMeter('Time', ':6.3f')
    model.eval()
    aucs = test_loaders
    with torch.no_grad():
        for _test in test_loaders.keys():
            test_loader = test_loaders[_test]
            pred_list, target_list = [], []
            end = time.time()
            for i, (images, targets) in enumerate(test_loader):
                images, targets = images.to(device), targets.to(device)
                if args.loss == 'DROLT':
                    feats, preds = model(normalize(images))
                    preds = torch.sigmoid(preds)
                else:
                    preds = torch.sigmoid(model(normalize(images)))

                pred_list.append(preds.detach().cpu().numpy())
                target_list.append(targets.detach().cpu().numpy())

                batch_time.update(time.time() - end)
                end = time.time()
            aucs[_test] = auc_roc_score(np.concatenate(target_list), np.concatenate(pred_list))

    return aucs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRAUC')

    parser.add_argument('--data_dir', metavar='DIR', default='/path/to/your/data/',
                        help='root path of dataset')    
    parser.add_argument('--save_dir', default='/path/to/your/logdir', help='where to save model')
    parser.add_argument('-d', '--dataset', metavar='DATA', default='MELANOMA', choices = ["CIFAR10","CIFAR100","MNIST","TINYIMAGENET", "TINYIMAGENET-H", "MELANOMA"])

    parser.add_argument('--gpu', default=0, type=int, help='gpu numbers')
    parser.add_argument('--num-workers', default=4, type=int, help='worker numbers')

    parser.add_argument('--loss', type=str, default="DRAUC", choices = ["DRAUC", "CE", "AUCMLoss", "FocalLoss", "WDRO", "DROLT", "DROCE", "CDRAUC"])
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--model', default='efficientnetb0', choices=['resnet20', 'resnet32', 'resnet56','small_cnn','efficientnetb1', 'efficientnetb0', 'densenet121'],
                        help='backbone of model')

    parser.add_argument('--pretrained', default="", type=str,
                        help='whether using pretrained model')

    parser.add_argument('--lr-schedule', default="step", type=str, choices=["step", "cos", "None"],
                        help='scheduler')  
    parser.add_argument('--decay-epochs', default=[50, 75], type=list,
                        help='num of epochs to decay lr')  

    parser.add_argument('--im-ratio-train', default=0.1, type=float, help='proportion of positive images in trainset')
    parser.add_argument('--im-ratio-test', default=[0.5], 
                type=list, help='proportion of positive images in testset')
    parser.add_argument('--train-valid-ratio', default=0.8, type=float, help='train / train + valid in trainset')
    #AUC settings
    parser.add_argument('--margin', default=1.0, type=float)
    parser.add_argument('--k', default=2.0, type=float)

    #DRO settings
    parser.add_argument('--attack-lr', default=15., type=float)
    parser.add_argument('--epsilon', default=128., type=float)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--norm', default=2, type=int)
    parser.add_argument('--projection', default=False, type=bool)
    parser.add_argument('--rho', default=128./255, type=float)

    parser.add_argument('--lambda_', default=1.0, type=float)
    # parser.add_argument('--lambda-factor', default=0.04, type=float)
    parser.add_argument('--fix-lambda', default=True, type=bool)

    # parser.add_argument('--lamda_init', default=1.0, type=float, help='initial value for lambda')
    parser.add_argument('--lambda-lr', default=0.02, type=float, help='learning rate to update lambda')
    parser.add_argument('--lambda-period', default=10, type=int, help='period for updating lambda')

    #WDRO settings
    parser.add_argument('--lambda-grad', default=0.004, type=float, help='lambda_grad in WDRO')

    #DROLT settings
    parser.add_argument('--DROLT-lambda', default=0.5, type=float, help='lambda in DROLT')
    parser.add_argument('--DROLT-epsilon', default=1, type=float, help='lambda in DROLT')
    parser.add_argument('--DROLT-warmup-epochs', default=-1, type=int, metavar='N',
                        help='number of total epochs to run')
    #AUCMLoss settings
    parser.add_argument('--epoch-decay', default=0.003, type=float)
    
    parser.add_argument('--warmup-epochs', default=0, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    main(args)

