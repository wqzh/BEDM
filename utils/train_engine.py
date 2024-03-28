
import collections
import numpy as np
from numpy import concatenate as concat
import pickle
from scipy.spatial.distance import cdist

import os

import torch
import torch.nn.functional as F


import utils.losses as util_losses
from utils import herding
from utils.orth import  dbt_orth, ker_orth
from utils.confusemat import ConfusionMatrix

def train_epoch(task_id, epoch, args, model, dataset, 
                optimizer, scheduler, trainloader, testloader, losses):
    
    metrics = collections.defaultdict(float)
    lambda_o = args.lambda_o
    old_classes = None if task_id==0 else model._network.classifier.alpha.shape[0]
    Nt_1, Nt = dataset.increments[task_id], dataset.increments[task_id+1]
    
    lambda_t = np.sqrt(Nt/(Nt - Nt_1))
    # print("Nt-1: {}, Nt : {}, lambda_t: {}".format(Nt_1, Nt, lambda_t))
    # e.g. Nt-1: 60, Nt : 70, lambda_t: 2.6457513110645907
    
    lr = scheduler.get_last_lr()[0]
    correct, nsamples = 0, 0
    model.network.train()
    for iters, (data, targets) in enumerate(trainloader, 1):
        data, targets = data.cuda(), targets.long().cuda()
        outputs = model.network(data)
        optimizer.zero_grad()
        
        
        ################ Compute Loss  BEGIN ################
        _, logits, atts = outputs["raw_features"], outputs["logits"], outputs["attention"]
        
        with torch.no_grad():
            correct += targets.eq(logits.argmax(1)).sum()
            nsamples += logits.shape[0]
        
        
        ###  [1] classification loss
        loss_cls = F.cross_entropy(logits, targets)
        metrics["cls"] += loss_cls.item()
            
        ### [2] orth loss
        diff = dbt_orth(model, args.backbone)
        # diff = ker_orth(model, args.backbone)
        loss_orth = lambda_o * diff
        metrics["orth"] += loss_orth.item()    
        
        if len(losses) == 2:  
            # first session, use 2 losses
            loss = loss_cls + loss_orth
        else: 
            # not first session, use 3 losses
            ### [3] distill loss
            _outputs = model._network(data)
            _logits, _atts = _outputs["logits"], _outputs["attention"]
            
            
            t_logprob = F.log_softmax(logits[:, :old_classes], dim=1)  # teacher
            s_prob = F.softmax(_logits, dim=1)    # student
            diff_logit = F.kl_div(t_logprob, s_prob, reduction='batchmean')
            
            diff_feat = util_losses.diff_feature(atts, _atts)
            metrics["diff_logit"] += diff_logit.item()
            metrics["diff_feat"] += diff_feat.item()
            diff_hybrid = diff_logit if diff_logit < args.tau else diff_feat
            loss_distill =  lambda_t * diff_hybrid
            metrics["distil"] += loss_distill.item()
            
            loss = loss_cls + loss_orth + loss_distill
        
        ################ Compute Loss  END ################
        
        loss.backward()
        optimizer.step()
        # if iters == 2 : break  # quick test
    scheduler.step()
    
    TE = f'T{task_id+1}/{dataset.n_task}, E{epoch}/{args.epochs} => '
    loss_info = ', '.join("{}: {}".format(l, round(a/iters, 3)) for l,a in metrics.items())
    print(f'{TE}{loss_info}, {correct}/{nsamples}={correct/nsamples}, lr: {lr:.6f}', flush=True)
    

@torch.no_grad()
def eval_task0(model, dataloader, verbose=False):
    """ eval using logits
    """
    samples, correct = 0, 0
    for iters, (data, targets) in enumerate(dataloader, 1):
        samples += targets.shape[0]
        data, targets = data.cuda(), targets.cuda()
        outputs = model.network(data)
        logits = outputs['logits']
        correct += targets.eq(logits.argmax(1)).sum()
        print(targets.eq(logits.argmax(1)).sum().item(), end=',', flush=True)
    print()
    print(f'eval_task : {correct}/{samples} = {correct/samples}')
    return correct/samples


@torch.no_grad()
def eval_task(model, dataloader, verbose=False):
    """ eval using logits
    """
    
    num_classes = model.network.classifier.alpha.shape[0]
    print('---> current num_classes', num_classes)
    conf = ConfusionMatrix(num_classes, verbose=verbose)
    
    samples, correct = 0, 0
    for iters, (data, targets) in enumerate(dataloader, 1):
        samples += targets.shape[0]
        data, targets = data.cuda(), targets.cuda()
        outputs = model.network(data)
        logits = outputs['logits']
        predicts = logits.argmax(1)
        correct += targets.eq(predicts).sum()
        print(targets.eq(predicts).sum().item(), end=',', flush=True)
        
        conf.update(predicts, targets)
    print()
    print(f'acc: {correct}/{samples} = {correct/samples}')
    
    conf.summary()
    return correct/samples


@torch.no_grad()
def eval_task_cdist(model, dataset, dataloader):
    """ eval using `cdist` to measure extracted feature and  exemplars-feature-mean .
    """
    EPSILON = 1e-8
    
    features, targets = model.extract_features(dataloader)
    features /= (np.linalg.norm(features, axis=1, keepdims=True) + EPSILON)
    # features (5000, 64)
    class_means = dataset.exemplars_mean[: model.network.classifier.n_classes].copy()
    # class_means (50, 64)
    sqdist = cdist(class_means, features, 'sqeuclidean').T # values >= 0
    #print(sqdist.min(), sqdist.max()) #0.001517148907679225 0.7936683736061481
    
    pseudo_logit = -sqdist  # bigger distance(<0) means more similar
    pseudo_label = np.argmax(pseudo_logit, 1)
    print(collections.Counter(pseudo_label))
    
    #print('pseudo_label', pseudo_label.shape, pseudo_label.max(), pseudo_label.dtype)
    #print('targets', targets.shape, targets.max(), targets.dtype) #targets (5000,) 49 int32
    # sqdist (5000, 50)
    
    correct = np.equal(pseudo_label, targets).sum()
    samples = targets.shape[0]
    
    print(f'eval_task_cdist: {correct}/{samples} = {correct/samples}')
    return correct/samples


@torch.no_grad()
def build_exemplars(args, dataset, model):
    EPSILON = 1e-8
    
    task_id, n_exemplar = dataset.task_id, dataset.n_exemplar
    low_range, high_range = dataset.increments[task_id], dataset.increments[task_id+1]
    
    print(f"Building exemplars & Updating memory cls-id {low_range} - {high_range}")
    exemplars_idx = []
    exemplars_data = []
    exemplars_targets = []
    
    for class_id in range(0, high_range):
        data, loader = dataset.get_custom_loader(class_id, src='train', tsf="test")
        _,   _loader = dataset.get_custom_loader(class_id, src='train', tsf="flip") # apply horizontal-flip
        
        features, targets = model.extract_features(loader)  # feature from original image
        features_,    _   = model.extract_features(_loader) # feature from flipped image
        
        ## change herding method here
        if class_id >= low_range:  # herd exemplars for new class
            select_idx = herding.icarl_selection(features, n_exemplar)
            
            exemplars_idx.append(select_idx)
            exemplars_data.append(data[select_idx])
            exemplars_targets.append(targets[select_idx])
        else: # old class
            select_idx = dataset.exemplars_idx[class_id]
        
        # re-compute mean
        features_norm  = features / (np.linalg.norm(features, axis=1, keepdims=True) + EPSILON)
        features_norm_ = features_ / (np.linalg.norm(features_, axis=1, keepdims=True) + EPSILON)
        selected  = features_norm[select_idx, ...]
        selected_ = features_norm_[select_idx, ...]
        #exemplar_mean = np.mean(selected + selected_, 0)
        exemplar_mean = (np.mean(selected,0) + np.mean(selected_, 0))/2
        
        exemplar_mean /= (np.linalg.norm(exemplar_mean) + EPSILON)
        if class_id == 0 : print('class id', class_id, exemplar_mean[:10])
        else: print(class_id, end=',', flush=True)
        
        #print('examplar_mean', exemplar_mean.shape)
        dataset.exemplars_mean[class_id, :] = exemplar_mean
    print()    
    
    
    dataset.exemplars_idx += exemplars_idx    # update exemplars index
    if task_id == 0:
        dataset.exemplars_data = concat(exemplars_data)       # update exemplars data
        dataset.exemplars_targets = concat(exemplars_targets) # update exemplars targets
    else:
        dataset.exemplars_data = concat([dataset.exemplars_data, *exemplars_data])
        dataset.exemplars_targets = concat([dataset.exemplars_targets, *exemplars_targets])
    print(f'Exemplars Now: data :{dataset.exemplars_data.shape}, targets: {dataset.exemplars_targets.shape} {len(dataset.exemplars_idx)}')
    print('==== Building exemplars & Updating memory Done ====')
    
    
def analyze_results(args, dataset, model):
    pass

def windup_task(args, dataset, model):
    task_id = dataset.task_id
    dataset_name = dataset.name
    initial = dataset.initial
    increment = dataset.increment
    
    if not os.path.exists(f'./exp/{dataset_name}/init-{initial}_inc-{increment}/'):
        os.makedirs(f'./exp/{dataset_name}/init-{initial}_inc-{increment}/')
        print(f'make folder: ./exp/{dataset_name}/init-{initial}_inc-{increment}/')
    else:
        assert args.overwrite_resume, 'overwrite_resume the existed folder'
        print(f'overwrite folder: ./exp/{dataset_name}/init-{initial}_inc-{increment}')
        
    if args.save:
        net_path = f'./exp/{dataset_name}/init-{initial}_inc-{increment}/{args.label}_net_{task_id}.pt'
        meta_path = f'./exp/{dataset_name}/init-{initial}_inc-{increment}/{args.label}_meta_{task_id}.pkl'
        
        torch.save(model.network.state_dict(), net_path)
        with open(meta_path, "wb+") as f:
            pickle.dump(
                [dataset.exemplars_idx, dataset.exemplars_mean,
                 dataset.exemplars_data, dataset.exemplars_targets],
                f
            )
        print(f"Network & Meta-data saved at {net_path}, {meta_path}.")
    