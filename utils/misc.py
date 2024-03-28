
import yaml
import json
import os
import os.path as osp
import pickle

import random
import numpy as np
import torch

def logger_args(args):
    print('============== ARGS BEGIN ============================> ')
    for k, v in args.items():
        print(k, ":", v)
    print('<============= ARGS END ============================= ')


def set_seed(seed=1234567, deterministic=True):
    print("Set seed {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        print("CUDA algos are determinists but very slow!")
    else:
        print("CUDA algos are not determinists but faster!")
    torch.backends.cudnn.deterministic = deterministic

def load_resume(args, model, dataset):
    load_id, net, meta = args.resume, args.net, args.meta
    
    print(f'Resume task_id: {load_id} model & meta.')
    dataset.init_task = load_id + 1
    
    model.network.load_state_dict(torch.load(net))
    #model.network.load_state_dict(torch.load(net), strict=False)
    
    with open(meta, "rb") as f:
        dataset.exemplars_idx, dataset.exemplars_mean, \
        dataset.exemplars_data, dataset.exemplars_targets  = pickle.load(f)
        
        # podnet format
        # https://github.com/arthurdouillard/incremental_learning.pytorch
        #dataset.exemplars_data, dataset.exemplars_targets, \
        #dataset.exemplars_idx, dataset.exemplars_mean  = pickle.load(f)
        #print(type(dataset.exemplars_idx), type(dataset.exemplars_mean), type(dataset.exemplars_data), type(dataset.exemplars_targets))
        #<class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'list'> <class 'numpy.ndarray'>
        #print(dataset.exemplars_idx.shape, dataset.exemplars_mean.shape, len(dataset.exemplars_data), dataset.exemplars_targets.shape)
        
    #print(f' ==== Load Exemplars: data: {dataset.exemplars_data.shape}, targets:{dataset.exemplars_targets.shape} ====')
    #trainloader, testloader = dataset.get_loader(load_id, True)
    #print('test on the dataset.')

def overwrite_remind(args):
    if args.overwrite_resume: 
        return
    else:
        ckpt_path = f'./exp/{args.dataset}'
        if osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0:
            raise Exception('overwrite_resume is false, but the dir {} is not empty'.format(ckpt_path))
    
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    
def dict2obj(dct):
    """ 
    Transfer a `dict` object to `Dict` object. 
    So we can use `.` operator to acces the data.
    e.g. args['epochs']  -> args.epochs
    """
    if not isinstance(dct, dict):
        return dct
    obj = Dict()
    for k, v in dct.items():
        obj[k] = dict2obj(v)
    return obj


def get_option(path):
    """
    It is recommended to write all the arguments in one `.yaml` or `.json` file.
    """
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            #return yaml.load(f, Loader=yaml.FullLoader)
            return dict2obj(yaml.load(f, Loader=yaml.FullLoader))
        elif path.endswith(".json"):
            return json.load(f)["config"]
        else:
            raise Exception("Unknown file type {}.".format(path))

def get_options(paths):
    assert isinstance(paths, list), f'a list of configuration file path, not {paths}'
    args = {}
    for path in paths:
        if os.path.exists(path):
            arg = get_option(path)
        #print(path, arg)
        args.update(arg)
    #print(args)
    return dict2obj(args)


if __name__ == "__main__":
    #args = get_options(['./options/test.yaml'])
    arg = get_option('./options/cifar100.yaml')
    #print(arg)
    
    inst = dict2obj(arg)
