

from data.iImgnet import iImageNet as iImageNetData
from network.basic import ImgnetNet

from utils.factory import set_logging_format, get_optimizer, get_scheduler
from utils.train_engine import train_epoch, eval_task, eval_task_cdist, \
                            build_exemplars, analyze_results, windup_task
from utils.misc import load_resume, get_option, set_seed, logger_args, overwrite_remind

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, 
                    choices=['imgnet100', 'imgnet1000'],
                    help="dataset name")
meta_arg= parser.parse_args()


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import logging
set_logging_format()
logger = logging.getLogger(__name__)

# load arguments
set_seed()

args = get_option(f'./options/{meta_arg.dataset}.yaml')
# args = get_option('./options/imgnet100.yaml')
# args = get_option('./options/imgnet1000.yaml')
logger_args(args)
assert args.save, "You'd better save the model and meta-data"

# create dataset & model
imgnetdata = iImageNetData(args)
model = ImgnetNet(args)
model.cuda()

# print(model.network.convnet)
overwrite_remind(args)


# start training each task
accuracy_list = []
n_task = imgnetdata.n_task
for task_id in range(0, imgnetdata.n_task): #imgnetdata.init_task
    model.task_id, imgnetdata.task_id = task_id, task_id
    
    # classifier add new weight
    model.network.classifier.add_classes(args.initial if task_id == 0 else args.increment) # e.g. 50,10,10,...
    
    # resume check
    if args.resume is not None: # use resume
        if args.resume == task_id:
            load_resume(args, model, imgnetdata)
            if model._network is None : model._network = model.network.copy().eval() 
            _, testloader = imgnetdata.get_loader(task_id, True)
            eval_task(model, testloader)
            
            args.resume = None
        elif args.resume > task_id:
            print(f'==== Use Resume. Skip Training on T{task_id+1}/{n_task}. ====')
        continue
    
    # optimizer & scheduler initialization
    args.epochs = args.base_epochs if task_id == 0 else args.increment_epochs
    params = []
    for group_name, group_params in model.network.get_group_parameters().items():
        params.append({"params": group_params, "lr": args.lr})
        print(f"Param Group: {group_name}, lr: {args.lr}")
    
    
    optimizer_kwargs = args.optimizer_kwargs
    optimizer_kwargs['params'] = params
    optimizer = get_optimizer(optimizer_kwargs)
    
    scheduler_kwargs = args.scheduler_kwargs
    scheduler_kwargs["optimizer"] = optimizer
    scheduler_kwargs["epochs"] = args.epochs
    scheduler = get_scheduler(scheduler_kwargs)
    
    
    # dataloader
    imgnetdata.trainset, imgnetdata.testset = None, None
    trainloader, testloader = imgnetdata.get_loader(task_id, True)
    
    # use 2 losses in first task, otherwise all 3 losses are applied
    losses = args.losses[:2] if task_id == 0  else args.losses 
    
    
    # train one task
    for epoch in range(1, args.epochs+1):
        train_epoch(task_id, epoch, args, model, imgnetdata, \
                    optimizer, scheduler, trainloader, testloader, losses)
        
        # evaluate every `eval_frequency` epochs
        if epoch % args.eval_frequency == 0:
            # eval_task(model, trainloader)
            eval_task(model, testloader)
    
    # save old network
    model._network = model.network.copy().eval() 
    
    ## build exemplars
    build_exemplars(args, imgnetdata, model)
    
    ## eval task
    acc = eval_task(model, testloader)
    eval_task_cdist(model, imgnetdata, testloader)
    accuracy_list.append(acc)
    
    ## wind up task / resume model
    windup_task(args, imgnetdata, model)
    
    ## accuracy analysis / record 
    analyze_results(args, imgnetdata, model)
    
    print('----------->', accuracy_list, end='\n\n')