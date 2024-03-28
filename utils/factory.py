from torch.optim import lr_scheduler
from torch import optim
import torch

import logging


def get_optimizer(config):
    optimizer, params, lr, weight_decay = \
        config['type'], config['params'], config['lr'], config['weight_decay']
    
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer == "sgd_nesterov":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown Optimizer type {config['type']}.")
    

def get_scheduler(config):
    scheduler, optimizer, epochs, = \
        config['type'], config['optimizer'], config['epochs']
    
    if scheduler == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, epochs )
    else:
        raise ValueError(f"Unknown LR Sheduler type {config['type']}.")


def set_logging_format(logging_level='info'):
    logging_level = logging_level.lower()

    if logging_level == "critical":
        level = logging.CRITICAL
    elif logging_level == "warning":
        level = logging.WARNING
    elif logging_level == "info":
        level = logging.INFO
    else:
        level = logging.DEBUG
    
    logging.basicConfig(
        format='%(asctime)s [%(filename)s] line-%(lineno)d: %(message)s ', 
        datefmt='%m-%d %H:%M:%S', level=level
    )


if __name__ == "__main__":
    from network import rebuffi_cifar_resnet as rebuffi
    model = rebuffi.resnet_rebuffi()
    model.cuda()
    
    params = model.parameters()
    optimizer_config = {'type':'sgd', 'params':params, 'lr': 0.1, 'weight_decay': 0.0005}
    optimizer = get_optimizer(optimizer_config)
    
    scheduler_config = {'type':'cosine', 'optimizer':optimizer, 'epochs':10}
    scheduler = get_scheduler(scheduler_config)



