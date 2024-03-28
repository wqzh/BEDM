
from copy import deepcopy as dcp
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch

import network.rebuffi_cifar_resnet as rebuffi
from network.resnet import resnet18

from network import classifier
from network.scalar import FactorScalar

class BasicNet(nn.Module):
    def __init__(self, 
                 features_dim=64,
                 convnet="",
                 device='cuda'):
        super(BasicNet, self).__init__()
        
        self.device = device
        if convnet=="rebuffi":
            self.convnet = rebuffi.resnet_rebuffi()
        elif convnet=="resnet18":
            self.convnet = resnet18()
        else:
            raise Exception('undefined convnet {}'.format(convnet))
        
        self.convnet.to(device)
        self.classifier = classifier.CosineClassifier(features_dim=features_dim)
        #self.scalar = FactorScalar(initial_value=1.)
    
    def forward(self, x):
        outputs = self.convnet(x) # 'raw_features', 'features', 'attention'
        
        features = outputs["raw_features"]
        clf_outputs = self.classifier(features) # 'raw_logits', 'logits'
        outputs.update(clf_outputs)
        
        return outputs
    
    def get_group_parameters(self):
        groups = {"convnet": self.convnet.parameters()}
        # groups = {}
        
        if hasattr(self, "scalar") and isinstance(self.scalar, FactorScalar):
            groups["scalar"] = self.scalar.parameters()
        if hasattr(self.classifier, "new_weights"):
            groups["new_weights"] = self.classifier.new_weights
        if hasattr(self.classifier, "old_weights"):
            groups["old_weights"] = self.classifier.old_weights
        
        return groups
    
    
    @torch.no_grad()
    def extract(self, x):
        outputs = self.convnet(x)
        return outputs['raw_features']
    
    def copy(self,):
        return dcp(self,)
    
    
class Cifar100Net(nn.Module):
    def __init__(self, 
                 args, 
                 device='cuda'):
        super(Cifar100Net, self).__init__()
        
        self.device = device
        self.task_id = 0
        self.epoch = 0
        
        self.network = BasicNet(
            convnet=args.backbone, # "rebuffi"
            features_dim=args.features_dim
        )
        self._network = None # old (teacher) network
    
    def add_classes(self, n_classes):
        self.new_model.classifier.add_classes(n_classes)
    
    @torch.no_grad()
    def extract_features(self, loader):
        features, targets = [], []
        for data, target in loader:
            _targets = target.numpy()
            _features = self.network.extract(data.cuda()).detach().cpu().numpy()
            
            features.append(_features)
            targets.append(_targets)
        
        return np.concatenate(features), np.concatenate(targets)
    
    
    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()


class ImgnetNet(nn.Module):
    def __init__(self, 
                 args,
                 device='cuda'):
        super(ImgnetNet, self).__init__()
        
        self.device = device
        self.task_id = 0
        self.epoch = 0
        
        self.network = BasicNet(
            convnet=args.backbone, #"resnet18",
            features_dim=args.features_dim
        )
        self._network = None # old (teacher) network
    
    def add_classes(self, n_classes):
        self.new_model.classifier.add_classes(n_classes)
    
    @torch.no_grad()
    def extract_features(self, loader):
        features, targets = [], []
        for data, target in loader:
            _targets = target.numpy()
            _features = self.network.extract(data.cuda()).detach().cpu().numpy()
            
            features.append(_features)
            targets.append(_targets)
        
        return np.concatenate(features), np.concatenate(targets)
    
    
    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()
    
    
    
if __name__ == "__main__":
    
    model = Cifar100Net()
    model.network.classifier.add_classes(50)
    out1 = model.network(torch.rand(1,3,32,32).cuda())
    model._network = model.network.copy().eval()
    
    model.network.classifier.add_classes(10)
    out2 = model.network(torch.rand(2,3,32,32).cuda())
    
    # torch.save(model.network.state_dict(), './ware/model.pt',)
