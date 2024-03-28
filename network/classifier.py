

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.distance import stable_cosine_distance

import logging
logger = logging.getLogger(__name__)


class CosineClassifier(nn.Module):
    def __init__(self, features_dim):
        super(CosineClassifier, self).__init__()
        self.device = 'cuda'
        self.n_classes = 0
        self._weights = nn.ParameterList([])
        self.proxy_per_class = 10
        self.features_dim = features_dim
        self.bias = None
        self.scaling = 3.
        self.gamma = 1.
        
        # adaptive alpha in eqn. 6. New alpha vectors are added in func `self.add_classes()`
        self._alpha = nn.ParameterList([])  
        # self.register_buffer('_alpha', )
        
    
    @property
    def weights(self):
        return torch.cat([_weight for _weight in self._weights])
    
    @property
    def new_weights(self):
        return self._weights[-1]
    
    @property
    def alpha(self):
        return torch.cat([_alpha for _alpha in self._alpha])
    
    def forward(self, features):
        weights = self.weights
        
        features = self.scaling * F.normalize(features, p=2, dim=-1)
        weights = self.scaling * F.normalize(weights, p=2, dim=-1)
        raw_similarities = -stable_cosine_distance(features, weights)
        
        if self.proxy_per_class > 1:
            similarities = self._reduce_proxies(raw_similarities)
        else:
            similarities = raw_similarities
        
        return {"logits": similarities, "raw_logits": raw_similarities}
    
    def add_classes(self, n_classes):
        logger.info(f'add_class: old classes: {self.n_classes}, new classes: {n_classes}')
        ## add new weight
        new_weights = nn.Parameter(torch.zeros(self.proxy_per_class * n_classes, self.features_dim))
        nn.init.kaiming_normal_(new_weights, nonlinearity="linear")
        self._weights.append(new_weights)

        ## add new alpha
        new_alpha = nn.Parameter(torch.zeros(n_classes))
        nn.init.normal_(new_alpha)
        self._alpha.append(new_alpha)
        
        self.to(self.device)
        self.n_classes += n_classes
        
    
    
    def _reduce_proxies(self, similarities):
        """
          similarities : (batch_size, n_classes * proxy_per_class)
             return    : (batch_size, n_classes)
        """
        n_classes = similarities.shape[1] / self.proxy_per_class
        assert n_classes.is_integer(), (similarities.shape[1], self.proxy_per_class)
        n_classes = int(n_classes)
        bsz = similarities.shape[0]
        
        simi_per_class = similarities.view(bsz, n_classes, self.proxy_per_class)
        
        # 1. adaptive balanced softmax
        # attentions = F.softmax(self.alpha.unsqueeze(1).tile(1, self.proxy_per_class) * simi_per_class, dim=-1) 
        attentions = F.softmax(self.alpha.unsqueeze(1).repeat(1, self.proxy_per_class) * simi_per_class, dim=-1) 
        
        # 2. vanilla softmax
        # attentions = F.softmax(self.gamma * simi_per_class, dim=-1)  
        
        return (simi_per_class * attentions).sum(-1)
    
    
if __name__ == "__main__":
    cosincls = CosineClassifier()
    #print(cosincls.__dict__)
    cosincls.add_classes(50)