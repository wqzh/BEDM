import os
import os.path as osp
import numpy as np
from PIL import Image
from numpy import concatenate as concat

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from torchvision import transforms

import logging
logger = logging.getLogger(__name__)

class iCifar100:
    '''
    Incremental Cifar100 Dataset.
    '''
    
    def __init__(
            self,
            args,
            root='data',  # a dir which includes the folder `cifar-100-python/`
            workers=0,
            batch_size=128,
            initial=50,
            increment=10,
            n_exemplar = 20,
            class_order=None,
            drop_last=False
        ):
        
        self.args = args
        self.name = "cifar100"
        root = args.root
        
        class_order = args.class_order
        assert isinstance(class_order, list), 'class_order should be list'
        assert sum(class_order)==4950, 'clsid(range:0~99) in class_order should be unique'
        assert (100-initial) % increment == 0, 'not divisible'
        
        self.n_classes = 100
        self.features_dim = args.features_dim
        
        self.n_exemplar = n_exemplar
        self.initial = initial
        self.increment = increment
        self.n_task = 1 + (100-initial)//increment
        self.increments = [0, initial] # e.g. [0, 50, 60, ...]
        for i in range(1, self.n_task): self.increments.append(self.increments[-1]+increment) 
        
        self.task_id = 0
        self.init_task = 0
        
        self.exemplars_idx = []      # e.g. [[1, 43,..., 490]], list of list, element range: [0,500)
        self.exemplars_data = None   # e.g. (20*n, 32, 32, 3) <numpy.ndarray> uint8
        self.exemplars_targets = None  # e.g. [0, 0,..., 1, 1,...,99, 99], <numpy.ndarray>
        self.exemplars_mean = np.zeros((self.n_classes, self.features_dim)) #float64
        
        self.workers = workers
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.trainset, self.testset = None, None
        
        self.class_order = class_order
        self.__dict__.update(**download_cifar100(root))
        self.map_new_class_index()
        
        self.train_tsf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        self.test_tsf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        self.flip_tsf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1.),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        
    
    @staticmethod
    def _map_new_class_index( y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))
    
    def map_new_class_index(self,):
        """ e.g. 
        old class index : [87, 0, 52, 58, 44, 91, 68, ...]
                           ↓   ↓   ↓   ↓   ↓   ↓   ↓
        new class index : [0,  1,  2,  3,  4,  5,  6, ...]
        """
        print('==> setting class order:', self.class_order)
        for key in ['y_train', 'y_test']:
            self.__dict__[key] = self._map_new_class_index(self.__dict__[key], self.class_order)
    
    def get_loader(self, task_id, test=False):
        """ Get trainloader & testloader for every task. 
        trainloader : trainset + exemplar,  train_transform
        testloader  :       testset,         test_transform
        """
        
        # use cache (not first epoch)
        if self.trainset is not None:
            trainloader = DataLoader(self.trainset, batch_size=self.batch_size, 
                                shuffle=True,num_workers=self.workers,)
            if test:
                testloader = DataLoader(self.testset, batch_size=self.batch_size, 
                                shuffle=False, num_workers=self.workers,)
                return trainloader, testloader
            return trainloader, None
        
        # create cache, only in the first epoch
        low_range, high_range = self.increments[task_id], self.increments[task_id+1] # e.g. [0,50)
        
        idx = np.where(np.logical_and(self.y_train >= low_range, self.y_train < high_range))[0]
        x_train, y_train = self.x_train[idx], self.y_train[idx] # ndarray, list
        if self.exemplars_data is not None:
            x_exemplar, y_exemplar = self.exemplars_data, self.exemplars_targets # ndarray, ndarray
            x_train    = concat([x_train, x_exemplar])   #concat: ndarray, ndarray.
            y_train = concat([y_train, y_exemplar])   #concat: list, ndarray. Legal operation in numpy
        #print('self.x_train ', self.x_train.shape)
        trainset = DummyDataset(x_train, y_train, self.train_tsf)
        self.trainset = trainset
        trainloader = DataLoader( trainset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.workers,)
        
        if test: # get testloader
            idx = np.where(np.logical_and(self.y_test >= 0, self.y_test < high_range))[0]
            x_test, y_test = self.x_test[idx], self.y_test[idx]
            
            testset = DummyDataset(x_test, y_test, self.test_tsf)
            self.testset = testset
            testloader = DataLoader(
                            testset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.workers,)
            logger.info(f'T{task_id+1}/{self.n_task}, cls-id:{low_range}-{high_range}, train: {len(trainset)}, test: {len(testset)}')
            #return {'trainloader' : trainloader, 'testloader' : testloader}
            return trainloader, testloader
        
        #return {'trainloader' : trainloader} 
        logger.info(f'T{task_id+1}/{self.n_task}, cls-id:{low_range}-{high_range}, train: {len(trainset)}')
        return trainloader, None
    
    def get_custom_loader(self, class_id, src='train', tsf='test',):
        """ Get dataloader for specified class-id. 
        src  : data from `train` or `test` set.         choices :['train', 'test']
        tsf  : transform mode to be applied in Dataset. choices :['train', 'test', 'flip']
        """
        
        data, targets = [], []
        x, y = (self.x_train, self.y_train,) if src=='train' else (self.x_test, self.y_test)
        #tsf = self.train_tsf if tsf=='train' else self.test_tsf
        if tsf=='flip': tsf_ = self.flip_tsf
        elif tsf=='test': tsf_ = self.test_tsf
        else: tsf_ = self.train_tsf
        
        #low_range, high_range = self.increments[task_id], self.increments[task_id+1]
        #for class_id in range(low_range, high_range):
        #    idx = np.where(np.logical_and(y >= class_id, y < class_id+1))[0]
        #    data.append(x[idx])
        #    targets.append(y[idx])
        idx = np.where(np.logical_and(y >= class_id, y < class_id+1))[0]
        data.append(x[idx])
        targets.append(y[idx])
        
        assert len(data), f'class_id:{class_id}, empty data! Handle this error.'
        data = concat(data)
        targets = concat(targets)
        loader = DataLoader(
                    DummyDataset(data, targets, tsf_),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.workers,)
        
        #logger.info(f'Get class-id:{class_id}, from: {src}set, tsf: {tsf}, data:{data.shape}')
        return data, loader
    

class DummyDataset(Dataset):
    def __init__(self, x, y, tsf):
        """
        x : ndarray
        y : ndarray or list
        """
        self.x, self.y = x, y
        self.tsf = tsf

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        
        #x = Image.fromarray(x) #x.astype("uint8")
        img = self.tsf(x)
        # return {'data':img, 'targets':y}
        return img, y
    
    
def download_cifar100(root, cls_ord=None):
    if not osp.exists(root):
        os.mkdir(root)
        print(f'create folder {root}')
        
    if osp.exists(osp.join(root, 'cifar-100-python')):
        print('cifar-100-python already exists, skip download.')
        trainset = CIFAR100(root=root, train=True, download=False) # `data`, `targets`
        valset = CIFAR100(root=root, train=False, download=False)  # `data`, `targets`
    else:
        trainset = CIFAR100(root=root, train=True, download=True) # `data`, `targets`
        valset = CIFAR100(root=root, train=False, download=True)  # `data`, `targets`
        print(f'download cifar-100-python to {root}')
        
    print('train: ', type(trainset.data), trainset.data.shape, trainset.data.dtype) 
    #<class 'numpy.ndarray'> (50000, 32, 32, 3) uint8
    print('       ', type(trainset.targets), len(trainset.targets), type(trainset.targets[0]))
    #<class 'list'> 50000
    print('test:  ',type(valset.data), valset.data.shape, valset.data.dtype)
    #<class 'numpy.ndarray'> (10000, 32, 32, 3) uint8
    print('       ', type(valset.targets), len(valset.targets), type(valset.targets[0]))
    #<class 'list'> 10000
    #return [trainset.data, trainset.targets, valset.data, valset.targets]
    
    cifar100 = { 'x_train' : trainset.data, 'y_train' : trainset.targets,
                 'x_test'  : valset.data,   'y_test'  : valset.targets,}
    return cifar100


if __name__ == "__main__":
    import os, sys
    
    dirname = os.path.dirname(__file__)
    project_path = os.path.dirname(dirname)
    utils_path = f'{project_path}/utils'
    if project_path not in sys.path: sys.path.insert(0, project_path)
    # if utils_path not in sys.path: sys.path.insert(0, utils_path)
    print(sys.path)
    
    from utils.misc import get_option
    args = get_option('./options/cifar100.yaml')
    cifar = iCifar100(args)
    
    