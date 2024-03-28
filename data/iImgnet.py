import os.path as osp
from PIL import Image
import numpy as np
from numpy import concatenate as concat

from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision import transforms

import logging
logger = logging.getLogger(__name__)


class iImageNet:
    '''
    Incremental ImageNet Dataset.
    '''
    
    def __init__(
            self,
            args,
            workers=0,
            batch_size=128,
            initial=50,
            increment=10,
            n_exemplar = 20,
            class_order=None,
            drop_last=False
        ):
        self.args = args
        self.name = args.dataset
        
        n_classes = args.n_classes
        class_order = args.class_order
        assert isinstance(class_order, list), 'class_order should be list'
        assert sum(class_order)==n_classes*(n_classes-1)//2, f'clsid(range:0~{n_classes}) in class_order should be unique'
        assert (n_classes-initial) % increment == 0, 'not divisible'
        
        self.n_classes = args.n_classes
        self.features_dim = args.features_dim
        
        self.n_exemplar = n_exemplar
        self.initial = initial
        self.increment = increment
        self.n_task = 1 + (n_classes-initial)//increment
        self.increments = [0, initial] # e.g. [0, 50, 60, ...]
        for i in range(1, self.n_task): self.increments.append(self.increments[-1]+increment) 
        
        self.task_id = 0
        self.init_task = 0
        
        self.exemplars_idx = []      # e.g. [[1, 43,..., 490]], list of list, element range: [0,500)
        self.exemplars_data = None   # e.g. (20*n, h, w, c) <numpy.ndarray> uint8
        self.exemplars_targets = None  # e.g. [0, 0,..., 1, 1,...,99, 99], <numpy.ndarray>
        self.exemplars_mean = np.zeros((self.n_classes, self.features_dim)) #float64
        
        self.workers = workers
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.trainset, self.testset = None, None
        
        self.class_order = class_order
        
        # self.get_dataset(args.metadata_path[0], is_train=True)
        # self.get_dataset(args.metadata_path[1], is_train=False)
        self.__dict__.update(**self.get_dataset(args.metadata_path[0], is_train=True))
        self.__dict__.update(**self.get_dataset(args.metadata_path[1], is_train=False))
        
        # self.map_new_class_index()
        
        self.train_tsf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=63 / 255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        self.test_tsf = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.flip_tsf = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        
    def get_loader(self, task_id, test=False):
        """ Get trainloader & testloader for every task. 
        trainloader : trainset + exemplar,  train_transform
        testloader  :       testset,         test_transform
        """
        
        # use cache (not first epoch)
        if self.trainset is not None:
            trainloader = DataLoader(self.trainset, batch_size=self.batch_size, 
                                shuffle=True, num_workers=self.workers)
            if test:
                testloader = DataLoader(self.testset, batch_size=self.batch_size, 
                                shuffle=False, num_workers=self.workers)
                return trainloader, testloader
            return trainloader, None
        
        # create cache, only in the first epoch
        low_range, high_range = self.increments[task_id], self.increments[task_id+1] # e.g. [0,50)
        
        idx = np.where(np.logical_and(self.y_train >= low_range, self.y_train < high_range))[0]
        print('idx', idx.shape)
        
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
    
    
    def get_dataset(self, metadata_path, is_train):
        root = self.args.root
        
        split = "train" if is_train else "val"
        class_order = self.args.class_order
        index = [i for i in range(len(class_order))]
        maps = {k:v for k,v in zip(class_order, index)}

        data, targets = [], []
        with open(metadata_path) as f:
            for line in f:
                path, target = line.strip().split(" ")
                data.append(osp.join(root, path))
                #targets.append(int(target))      #[0,0,.., 1,1,...]
                targets.append(maps[int(target)])      #[69, 69,...,23, 23,...]
        
        if is_train:
            # self.trainset.data, self.trainset.targets = data, targets
            print('Load train set done!', 'samples :', len(data)) # e.g.  imgnet100: 129395; imgnet1000: 1281167
            print(data[0], target[0])
            # return {'x_train' : data, 'y_train' : targets}
            return {'x_train' : np.array(data), 'y_train' : np.array(targets)}
        else:
            # self.testset.data, self.testset.targets = data, targets
            print('Load test set done!', 'samples :', len(data)) # e.g.  imgnet100: 5000; imgnet1000: 50000
            print(data[0], target[0])
            # return {'x_test' : data,  'y_test'  : targets}
            return {'x_test' : np.array(data),  'y_test'  : np.array(targets)}
        

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
        path, y = self.x[idx], self.y[idx]
        
        #x = Image.fromarray(x) #x.astype("uint8")
        x = Image.open(path).convert('RGB')
        img = self.tsf(x)
        # return {'data':img, 'targets':y}
        return img, y
         
            
if __name__ == "__main__":
    import os, sys
    
    dirname = os.path.dirname(__file__)
    project_path = os.path.dirname(dirname)
    utils_path = f'{project_path}/utils'
    if project_path not in sys.path: sys.path.insert(0, project_path)
    # if utils_path not in sys.path: sys.path.insert(0, utils_path)
    print(sys.path)

    
    from utils.misc import get_option, logger_args
    args = get_option('./options/imgnet100.yaml')
    # args = get_option('./options/imgnet1000.yaml')
    logger_args(args)
    assert args.save, "You'd better save the model and meta-data"

    imgnet100 = iImageNet(args)
    