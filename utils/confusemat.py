
import numpy as np


class ConfusionMatrix(object):

    def __init__(self,num_classes:int,
                 labels:list=[],
                 verbose=False):
        self.matrix=np.zeros((num_classes,num_classes))
        self.num_classes=num_classes
        self.labels=labels
        self.verbose=verbose

    def update(self,preds,labels):
        for p,t in zip(preds,labels):
            self.matrix[p,t]+=1

    def summary(self):
        # Acccuracy
        sum_TP =0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i,i]
        acc = sum_TP/np.sum(self.matrix)
         
        predict = self.matrix.sum(0).astype('int')  # row sum, all predict
        truth = self.matrix.diagonal().astype('int')
        print("the model accuracy is ", acc)
        
        ret = {}
        ret['acc'] = [sum_TP, np.sum(self.matrix), round(acc, 4)]
        for i, (p, t) in enumerate(zip(predict, truth)):
            ret[f'cls-{i}'] = [t, p, round(t/p, 4)]
        if self.verbose:
            for k,v in ret.items():
                print(k, v)
        return ret
        
if __name__ == "__main__":
    conf = ConfusionMatrix(2, verbose=True)
    
    pred = [1, 0, 1, 1]
    targ = [0, 1, 1, 0]
    conf.update(preds = pred, labels=targ)
    
    pred = [1, 0, 1, 1, 0, 0, 1]
    targ = [0, 1, 1, 0, 0, 1, 1]
    conf.update(preds = pred, labels=targ)
    
    ret = conf.summary()
    # print(ret)