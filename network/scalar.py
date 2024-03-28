

import torch.nn as nn
import torch

class FactorScalar(nn.Module):

    def __init__(self, initial_value=1., **kwargs):
        super().__init__()

        self.factor = nn.Parameter(torch.tensor(initial_value))

    def on_task_end(self):
        pass

    def on_epoch_end(self):
        pass

    def forward(self, inputs):
        return self.factor * inputs

    def __mul__(self, other):
        return self.forward(other)

    def __rmul__(self, other):
        return self.forward(other)
    

if __name__ == '__main__':
    factor = FactorScalar()