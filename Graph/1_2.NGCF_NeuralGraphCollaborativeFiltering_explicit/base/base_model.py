import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
