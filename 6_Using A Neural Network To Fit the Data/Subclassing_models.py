import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class SubclassSequential(nn.Module):
    def __init__(self, n_neurons):
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
                    ('hidden_linear', nn.Linear(1, n_neurons)),
                    ('hidden_activation', nn.Tanh()),
                    ('output_linear', nn.Linear(n_neurons, 1))
                                                ]))
    def forward(self, input):
        return self.model(input)





class SubclassModel(nn.Module):
    def __init__(self):
        super().__init__()  # <1>

        self.hidden_linear = nn.Linear(1, 13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)

        return output_t


class SubclassFunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden_linear = nn.Linear(1, 14)
        # <1>
        self.output_linear = nn.Linear(14, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = torch.tanh(hidden_t)  # <2>
        output_t = self.output_linear(activated_t)

        return output_t
