import torch
from torch import nn
from Utils import commons_utils, cuda_utils
from Models.Activation.CustomActivation import CustomActivation

# Mish Dual Adaptive
class MiDA_old(CustomActivation):
    def __init__(self, dim, modelo_gen, init_strategy={'name': "default"}):
        super(MiDA_old, self).__init__(dim, modelo_gen, dominio_0_1=False)

        self.a = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))
        self.b = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))

        if init_strategy['name'] == "default":
            self._default_weights_initialization()
        else:
            self._initialize_weights(init_strategy, self.a)
            self._initialize_weights(init_strategy, self.b)

    def _default_weights_initialization(self):
        a_mean_cte = 1
        b_mean_cte = 0

        a_std_cte = 0.05
        b_std_cte = 0.001

        nn.init.normal_(self.a, mean=a_mean_cte, std=a_std_cte)
        nn.init.normal_(self.b, mean=b_mean_cte, std=b_std_cte)

    def forward(self, x):
        result = self.a * x * torch.tanh(torch.log(1 + torch.exp(x + self.b)))
        return result

    def get_parameters_names(self):
        return ['mida_alpha', 'mida_beta']
    
    def get_parameter_statistics(self, name):
        if name == 'mida_alpha':
            return self.a.mean().item(), self.a.max().item(), self.a.min().item()
        elif name == 'mida_beta':
            return self.b.mean().item(), self.b.max().item(), self.b.min().item()
        else:
            raise Exception("The name '" + name +"' is not a valid trainable parameter name")