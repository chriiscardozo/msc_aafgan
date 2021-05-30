import torch
from torch import nn
from Utils import commons_utils, cuda_utils
from Models.Activation.CustomActivation import CustomActivation

class SHReLU(CustomActivation):
    def __init__(self, dim, modelo_gen, init_strategy={'name': "default"}):
        super(SHReLU, self).__init__(dim, modelo_gen, dominio_0_1=False)

        self.t = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))

        if init_strategy['name'] == "default":
            self._default_weights_initialization()
        else:
            self._initialize_weights(init_strategy, self.l)

    def _default_weights_initialization(self):
        t_mean_cte = 0.0
        t_std_cte = 0.2

        nn.init.normal_(self.t, mean=t_mean_cte, std=t_std_cte)

    def forward(self, x):
        # SHReLU - Suavizacao hiperbolica da ReLU
        # shrelu(t) = 0.5*(x + sqrt(x^2 + t^2))

        result = 0.5 * (x + torch.sqrt((x**2) + (self.t**2)))

        return result

    def get_parameters_names(self):
        return ['shrelu_tau']
    
    def get_parameter_statistics(self, name):
        if name == 'shrelu_tau':
            return self.t.mean().item(), self.t.max().item(), self.t.min().item()
        else:
            raise Exception("The name '" + name +"' is not a valid trainable parameter name")