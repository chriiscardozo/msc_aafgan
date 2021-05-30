import torch
from torch import nn
from Utils import cuda_utils
from Models.Activation.CustomActivation import CustomActivation

class dSiLU(CustomActivation):
    def __init__(self, dim, modelo_gen, dominio_0_1=True, dcgan=False, init_strategy={'name': "default"}):
        super(dSiLU, self).__init__(dim, modelo_gen, dominio_0_1)

        self.b = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))

        if init_strategy['name'] == "default":
            self._default_weights_initialization()
        else:
            self._initialize_weights(init_strategy, self.b, param_name='b')

    def _default_weights_initialization(self):
        b_cte = 0.58
        if self.modelo_gen: b_cte = 1
        self.b = nn.Parameter(self.b.new_full(self.dim, fill_value=b_cte, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))

    def forward(self, x):
        if self.dominio_0_1:
            result = ((((torch.exp(self.b * x) * (1 + torch.exp(self.b * x) + self.b * x))/(1 + torch.exp(self.b * x))**2))+0.1)/1.28
        else:
            result = ((((((torch.exp(self.b * x) * (1 + torch.exp(self.b * x) + self.b * x))/(1 + torch.exp(self.b * x))**2))+0.1)/1.28) - 0.5) * 2

        return result

    def get_beta_statistics(self):
        return self.b.mean().item(), self.b.max().item(), self.b.min().item()

    def is_asymmetric(self):
        return False
