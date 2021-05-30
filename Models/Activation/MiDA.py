import torch
from torch import nn
from Utils import cuda_utils
from Models.Activation.CustomActivation import CustomActivation

# Mish Dual Adaptive w/ BHSA as self-gating function
class MiDA(CustomActivation):
    def __init__(self, dim, modelo_gen, init_strategy={'name': "default"}):
        super(MiDA, self).__init__(dim, modelo_gen, dominio_0_1=False)

        self.l = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))
        self.t = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))

        if init_strategy['name'] == "default":
            self._default_weights_initialization()
        else:
            self._initialize_weights(init_strategy, self.l, param_name='l')
            self._initialize_weights(init_strategy, self.t, param_name='t')

    def _default_weights_initialization(self):
        t_mean_cte = 0.25
        l_mean_cte = 0.7

        t_std_cte = 0.01
        l_std_cte = 0.01

        nn.init.normal_(self.t, mean=t_mean_cte, std=t_std_cte)
        nn.init.normal_(self.l, mean=l_mean_cte, std=l_std_cte)


    def function_bhsa(self, x, ptau, plambda):
        h1 = 0.5 * torch.sqrt((1 + 2*plambda*x)**2 + (4*(ptau**2)))
        h2 = 0.5 * torch.sqrt((1 - 2*plambda*x)**2 + (4*(ptau**2)))
        return h1 - h2

    def forward(self, x):
        softmax = torch.log(1 + torch.exp(x))
        result = x * self.function_bhsa(softmax, self.t, self.l)
        return result

    def get_parameters_names(self):
        return ['mida_tau', 'mida_lambda']
    
    def get_parameter_statistics(self, name):
        if name == 'mida_tau':
            return self.t.mean().item(), self.t.max().item(), self.t.min().item()
        elif name == 'mida_lambda':
            return self.l.mean().item(), self.l.max().item(), self.l.min().item()
        else:
            raise Exception("The name '" + name +"' is not a valid trainable parameter name")