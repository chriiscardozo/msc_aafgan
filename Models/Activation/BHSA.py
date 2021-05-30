import torch
from torch import nn
from Utils import cuda_utils
from Models.Activation.CustomActivation import CustomActivation
class BHSA(CustomActivation):
    def __init__(self, dim, modelo_gen, dominio_0_1=True, dcgan=False, init_strategy={'name': "default"}):
        super(BHSA, self).__init__(dim, modelo_gen, dominio_0_1)

        self.l = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))
        self.t = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))

        if init_strategy['name'] == "default":
            self._default_weights_initialization()
        else:
            self._initialize_weights(init_strategy, self.t, param_name='t')
            self._initialize_weights(init_strategy, self.l, param_name='l')

    def _default_weights_initialization(self):
        # for generator
        t_mean = 0.25
        t_std = 0.05
        l_mean = 0.5
        l_std = 0.05

        # for discriminator 
        if(self.dominio_0_1):
            t_mean = 0.7
            t_std = 0.05
            l_mean = 0.5
            l_std = 0.05

        nn.init.normal_(self.t, mean=t_mean, std=t_std)
        nn.init.normal_(self.l, mean=l_mean, std=l_std)

    def forward(self, x):
        # bi-hiperbolic simetric adaptative
        # bhsa(l, t) = h1 - h2

        if self.dominio_0_1:
            # h1 = torch.sqrt( ((self.l**2) * (x + (1/(4*self.l)))**2) + self.t1**2 )
            # h2 = torch.sqrt( ((self.l**2) * (x - (1/(4*self.l)))**2) + self.t2**2 )
            h1 = 0.25 * torch.sqrt((1 + 4*self.l*x)**2 + (16*(self.t**2)))
            h2 = 0.25 * torch.sqrt((1 - 4*self.l*x)**2 + (16*(self.t**2)))
            result = h1 - h2 + 0.5
        else:
            # h1 = torch.sqrt( ((self.l**2) * (x + (1/(2*self.l)))**2) + self.t1**2 )
            # h2 = torch.sqrt( ((self.l**2) * (x - (1/(2*self.l)))**2) + self.t2**2 )
            h1 = 0.5 * torch.sqrt((1 + 2*self.l*x)**2 + (4*(self.t**2)))
            h2 = 0.5 * torch.sqrt((1 - 2*self.l*x)**2 + (4*(self.t**2)))
            result = h1 - h2

        return result

    def get_lambda_statistics(self):
        return self.l.mean().item(), self.l.max().item(), self.l.min().item()

    def is_asymmetric(self):
        return False

    def get_t_statistics(self):
        return self.t.mean().item(), self.t.max().item(), self.t.min().item()
    
    def get_parameters_names(self):
        return ['bhsa_tau', 'bhsa_lambda']
    
    def get_parameter_statistics(self, name):
        if name == 'bhsa_tau':
            return self.t.mean().item(), self.t.max().item(), self.t.min().item()
        elif name == 'bhsa_lambda':
            return self.l.mean().item(), self.l.max().item(), self.l.min().item()
        else:
            raise Exception("The name '" + name + "' is not a valid trainable parameter name")