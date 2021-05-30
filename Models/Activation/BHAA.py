import torch
from torch import nn
from Utils import cuda_utils
from Models.Activation.CustomActivation import CustomActivation
class BHAA(CustomActivation):
    def __init__(self, dim,modelo_gen, dominio_0_1=True, dcgan=False, init_strategy={'name': "default"}, truncated=True):
        super(BHAA, self).__init__(dim, modelo_gen, dominio_0_1)

        self.truncated = truncated

        self.l = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))
        self.t1 = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))
        self.t2 = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))

        if init_strategy['name'] == "default":
            self._default_weights_initialization()
        else:
            self._initialize_weights(init_strategy, self.t1)
            self._initialize_weights(init_strategy, self.t2)
            self._initialize_weights(init_strategy, self.l)

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

        nn.init.normal_(self.t1, mean=t_mean, std=t_std)
        nn.init.normal_(self.t2, mean=t_mean, std=t_std)
        nn.init.normal_(self.l, mean=l_mean, std=l_std)

    def forward(self, x):
        result = self.evaluate_raw_bhaa(x)
        
        # Using hard limitation to avoid out of range error
        if self.truncated:
            cpy = result.clone()
            cpy[result > 1] = 1
            if self.dominio_0_1: cpy[result < 0] = 0
            else: cpy[result < -1] = -1
            return cpy
        else:
            return result
    
    def evaluate_raw_bhaa(self, x):
        # bi-hiperbólica assimétrica adaptativa (BHAA)
        # bhaa(l, t1, t2) = h1 - h2
        # h1 = sqrt(( (l^2)*((x + (1/(2*l)))^2) ) + t1^2)
        # h2 = sqrt(( (l^2)*((x + (1/(2*l)))^2) ) + t2^2)
        if self.dominio_0_1:
            # h1 = torch.sqrt( ((self.l**2) * (x + (1/(4*self.l)))**2) + self.t1**2 )
            # h2 = torch.sqrt( ((self.l**2) * (x - (1/(4*self.l)))**2) + self.t2**2 )
            h1 = 0.25 * torch.sqrt((1 + 4*self.l*x)**2 + (16*(self.t1**2)))
            h2 = 0.25 * torch.sqrt((1 - 4*self.l*x)**2 + (16*(self.t2**2)))
            result = h1 - h2 + 0.5
        else:
            # h1 = torch.sqrt( ((self.l**2) * (x + (1/(2*self.l)))**2) + self.t1**2 )
            # h2 = torch.sqrt( ((self.l**2) * (x - (1/(2*self.l)))**2) + self.t2**2 )
            h1 = 0.5 * torch.sqrt((1 + 2*self.l*x)**2 + (4*(self.t1**2)))
            h2 = 0.5 * torch.sqrt((1 - 2*self.l*x)**2 + (4*(self.t2**2)))
            result = h1 - h2
        
        return result

    def get_lambda_statistics(self):
        return self.l.mean().item(), self.l.max().item(), self.l.min().item()

    def is_asymmetric(self):
        return True

    def get_t1_statistics(self):
        return self.t1.mean().item(), self.t1.max().item(), self.t1.min().item()

    def get_t2_statistics(self):
        return self.t2.mean().item(), self.t2.max().item(), self.t2.min().item()

    def get_parameters_names(self):
        return ['bhaa_tau1', 'bhaa_tau2', 'bhaa_lambda']
    
    def get_parameter_statistics(self, name):
        if name == 'bhaa_tau1':
            return self.t1.mean().item(), self.t1.max().item(), self.t1.min().item()
        elif name == 'bhaa_tau2':
            return self.t2.mean().item(), self.t2.max().item(), self.t2.min().item()
        elif name == 'bhaa_lambda':
            return self.l.mean().item(), self.l.max().item(), self.l.min().item()
        else:
            raise Exception("The name '" + name + "' is not a valid trainable parameter name")
