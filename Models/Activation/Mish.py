import torch
from torch import nn
from Utils import commons_utils, cuda_utils
from Models.Activation.CustomActivation import CustomActivation

class Mish(CustomActivation):
    def __init__(self, dim, modelo_gen):
        super(Mish, self).__init__(dim, modelo_gen, dominio_0_1=False)

    def _default_weights_initialization(self):
        pass

    def forward(self, x):
        result = x * torch.tanh(torch.log(1 + torch.exp(x)))
        return result

    def get_parameters_names(self):
        return []
    
    def get_parameter_statistics(self, name):
        raise Exception("This function does not have adptive parameters, this method should not be called.")