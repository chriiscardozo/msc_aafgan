from torch import nn

class CustomActivation(nn.Module):
    def __init__(self, dim, modelo_gen, dominio_0_1):
        super(CustomActivation, self).__init__()

        self.modelo_gen = modelo_gen
        self.dominio_0_1 = dominio_0_1

        if(not isinstance(dim, list) and not isinstance(dim, tuple)): dim = [1, dim]
        self.dim = dim
    
    def _initialize_weights(self, strategy, tensor, param_name = None):
        if strategy['name'] == "xavier_normal":
            nn.init.xavier_normal_(tensor)
        elif strategy['name'] == "normal":
            if param_name in strategy:
                nn.init.normal_(tensor, mean=strategy[param_name]['mean'], std=strategy[param_name]['std'])
            else:
                nn.init.normal_(tensor, mean=strategy['mean'], std=strategy['std'])
        else:
            raise 'weight initializer strategy not implemented: ' + strategy
    
    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def has_parameters(self):
        return True
    
    def get_parameters_names(self):
        raise Exception("Missing child impl")
    
    def get_parameter_statistics(self, name):
        raise Exception("Missing child impl")