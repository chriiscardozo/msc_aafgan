import torch
from torch import nn
from Utils import cuda_utils
from Models.Activation.CustomActivation import CustomActivation

class BHANA(CustomActivation):
    def __init__(self, dim,modelo_gen, dominio_0_1=True, dcgan=False, init_strategy={'name': "default"}, preferred_impl="math"):
        super(BHANA, self).__init__(dim, modelo_gen, dominio_0_1)

        self.l = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))
        self.t1 = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))
        self.t2 = nn.Parameter(torch.empty(self.dim, device=cuda_utils.DEVICE, dtype=cuda_utils.DTYPE))

        self.preferred_impl = preferred_impl

        if init_strategy['name'] == "default":
            self._default_weights_initialization()
        else:
            self._initialize_weights(init_strategy, self.t1)
            self._initialize_weights(init_strategy, self.t2)
            self._initialize_weights(init_strategy, self.l)

    def _default_weights_initialization(self):
        # for generator
        t1_mean = t2_mean = 0.25
        t1_std = t2_std = 0.05
        l_mean = 0.5
        l_std = 0.05

        # for discriminator 
        if(self.dominio_0_1):
            t1_mean = t2_mean = 0.7
            t1_std = t2_std = 0.05
            l_mean = 0.5
            l_std = 0.05

        nn.init.normal_(self.t1, mean=t1_mean, std=t1_std)
        nn.init.normal_(self.t2, mean=t2_mean, std=t2_std)
        nn.init.normal_(self.l, mean=l_mean, std=l_std)

    def forward(self, x):
        result = self.evaluate_raw_bhaa(x)

        # the latest normalized attempt using raw math formulas
        if self.preferred_impl == "math":
            if self.dominio_0_1:
                derivative_points = torch.stack([(self.t1 - self.t2)/(4*self.l*(self.t1 + self.t2)), (self.t1 + self.t2)/(4*self.l*(self.t1 - self.t2))]).view(2,-1).transpose(0,1)
                maximum_function_value = self.evaluate_raw_bhaa(torch.max(derivative_points, dim=1)[0]).flatten() # [0] to get values; [1] to get indexes
                minimum_function_value = self.evaluate_raw_bhaa(torch.min(derivative_points, dim=1)[0]).flatten() # [0] to get values; [1] to get indexes

                class_case = ((torch.abs(self.t1-self.t2)/(self.t1-self.t2))/2) + 0.5 # 0 or 1 (known bug if t1 = t2 or t1-t2 too small)
                class_case = torch.round(class_case).type(torch.int8)

                minimum_case = ((-1)**class_case)*(minimum_function_value**(1-class_case))
                maximum_case = (maximum_function_value**class_case)*(1**(1-class_case))

                result = (result - minimum_case)/(maximum_case + 0.01 - minimum_case)
            else:
                derivative_points = torch.stack([(self.t1 - self.t2)/(2*self.l*(self.t1 + self.t2)), (self.t1 + self.t2)/(2*self.l*(self.t1 - self.t2))]).view(2,-1).transpose(0,1)
                maximum_function_value = self.evaluate_raw_bhaa(torch.max(derivative_points, dim=1)[0]).flatten() # [0] to get values; [1] to get indexes
                minimum_function_value = self.evaluate_raw_bhaa(torch.min(derivative_points, dim=1)[0]).flatten() # [0] to get values; [1] to get indexes

                class_case = ((torch.abs(self.t1-self.t2)/(self.t1-self.t2))/2) + 0.5 # 0 or 1 (known bug if t1 = t2 or t1-t2 too small)
                class_case = torch.round(class_case).type(torch.int8)

                minimum_case = ((-1)**class_case)*(minimum_function_value**(1-class_case))
                maximum_case = (maximum_function_value**class_case)*(1**(1-class_case))

                result = (result - minimum_case)/(maximum_case + 0.01 - minimum_case)
                result = (result - 0.5)*2

        # Normalized version using conditionals (ifs) instead of raw math formulas
        elif self.preferred_impl == "if":
            if self.dominio_0_1:
                interval_low_limit = 0 
                interval_high_limit = 1

                derivative_points = torch.stack([(self.t1 - self.t2)/(4*self.l*(self.t1 + self.t2)), (self.t1 + self.t2)/(4*self.l*(self.t1 - self.t2))]).view(2,-1).transpose(0,1)
                maximum_function_value = self.evaluate_raw_bhaa(torch.max(derivative_points, dim=1)[0]).flatten() # [0] to get values; [1] to get indexes
                minimum_function_value = self.evaluate_raw_bhaa(torch.min(derivative_points, dim=1)[0]).flatten() # [0] to get values; [1] to get indexes

                mask_max = torch.logical_and(self.t1 > self.t2, (self.t1 - self.t2).abs() > 0.00001).reshape(-1)
                mask_min = torch.logical_and(self.t2 > self.t1, (self.t1 - self.t2).abs() > 0.00001).reshape(-1)

                if mask_max.any():
                    result[:, mask_max] = ((result - interval_low_limit) / (maximum_function_value - interval_low_limit))[:, mask_max]
                if mask_min.any():
                    result[:, mask_min] = ((result - minimum_function_value) / (interval_high_limit - minimum_function_value))[:, mask_min]
            else:
                interval_low_limit = -1
                interval_high_limit = 1

                derivative_points = torch.stack([(self.t1 - self.t2)/(2*self.l*(self.t1 + self.t2)), (self.t1 + self.t2)/(2*self.l*(self.t1 - self.t2))]).view(2,-1).transpose(0,1)

                maximum_function_value = self.evaluate_raw_bhaa(torch.max(derivative_points, dim=1)[0]).flatten() # [0] to get values; [1] to get indexes
                minimum_function_value = self.evaluate_raw_bhaa(torch.min(derivative_points, dim=1)[0]).flatten() # [0] to get values; [1] to get indexes

                mask_max = torch.logical_and(self.t1 > self.t2, (self.t1 - self.t2).abs() > 0.00001).reshape(-1)
                mask_min = torch.logical_and(self.t2 > self.t1, (self.t1 - self.t2).abs() > 0.00001).reshape(-1)

                if mask_max.any():
                    result[:, mask_max] = ((result - interval_low_limit) / (maximum_function_value - interval_low_limit))[:, mask_max]
                if mask_min.any():
                    result[:, mask_min] = ((result - minimum_function_value) / (interval_high_limit - minimum_function_value))[:, mask_min]

                result = (result - 0.5)*2
        else:
            raise self.preferred_impl + ' is not a valid implementation id. Options are: [math, if]'
        ###
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
        return ['bhana_tau1', 'bhana_tau2', 'bhana_lambda']
    
    def get_parameter_statistics(self, name):
        if name == 'bhana_tau1':
            return self.t1.mean().item(), self.t1.max().item(), self.t1.min().item()
        elif name == 'bhana_tau2':
            return self.t2.mean().item(), self.t2.max().item(), self.t2.min().item()
        elif name == 'bhana_lambda':
            return self.l.mean().item(), self.l.max().item(), self.l.min().item()
        else:
            raise Exception("The name '" + name + "' is not a valid trainable parameter name")
