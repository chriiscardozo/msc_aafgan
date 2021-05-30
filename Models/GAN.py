# import Parzen as P
from Models.Activation.BHAA import BHAA
from Models.Activation.BHANA import BHANA
from Models.Activation.BHSA import BHSA
from Models.Activation.SHReLU import SHReLU
from Models.Activation.dSiLU import dSiLU
from Models.Reshape import Reshape
from Models.Activation.MiDA_old import MiDA_old
from Models.Activation.MiDA import MiDA
from Models.Activation.Mish import Mish
from Utils import cuda_utils, commons_utils
import torch
import os, time
from pre_trained_classifier import classifier_accuracy
from torch import nn
from Models.FID import FID
from Models.Parzen import Parzen as P

class Net(torch.nn.Module):
    
    def __init__(self, config, activation_option, hidden_model, is_conditional, is_discriminator, n_classes=10):
        super(Net, self).__init__()

        self.config = config
        self._n_in = config["n_in"]
        self._next_n_in = config["n_in"]
        self._n_out = config["n_out"]
        self.activation_option = activation_option
        self.is_discriminator = is_discriminator
        self.hidden_model = hidden_model
        self.features_layers = None
        self.labels_layers = None
        self.mlp_layers = None
        self.layers = [] # all layers
        self.hidden_with_adaptive_layers = []

        if is_conditional:
            self.n_classes = n_classes
            # Building the labels layers if is conditional GAN
            if "labels_layers" in config:
                self._next_n_in = config["CONDITIONAL_LABELS"]
                self.labels_layers = []
                for layer_config in config["labels_layers"]:
                    layer = self._extract_layer(layer_config)
                    self.labels_layers.append(layer)
                    self.layers.append(layer)
                self.labels_layers = torch.nn.Sequential(*self.labels_layers)
            
        # Building the features (image) layers if is conditional GAN
        if "features_layers" in config:
            self.features_layers = []
            for layer_config in config["features_layers"]:
                layer = self._extract_layer(layer_config)
                self.features_layers.append(layer)
                self.layers.append(layer)
            self.features_layers = torch.nn.Sequential(*self.features_layers)

        # Building a regular layers seq for a regular GAN or the concatenation X_Y result for a conditional GAN
        if "layers" in config:
            self.mlp_layers = []
            for layer_config in config["layers"]:
                layer = self._extract_layer(layer_config)
                self.mlp_layers.append(layer)
                self.layers.append(layer)
            self.mlp_layers = torch.nn.Sequential(*self.mlp_layers)

        if "INIT_NET_WEIGHTS" in config:
            print("Initialising especific weights for network: " + str(config["INIT_NET_WEIGHTS"]))
            self.apply(self.default_batchnorm_init)
            if config["INIT_NET_WEIGHTS"]["name"] == "default":
                self.apply(self.default_weights_init)
            elif config["INIT_NET_WEIGHTS"]["name"] == "glorot":
                self.apply(self.glorot_normal_weights_init)
            elif config["INIT_NET_WEIGHTS"]["name"] == "he_normal":
                self.apply(self.he_normal_weights_init)
            else:
                raise "Net W initialization not implemented yet: " + config["INIT_NET_WEIGHTS"]

    def _extract_layer(self, layer_config):
        layer = None

        # Fully Connected layers
        if layer_config["type"] == "Linear":
            if "in" in layer_config: self._next_n_in = layer_config["in"]
            out = self._n_out if "flag" in layer_config and layer_config["flag"] == "END" else layer_config["out"]
            layer = torch.nn.Linear(self._next_n_in, out)
            self._next_n_in = out
        # Convolutional Layers
        if layer_config["type"] == "Conv2d":
            add_bias = "bias" in layer_config and int(layer_config["bias"]) == 1
            layer = torch.nn.Conv2d(layer_config["in_channels"], layer_config["out_channels"], layer_config["kernel_size"], layer_config["stride"], layer_config["padding"], bias=add_bias)
            if "out" in layer_config: self._next_n_in = layer_config["out"]
        if layer_config["type"] == "Deconv2d":
            add_bias = "bias" in layer_config and int(layer_config["bias"]) == 1
            layer = torch.nn.ConvTranspose2d(layer_config["in_channels"], layer_config["out_channels"], layer_config["kernel_size"], layer_config["stride"], layer_config["padding"], bias=add_bias)
            if "out" in layer_config: self._next_n_in = layer_config["out"]
        if layer_config["type"] == "relu":
            layer = torch.nn.ReLU()
        # ReLU, LeakyReLU, SHReLU layers
        if layer_config["type"] == "custom_relu":
            layer = torch.nn.LeakyReLU(layer_config["LRelu_v"]) if "LRelu_v" in layer_config else torch.nn.ReLU()
            if self.hidden_model == "MiDA_old":
                layer = MiDA_old(self._next_n_in, modelo_gen=(not self.is_discriminator), init_strategy=self.config["MIDA_INITIALIZER"])
                self.hidden_with_adaptive_layers.append(layer)
            if self.hidden_model == "SHReLU":
                layer = SHReLU(self._next_n_in, modelo_gen=(not self.is_discriminator), init_strategy=self.config["SHRELU_INITIALIZER"])
                self.hidden_with_adaptive_layers.append(layer)
            if self.hidden_model == "Mish":
                layer = Mish(self._next_n_in, modelo_gen=(not self.is_discriminator))
            if self.hidden_model == "MiDA":
                layer = MiDA(self._next_n_in, modelo_gen=(not self.is_discriminator), init_strategy=self.config["MIDA_INITIALIZER"])
                self.hidden_with_adaptive_layers.append(layer)
        if layer_config["type"] == "Reshape":
            layer = Reshape(layer_config["dims"])
        # Dropout layers
        if layer_config["type"] == "Dropout":
            layer = torch.nn.Dropout(layer_config["v"])
        # Batch Normalisation layers
        if layer_config["type"] == "BatchNorm":
            layer = torch.nn.BatchNorm2d(layer_config["v"])
        # Last Activation layers: sigmoid, tanh, BHAA, BHSA, dSiLU
        if layer_config["type"] == "last_activation":
            layer = torch.nn.Sigmoid() if self.is_discriminator else torch.nn.Tanh()
            if(self.activation_option == 'BHAA'):
                layer = BHAA(self._n_out,modelo_gen=(not self.is_discriminator),dominio_0_1=self.is_discriminator,init_strategy=self.config["BHAA_INITIALIZER"],truncated=False)
            elif(self.activation_option == 'BHATA'):
                layer = BHAA(self._n_out,modelo_gen=(not self.is_discriminator),dominio_0_1=self.is_discriminator,init_strategy=self.config["BHAA_INITIALIZER"],truncated=True)
            elif(self.activation_option == 'BHSA'): 
                layer = BHSA(self._n_out,modelo_gen=(not self.is_discriminator),dominio_0_1=self.is_discriminator,init_strategy=self.config["BHSA_INITIALIZER"])
            elif(self.activation_option == 'BHANA'):
                layer = BHANA(self._n_out,modelo_gen=(not self.is_discriminator),dominio_0_1=self.is_discriminator,init_strategy=self.config["BHANA_INITIALIZER"])
            elif(self.activation_option == 'DSILU'):
                layer = dSiLU(self._n_out,modelo_gen=(not self.is_discriminator),dominio_0_1=self.is_discriminator,init_strategy=self.config["DSILU_INITIALIZER"])
        return layer

    def get_last_activation(self):
        return self.layers[-1]

    def forward(self, x, y=None):
        if self.labels_layers is not None:
            if y is None: raise 'Missing labels (y) parameter'
            y = y.view(-1, self.n_classes)
            for l in self.labels_layers: y = l(y)
        if self.features_layers is not None:
            # old non-parallel impl
            # for l in self.features_layers:
            #     x = l(x)
            if cuda_utils.N_GPUS > 1: x = nn.parallel.data_parallel(self.features_layers, x, range(cuda_utils.N_GPUS))
            else: x = self.features_layers(x)
            if "n_out_features_layers" in self.config: x = x.view(-1, self.config["n_out_features_layers"])
        
        if y is not None: x = torch.cat([x, y], 1)

        if self.mlp_layers is not None:
            # old non-parallel impl
            # for l in self.mlp_layers:
            #     x = l(x)
            if cuda_utils.N_GPUS > 1: x = nn.parallel.data_parallel(self.mlp_layers, x, range(cuda_utils.N_GPUS))
            else: x = self.mlp_layers(x)

        return x

    # custom weights initialization called on netG and netD
    # "default" is based on DCGAN paper
    def default_batchnorm_init(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def default_weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

    def glorot_normal_weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_normal_(m.weight.data)

    def he_normal_weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)

class DNet(Net):
    
    def __init__(self, config, activation_option='default', hidden_model='default', is_conditional=False):
        super(DNet, self).__init__(config, activation_option, hidden_model, is_conditional, is_discriminator=True)

    def train_per_batch(self, N, X, Y, loss, conditional_labels):
        prediction = None
        if conditional_labels is None: prediction = self.forward(X)
        else: prediction = self.forward(X, conditional_labels)

        error = loss(prediction, Y)
        error.backward()
        accuracy = (prediction > 0.5).eq(Y > 0.5).sum().item()/N
        return error, accuracy

    def train(self, optim, loss, real_data, fake_data, real_y=None, fake_y=None):
        N = real_data.size(0)
        self.zero_grad()

        if (real_y is None and fake_y is not None) or (real_y is not None and fake_y is None):
            raise Exception("real_y and fake_y need to be None or not None (both)")

        error_real, accuracy_real = self.train_per_batch(N, real_data, cuda_utils.ones_target(N), loss, real_y)
        error_fake, accuracy_fake = self.train_per_batch(N, fake_data, cuda_utils.zeros_target(N), loss, fake_y)

        optim.step()

        return (error_real+error_fake), (accuracy_real+accuracy_fake)/2


class GNet(Net):
    
    def __init__(self, config, activation_option='default', hidden_model='default', is_conditional=False):
        super(GNet, self).__init__(config, activation_option, hidden_model, is_conditional, is_discriminator=False)

    def train(self, D, optim, loss, fake_data, fake_labels=None):
        N = fake_data.size(0)
        optim.zero_grad()

        prediction = D(fake_data, fake_labels)
        error = loss(prediction, cuda_utils.ones_target(N))
        error.backward()
        optim.step()

        return error


class GAN():
    def __init__(self, d_model, g_model, d_hidden_model, g_hidden_model, output_dir, config):
        self.output_dir = output_dir
        self.config = config
        self.is_conditional = "CONDITIONAL_LABELS" in config
        self.channels = config["CHANNELS"] if "CHANNELS" in config else 1
        self.generate_n_csv = config["GENERATE_N_IN_CSV"] if "GENERATE_N_IN_CSV" in config else 1000
        self.fid_enabled = "FID_ENABLED" in config and config["FID_ENABLED"] == 1
        self.enable_parzen_ll_metric = "ENABLE_PARZEN_LL_METRIC" in config and config["ENABLE_PARZEN_LL_METRIC"] == 1
        self.limit_samples_csv_generation = config["LIMIT_SAMPLES_CSV_GENERATION"] if "LIMIT_SAMPLES_CSV_GENERATION" in config else 1000
        self.ignore_conditional_acc_calculation = "IGNORE_CONDITIONAL_ACCURACY_CALCULATION" in config and config["IGNORE_CONDITIONAL_ACCURACY_CALCULATION"] == 1
        self.ignore_sample_generation = "IGNORE_SAMPLE_GENERATION" in config and config["IGNORE_SAMPLE_GENERATION"] == 1

        self.D = DNet(config=config["DISCRIMINATOR"], activation_option=d_model, hidden_model=d_hidden_model, is_conditional=self.is_conditional)
        self.G = GNet(config=config["GENERATOR"], activation_option=g_model, hidden_model=g_hidden_model, is_conditional=self.is_conditional)

        if cuda_utils.N_GPUS > 1:
            print("-> Data parallelism in GPU is on! (" + str(cuda_utils.N_GPUS) + " gpus are available)")

        self.D.to(cuda_utils.DEVICE)
        self.G.to(cuda_utils.DEVICE)

        if "INIT_NET_WEIGHTS" in config and config["INIT_NET_WEIGHTS"] == 1:
            self.D.apply(self.weights_init)
            self.G.apply(self.weights_init)

        self.d_optim = self.init_optimizer(config["OPTIMIZER"], self.D.parameters(), lr=config["LEARNING_RATE"], optional_params=config["OPTIMIZER_OPTIONAL_PARAMS"] if "OPTIMIZER_OPTIONAL_PARAMS" in config else None)
        self.g_optim = self.init_optimizer(config["OPTIMIZER"], self.G.parameters(), lr=config["LEARNING_RATE"], optional_params=config["OPTIMIZER_OPTIONAL_PARAMS"] if "OPTIMIZER_OPTIONAL_PARAMS" in config else None)

        if "LEARNING_RATE_DECAY" in config:
            self.lr_scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.d_optim, gamma=config["LEARNING_RATE_DECAY"]["DECAY_RATE"])
            self.lr_scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.g_optim, gamma=config["LEARNING_RATE_DECAY"]["DECAY_RATE"])

            if "LAST_EPOCH_DECAY" in config["LEARNING_RATE_DECAY"]: self.last_epoch_decay = config["LEARNING_RATE_DECAY"]["LAST_EPOCH_DECAY"]
            else: self.last_epoch_decay = -1
        else:
            self.lr_scheduler_dis = None
            self.lr_scheduler_gen = None

        self.loss = torch.nn.BCELoss()

        if self.is_conditional:
            self.classifier_model = classifier_accuracy.load_pre_trained_model(config["DATASET"])


    # custom weights initialization called on netG and netD (this was for cifar10 dataset specially)
    # based on DCGAN paper
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    def init_optimizer(self, strategy, parameters, lr, optional_params=None):
        if strategy == "adam":
            if optional_params is not None:
                print("Using custom params in the optimizer:", optional_params)
                # 0.999 is not random, it is the default in the pytorch library and usually not changed by models in the literature
                betas = (optional_params["beta1"], 0.999)
                return torch.optim.Adam(parameters, lr=lr, betas=betas)
            else:
                return torch.optim.Adam(parameters, lr=lr)
        elif strategy == "sgd":
            return torch.optim.SGD(parameters, lr=lr)
        else:
            raise 'Strategy not implemented yet: ' + strategy


    def train(self, data_loader, test_data, epochs, verbose_batch, verbose_epoch, visualization_noise, visualization_noise_labels):
        print("Training output to", self.output_dir)

        commons_utils.reset_dir(os.path.join(self.output_dir, commons_utils.DIR_SAMPLES_CSV))
        commons_utils.reset_dir(os.path.join(self.output_dir, commons_utils.DIR_SAMPLES_IMG))
        commons_utils.reset_dir(os.path.join(self.output_dir, commons_utils.DIR_STATS))
        commons_utils.reset_dir(os.path.join(self.output_dir, commons_utils.DIR_GRAPHICS))
        commons_utils.reset_dir(os.path.join(self.output_dir, cuda_utils.DIR_MODEL_CHECKPOINT))

        fid_scores = []
        x_axis = []
        d_losses = []
        g_losses = []
        d_accuracies = []
        times = []
        lls_avg = []
        lls_std = []
        classifier_accuracies = []
        activation_parameters = { "Gen": {}, "Dis": {} }

        if self.config["DOUBLE_TENSORS"]: test_data = test_data.double()

        if self.fid_enabled:
            fid = FID(test_data, self.config["DOUBLE_TENSORS"]) if self.fid_enabled else None
        else:
            fid = None

        self.verbose_start_time = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            d_error = g_error = d_accuracy = None

            for i_batch, (real_batch_x, real_batch_labels) in enumerate(data_loader):
                g_error, d_error, d_accuracy = self._execute_batch(real_batch_x, real_batch_labels, epoch, i_batch, verbose_batch)

            if self.lr_scheduler_dis is not None:
                if epoch < self.last_epoch_decay or self.last_epoch_decay == -1:
                    self.lr_scheduler_dis.step()
                    self.lr_scheduler_gen.step()

            if "FREEZE_CUSTOM_PARAMETERS_EPOCH" in self.config and self.config["FREEZE_CUSTOM_PARAMETERS_EPOCH"] == epoch:
                print("freezing the custom parameters")
                self.freeze_custom_activations(self.D)
                self.freeze_custom_activations(self.G)

            if self.ignore_sample_generation:
                commons_utils.generate_visualization_samples(epoch, visualization_noise, visualization_noise_labels, self.G, self.output_dir, self.channels, vector_dim=self.config["IMG_SIZE"])
                generatedImages, labels = commons_utils.generate_epoch_samples(self.config, epoch, self.G, self.output_dir, self.config["NOISE_SIZE"], self.channels, is_conditional=self.is_conditional, n_csv=self.generate_n_csv, N_CSV_LIMIT=self.limit_samples_csv_generation)

            # saving only last epoch state to save on disk size
            if epoch == epochs-1:
                cuda_utils.model_checkpoint(epoch, self.config, self.G, self.D, self.d_optim, self.g_optim, self.output_dir)

            # if(epoch % verbose_epoch == 0 or epoch == epochs-1):
            x_axis.append(epoch)
            d_losses.append(d_error)
            g_losses.append(g_error)
            d_accuracies.append(d_accuracy)

            if self.is_conditional and not self.ignore_conditional_acc_calculation:
                labels = labels.argmax(dim=1, keepdim=True)
                epoch_classifier_accuracy = commons_utils.calculate_classifier_accuracy(generatedImages, labels, self.classifier_model, self.config["DATASET"])
                classifier_accuracies.append(epoch_classifier_accuracy)

            self.register_activation_parameters(self.D, "Dis", activation_parameters)
            self.register_activation_parameters(self.G, "Gen", activation_parameters)

            # gpu parzen (not implemented)
            #ll_avg, ll_std = P.log_prob(test_data, generated_samples, gpu=(Util.DEVICE!='cpu'))
            #cpu parzen
            if self.enable_parzen_ll_metric:
                ll_avg, ll_std = P.log_prob(test_data, generatedImages)
                lls_avg.append(ll_avg)
                lls_std.append(ll_std)
                print("Epoch negative log-likelihood:",ll_avg, "("+"{:.2f}".format(time.time()-epoch_start)+" s)")

            if self.fid_enabled:
                fid_score = fid.calculate_fid(self.output_dir, epoch)
                fid_scores.append(fid_score)
                print("\nFID score for epoch %d: %.2f\n" % (epoch, fid_score))

            final_epoch_time = time.time() - epoch_start
            times.append(final_epoch_time)
            print("Epoch %d time is: %d" % (epoch, int(final_epoch_time)))


        general_info_dict = {   "x": x_axis, "times": times, "d_losses": d_losses, "g_losses": g_losses, 
                                "d_accuracies": d_accuracies, "lls_avg": lls_avg, "lls_std": lls_std, "pre_classifier_accuracies": classifier_accuracies,
                                "fid_scores": fid_scores
                            }
        for key_net in activation_parameters.keys():
            for key_param in activation_parameters[key_net].keys():
                general_info_dict[key_net + "_" + key_param] = activation_parameters[key_net][key_param]

        commons_utils.save_general_information(general_info_dict, self.output_dir)
        commons_utils.generate_graphics(x_axis,d_losses,g_losses,d_accuracies,lls_avg,lls_std,activation_parameters,self.output_dir)
        commons_utils.save_summary_model(self.output_dir, self.G, self.D, self.channels, noise_dim=self.config["NOISE_SIZE"], img_size=self.config["IMG_SIZE"], is_conditional=self.is_conditional)


    def _execute_batch(self, real_batch_x, real_batch_labels, epoch, i_batch, verbose_batch):
        N = real_batch_x.size(0)

        real_batch_x = real_batch_x.to(cuda_utils.DEVICE)
        real_batch_labels = real_batch_labels.to(cuda_utils.DEVICE)

        # Train D
        real_data_x = cuda_utils.images_to_vectors(real_batch_x, self.config["IMG_SIZE"]**2) if self.config["DATASET"] == 'MNIST' else real_batch_x
        real_data_labels = cuda_utils.one_hot_vector(real_batch_labels) if self.is_conditional else None
        if(self.config["DOUBLE_TENSORS"]):
            real_data_x = real_data_x.double()
            if real_data_labels is not None: real_data_labels = real_data_labels.double()

        noise_x = cuda_utils.noise(N, self.config["NOISE_SIZE"], self.channels)
        fake_data_labels = cuda_utils.fake_labels(N, self.config["DOUBLE_TENSORS"]) if self.is_conditional else None
        fake_data_x = self.G(noise_x, fake_data_labels).detach()

        d_error, d_accuracy = self.D.train(self.d_optim,self.loss,real_data_x,fake_data_x, real_data_labels, fake_data_labels)

        # Train G
        noise_x = cuda_utils.noise(N, self.config["NOISE_SIZE"], self.channels)
        fake_data_labels = cuda_utils.fake_labels(N, self.config["DOUBLE_TENSORS"]) if self.is_conditional else None
        fake_data_x = self.G(noise_x, fake_data_labels)

        g_error = self.G.train(self.D,self.g_optim,self.loss,fake_data_x,fake_data_labels)

        if(i_batch % verbose_batch == 0):
            duration = time.time() - self.verbose_start_time
            self.verbose_start_time = time.time()
            print("[%d, %d] D_loss: %.4f / G_loss: %.4f / D_accu: %.2f%% / Exec time: %.2f s" % (epoch, i_batch, d_error, g_error, d_accuracy*100.0, duration))

        return g_error.item(), d_error.item(), d_accuracy


    def register_activation_parameters(self, model, label, activation_parameters):
        for index, layer in enumerate(model.layers):
            if "has_parameters" in dir(layer) and layer.has_parameters():
                trainable_parameters_names = layer.get_parameters_names()
                for name in trainable_parameters_names:
                    dict_name_key = name + "_" + str(index)
                    values = layer.get_parameter_statistics(name)
                    if dict_name_key not in activation_parameters[label]: activation_parameters[label][dict_name_key] = []
                    activation_parameters[label][dict_name_key].append(values)

    
    def freeze_custom_activations(self, model):
        if(model.activation_option != "default"):
            model.get_last_activation().freeze_parameters()
        if(model.hidden_model != "default"):
            for layer in model.hidden_with_adaptive_layers:
                layer.freeze_parameters()
