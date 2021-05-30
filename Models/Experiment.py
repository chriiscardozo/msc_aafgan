from Models.GAN import GAN

import Utils.cuda_utils as cuda_utils
import Utils.commons_utils as commons_utils
import Utils.config_utils as config_utils
import Utils.amazon_utils as amazon_utils
import Utils.dataset_utils as dataset_utils
import Utils.notification_utils as notification_utils

import os
import time

class Experiment:
    def __init__(self, config_file_path, experiment_key="current", dataset=None):
        self._start = commons_utils.get_time()
        self._config_file_path = config_file_path

        config = config_utils.load_config(config_file_path, experiment_key)
        if "DATASET" not in config: config["DATASET"] = dataset
        config_utils.prepare_experiment_dir(config)
        cuda_utils.configure_dtype(config)

        self._is_conditional = "CONDITIONAL_LABELS" in config
        self.channels = config["CHANNELS"] if "CHANNELS" in config else 1

        self._visualization_noise = cuda_utils.noise(config["SAMPLES_IN_VISUALIZATION"], config["NOISE_SIZE"], self.channels)
        self._visualization_noise_labels = None
        if self._is_conditional: self._visualization_noise_labels = cuda_utils.fake_labels_balanced_ordered( config["SAMPLES_IN_VISUALIZATION"], 
                                                                                        config["CONDITIONAL_LABELS"] if self._is_conditional else 10, 
                                                                                        config["DOUBLE_TENSORS"])
        print("Loading the dataset:", config["DATASET"])
        train_data, test_data = dataset_utils.get_train_test_data(config, config["DATASET"])
        batches = len(train_data)

        print("Running", config["EPOCHS"], "epochs having", batches, "batches each one / The batch size is", config["BATCH_SIZE"])
        print("Device available: ", cuda_utils.DEVICE)
        if cuda_utils.DEVICE == 'cpu':
            raise Exception("Expecting gpu to run the experiments. Edit this if cpu is now valid option")

        model_key = None
        aws_tag = amazon_utils.get_aws_tag("K")
        if aws_tag is not None: model_key = aws_tag[0]

        if model_key is None:
            self._dis_models = amazon_utils.get_aws_tag("DIS_MODELS") or config["DIS_MODELS"]
            self._gen_models = amazon_utils.get_aws_tag("GEN_MODELS") or config["GEN_MODELS"]
            self._d_hidden_model = amazon_utils.get_aws_tag("DIS_HIDDEN_MODELS") or config["DIS_HIDDEN_MODELS"]
            self._g_hidden_model = amazon_utils.get_aws_tag("GEN_HIDDEN_MODELS") or config["GEN_HIDDEN_MODELS"]
        else:
            model_vars = dataset_utils.get_model_mapping_value(model_key)
            self._dis_models = [model_vars['d_model']]
            self._gen_models = [model_vars['g_model']]
            self._d_hidden_models = [model_vars['d_hidden_model']]
            self._g_hidden_models = [model_vars['g_hidden_model']]

        self._config = config
        self._train_data = train_data
        self._test_data = test_data

    def override_config(self, key, value):
        if key not in self._config: raise Exception("Failure on overriding Experiment config: the key '" + key + "' does not exist in config json.")
        print("Overriding config: config['" + key + "'] from '" + str(self._config[key]) + "' to '" + str(value) + "'")
        self._config[key] = value
    
    def add_custom_config(self, key, value):
        if key in self._config: raise Exception("Failure on adding custom config: the key '" + key + "' already exists in config json.")
        print("Adding custom config: config['" + key + "'] with value '" + str(value) + "'")
        self._config[key] = value

    def run(self):
        for d_model in self._dis_models:
            for g_model in self._gen_models:
                for d_hidden_model in self._d_hidden_model:
                    for g_hidden_model in self._g_hidden_model:
                        if (g_model != "default" or d_model != "default") and (d_hidden_model != "default" or g_hidden_model != "default"): continue
                        if "SKIP_DEFAULT" in self._config and d_model == "default" and g_model == "default" and d_hidden_model == "default" and g_hidden_model == "default":
                            print("default (output) configured to be skiped. Skipping it...")
                            continue
                        if "SKIP_DIFF_HIDDEN" in self._config and d_hidden_model != g_hidden_model and d_hidden_model != "default" and g_hidden_model != "default":
                            continue
                        if "SKIP_DIFF_OUTPUT" in self._config and d_model != g_model and d_model != "default" and g_model != "default":
                            continue
                        gan = self._run_model(d_model, g_model, d_hidden_model, g_hidden_model)

    def _run_model(self, d_model, g_model, d_hidden_model, g_hidden_model):
        output_dir = commons_utils.build_output_dir_path(self._config, d_model, g_model, d_hidden_model, g_hidden_model)
        if commons_utils.model_completed(output_dir): return

        for i, seed in enumerate(self._config["SEEDS"]):
            if "N_SEEDS" in self._config and i > self._config["N_SEEDS"] - 1: break

            output_dir_seed = os.path.join(output_dir, str(i))
            is_second_try_double = commons_utils.model_marked_as_double_error(output_dir_seed)

            if commons_utils.model_completed(output_dir_seed): continue
            if commons_utils.model_marked_as_error(output_dir_seed) and self._config["DOUBLE_TENSORS"] == 0: continue

            commons_utils.reset_dir(output_dir_seed)
            
            # Try a different seed for second double attempt (do not set the default one)
            if commons_utils.model_marked_as_double_error(output_dir_seed):
                seed = int(time.time())

            commons_utils.set_seed_as(seed)    

            try:
                gan = GAN(d_model=d_model, g_model=g_model, d_hidden_model=d_hidden_model, g_hidden_model=g_hidden_model, output_dir=output_dir_seed, config=self._config)
                gan.train(data_loader=self._train_data, test_data=self._test_data, epochs=self._config["EPOCHS"], verbose_batch=self._config["VERBOSE_BATCH_STEP"], 
                        verbose_epoch=self._config["VERBOSE_EPOCH_STEP"], visualization_noise=self._visualization_noise, 
                        visualization_noise_labels=self._visualization_noise_labels)

                commons_utils.mark_model_as_completed(output_dir_seed, self._config)
                amazon_utils.send_to_s3(self._config, output_dir_seed)
            except Exception as ex:
                if self._config["DOUBLE_TENSORS"] == 1:
                    print("Exception occurred even using DOUBLE_TENSORS. This is not good!")
                    # For Double, we try two times (second time with different seed)
                    # If for the second time it doesn't work, we just give up and mark it as completed
                    if is_second_try_double:
                        notification_utils.send_message("2x double failed, skipping: %s" % (output_dir_seed))
                        commons_utils.mark_model_as_completed(output_dir_seed, {"exception_error": ex})
                    else:
                        commons_utils.mark_model_as_double_cuda_error(output_dir_seed, ex)
                elif "CUDA error: device-side assert triggered" in str(ex):
                    print("Exception occurred, exiting and skipping seed (index=%d) %d" % (i, seed))
                    commons_utils.mark_model_as_cuda_error(output_dir_seed, ex)
                else:
                    commons_utils.mark_model_as_cuda_error(output_dir_seed, ex)
                    notification_utils.send_message("unexpected exception on: %s" % (output_dir_seed))
                    print("Unexpected exception occurred: " + str(ex))
                exit(0)

        notification_utils.send_message("Model marked as completed: " + output_dir)
        commons_utils.mark_model_as_completed(output_dir, self._config)

    def finish(self):
        commons_utils.exec_time(self._start, "Running")
        amazon_utils.mark_spot_request_as_cancelled()
        amazon_utils.shutdown_if_ec2()