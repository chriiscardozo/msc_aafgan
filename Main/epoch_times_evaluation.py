from Models.Experiment import Experiment

CONFIG_DCGAN = {
    "config_file": "celeba_dcgan.json",
    "model_key": "celeba_dcgan",
    "dataset": "CelebA",
    "overrides": {
        "FID_ENABLED": 0
    }
}

CONFIG_CONDGAN = {
    "config_file": "mnist_condgan.json",
    "model_key": "condgan_mlp",
    "dataset": "MNIST"
}

def evaluate(experiment_setup):
    print("Evaluating " + experiment_setup["config_file"] + ": " + experiment_setup["model_key"])
    e = Experiment(experiment_setup['config_file'], experiment_setup['model_key'], dataset=experiment_setup['dataset'])
    e.override_config("EPOCHS", 3)
    e.override_config("N_SEEDS", 3)
    e.override_config("EXPERIMENT_DIR", "time_evaluation_experiment_" + experiment_setup['model_key'])
    e.add_custom_config("IGNORE_CONDITIONAL_ACCURACY_CALCULATION", 1)
    e.add_custom_config("IGNORE_SAMPLE_GENERATION", 1)

    if "overrides" in experiment_setup:
        for key in experiment_setup["overrides"].keys():
            e.override_config(key, experiment_setup["overrides"][key])

    e.run()
    e.finish()

def main():
    for item in [CONFIG_CONDGAN, CONFIG_DCGAN]:
        evaluate(item)


if __name__ == "__main__":
    main()