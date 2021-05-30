import json
import os
from Utils import commons_utils

CONFIG_FODLER = "Config"

def load_config(filename, experiment_key="current"):
    with open(os.path.join(CONFIG_FODLER, filename)) as f:
        return json.load(f)[experiment_key]

def prepare_experiment_dir(config):
    if(not os.path.exists(config['EXPERIMENT_DIR'])):
        commons_utils.reset_dir(config['EXPERIMENT_DIR'])