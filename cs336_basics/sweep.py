import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import argparse

# use infra from train.py
from train import parse_arguments, main

@hydra.main(config_path="conf", config_name="config_90")
def run_sweep(cfg: DictConfig):
    print(f"Working directory: {os.getcwd()}")
    
    # use existing command line infrastructure
    # get the default arguments from train.py
    args = parse_arguments()
    
    # override in config
    for param_name, param_value in cfg.train_args.items():
        if hasattr(args, param_name):
            setattr(args, param_name, param_value)
    
    args.run_name = f"sweep_{os.path.basename(os.getcwd())}"
    
    print("Running train.py with parameters:")
    print(vars(args))
    
    # use main function fron train.py
    main(args)
    
    return 0

if __name__ == "__main__":
    run_sweep()
