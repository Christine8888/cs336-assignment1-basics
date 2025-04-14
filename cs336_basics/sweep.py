import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
import argparse

# Import parse_arguments and main from train
from train import parse_arguments, main

@hydra.main(config_path="conf", config_name="config_batch")
def run_sweep(cfg: DictConfig):
    print(f"Working directory: {os.getcwd()}")
    
    # Get the default arguments from train.py
    args = parse_arguments()
    
    # Update args with values from our config
    for param_name, param_value in cfg.train_args.items():
        if hasattr(args, param_name):
            setattr(args, param_name, param_value)
    
    # Set a custom run name to identify the sweep
    args.run_name = f"sweep_{os.path.basename(os.getcwd())}"
    
    print("Running with parameters:")
    print(vars(args))
    
    # Call the main function from train.py
    main(args)
    
    return 0

if __name__ == "__main__":
    run_sweep()
