import hydra
import subprocess
from omegaconf import DictConfig, OmegaConf
import os

@hydra.main(config_path="conf", config_name="config")
def run_sweep(cfg: DictConfig):
    # Create command line arguments from config
    cmd = ["uv", "run", "python", "train.py"]
    
    for k, v in cfg.train_args.items():
        cmd.extend([f"--{k}", str(v)])
    
    # print and run command correspodning to sweep
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # print outfits
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")
    
    return result.returncode

if __name__ == "__main__":
    run_sweep()