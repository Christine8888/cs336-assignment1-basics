#!/bin/bash
#SBATCH --job-name=tokenizer           # Name of the job
#SBATCH --partition=a1-batch-cpu        # Partition (queue) name
#SBATCH --qos=a1-batch-cpu-qos          # QoS name
#SBATCH --gpus=0                    # Number of GPUs (remove if you don't need GPUs)
#SBATCH --time=12:00:00             # Time limit hrs:min:sec
#SBATCH --output=output_%j.out      # Standard output log (_%j will be replaced by job ID)
#SBATCH --error=error_%j.err        # Standard error log
#SBATCH -c 8 

# Your commands here
uv run python tokenizer.py
