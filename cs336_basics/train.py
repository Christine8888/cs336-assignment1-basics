import torch
import torch.nn as nn
import transformer
import optimizer
import train_utils
import wandb
import numpy as np
import os
import time
import argparse
import json

log_wandb = True
DATA_DIR = "/data/c-cye/data_tokenized"
# DATA_DIR = "/home/c-cye/assignment1-basics/data_tokenized"
# DATA_DIR = "/users/christineye/cs336/assignment1-basics/data"

class TransformerTrainer:
    """Training setup for a Transformer LM."""
    def __init__(self, transformer_params, adamw_params, training_params, load_from=None):
        """
        Initialize the TransformerTrainer.

        Args:
            transformer_params (dict): transformer parameters
            adamw_params (dict): optimizer parameters
            training_params (dict): training setup parameters
            load_from (str, optional): path to load checkpoint from.
        """
        self.transformer_params = transformer_params
        self.adamw_params = adamw_params
        self.training_params = training_params
        self.load_from = load_from
        self.checkpoint_dir = "./checkpoints/"
        self.data_dir = DATA_DIR
        self.total_tokens = 0
        
        # ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # setup data paths
        self.train_path = os.path.join(self.data_dir, f"{training_params['dataset']}_tokenized-train.npy")
        self.valid_path = os.path.join(self.data_dir, f"{training_params['dataset']}_tokenized-valid.npy")
        
        # initialize model, including ablations
        if self.training_params["ablation"] == "no_layernorm":
            self.model = transformer.TransformerLMNoRMSNorm(**self.transformer_params, 
                                                          device=self.training_params["device"], 
                                                          dtype=self.training_params["dtype"])
        elif self.training_params["ablation"] == "postnorm":
            self.model = transformer.TransformerLMPostNorm(**self.transformer_params, 
                                                        device=self.training_params["device"], 
                                                        dtype=self.training_params["dtype"])
        elif self.training_params["ablation"] == "nope":
            self.model = transformer.TransformerLMNoPE(**self.transformer_params, 
                                                    device=self.training_params["device"], 
                                                    dtype=self.training_params["dtype"])
        elif self.training_params["ablation"] == "silu":
            # d_ff is overridden to be d_model * 4
            self.model = transformer.TransformerLMSiLU(**self.transformer_params, 
                                                        device=self.training_params["device"], 
                                                        dtype=self.training_params["dtype"])
        elif self.training_params["ablation"] == "weight_tying":
            self.model = transformer.TransformerLMWeightTying(**self.transformer_params, 
                                                            device=self.training_params["device"], 
                                                            dtype=self.training_params["dtype"])
        else:
            self.model = transformer.TransformerLM(**self.transformer_params, 
                                                  device=self.training_params["device"], 
                                                  dtype=self.training_params["dtype"])
        
        # compile
        self.model = torch.compile(self.model)
        torch.set_float32_matmul_precision(self.training_params["torch_precision"])
        torch.set_float32_matmul_precision('high')
        
        # initialize optimizer
        self.optim = optimizer.AdamW(self.model.parameters(), 
                                    **self.adamw_params, 
                                    device=self.training_params["device"], 
                                    dtype=self.training_params["dtype"])
        
        print('loaded model and optimizer')
        
        # load checkpoint if specified
        if self.load_from is not None:
            train_utils.load_checkpoint(self.load_from, self.model, self.optim)
            print('loaded checkpoint')
        
        # initialize run ID for logging
        self.run_id = time.strftime("%m%d_%H%M%S")
        self.load_data()

    def load_data(self):
        """
        Set up memory-efficient dataloading.
        """
        # load data memory-efficiently
        self.train_data = np.load(self.train_path, mmap_mode='r', allow_pickle = True).astype(np.uint16)
        self.valid_data = np.load(self.valid_path, mmap_mode='r', allow_pickle = True).astype(np.uint16)
        
        print('loaded data')
        
        # check loading data type
        assert np.max(self.valid_data) <= np.iinfo(np.uint16).max
    
    def setup_wandb(self):
        """
        Set up wandb logging.
        """
        wandb.init(
            project="cs336-basics", 
            name=f"run_{self.run_id}_{self.training_params['run_name']}", 
            config={
                "transformer_params": self.transformer_params,
                "adamw_params": self.adamw_params,
                "training_params": self.training_params,
                "dataset": self.training_params['dataset'],
            }
        )
    
    def train(self):
        """
        Train loop for the model.
        """
        # load data if not already loaded
        if not hasattr(self, 'train_data'):
            self.load_data()
        
        # setup wandb
        if log_wandb: self.setup_wandb()
        self.start_time = time.time()
        
        scaler = torch.cuda.amp.GradScaler() if self.training_params.get("amp", False) else None

        # train loop
        for i in range(self.training_params["n_iter"]):
            iter_start = time.time()

            batch, targets = train_utils.load_data(
                self.train_data, 
                self.training_params["batch_size"], 
                self.training_params["seq_len"], 
                self.training_params["device"],
                sample = self.training_params["sample"]
            )
            
            # set learning rate
            lr = train_utils.cosine_annealing(
                i, 
                self.training_params["alpha_max"], 
                self.training_params["alpha_min"], 
                self.training_params["T_w"], 
                self.training_params["T_c"]
            )

            for param_group in self.optim.param_groups:
                param_group["lr"] = lr
            
            # compute forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(batch)
                    loss = train_utils.CELoss(logits, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optim)
                grad_norm = train_utils.gradient_clipping(self.model.parameters(), 1.0) or 0.0
                scaler.step(self.optim)
                scaler.update()
            else:
                logits = self.model(batch)
                loss = train_utils.CELoss(logits, targets)
            
                loss.backward()
                grad_norm = train_utils.gradient_clipping(self.model.parameters(), 1.0) or 0.0
                self.optim.step()
            
            self.optim.zero_grad(set_to_none = True)
            self.total_tokens += self.training_params["batch_size"] * self.training_params["seq_len"]
        
            # log to wandb
            if log_wandb and i % self.training_params["log_every"] == 0: wandb.log({
                "loss": loss.item(),
                "learning_rate": self.optim.param_groups[0]["lr"],
                "grad_norm": grad_norm,
                "step": i,
                "wallclock": time.time() - self.start_time,
                "tok/s": self.training_params["batch_size"] * self.training_params["seq_len"] / (time.time() - iter_start),
            })
            
            # print(f"Step {i} loss: {loss.item()}")
            
            # save checkpoints
            if i % self.training_params["checkpoint_every"] == 0:
                save_path = os.path.join(self.checkpoint_dir, f"checkpoint_{i}.pt")
                train_utils.save_checkpoint(self.model, self.optim, i, save_path)
            
            # compute validation loss/perplexity
            if i % self.training_params["valid_every"] == 0:
                self.validate(i)
        
        # finish wandb logging
        if log_wandb: wandb.finish()
        
        # save final results to .txt
        with open(f"results_{self.run_id}.txt", "w") as f:
            f.write(f"Total tokens: {self.total_tokens}\n")
            f.write(f"Total time: {time.time() - self.start_time}\n")

            # compute final loss/perplexity
            valid_loss = self.validate(i)
            f.write(f"Final validation loss: {valid_loss}\n")

        save_path = os.path.join(self.checkpoint_dir, f"checkpoint_final.pt")
        train_utils.save_checkpoint(self.model, self.optim, i, save_path)
  
    def validate(self, step):
        """Compute validation loss/perplexity."""
        with torch.no_grad():
            self.model.eval()
            valid_loss = 0.0
            
            for _ in range(self.training_params["n_valid_batches"]):
                # compute validation loss/perplexity, using same batch size and seq len as training
                batch, targets = train_utils.load_data(
                    self.valid_data, 
                    self.training_params["batch_size"], 
                    self.training_params["seq_len"], 
                    self.training_params["device"]
                )
                logits = self.model(batch)
                loss = train_utils.CELoss(logits, targets)
                valid_loss += loss.item()
            
            valid_loss /= self.training_params["n_valid_batches"]
            perplexity = np.exp(valid_loss)
            
        if log_wandb: wandb.log({
            "valid_loss": valid_loss,
            "valid_perplexity": perplexity,
            "step": step,
            "total tokens": self.total_tokens,
        })
        print(valid_loss)
        
        self.model.train()


def parse_arguments():
    parser = argparse.ArgumentParser(description='train a Transformer LM')
    
    # transformer parameters
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1344, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000, help='RoPE theta parameter')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256, help='Maximum context length')
    parser.add_argument('--d_ff_ratio', type=float, default=None, help='Ratio of d_ff to d_model')

    # adamw parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='AdamW epsilon')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    
    # training parameters
    parser.add_argument('--n_iter', type=int, default=10000, help='Number of training iterations')
    parser.add_argument('--n_tokens', type=int, default=None, help='Number of tokens to train on')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Save checkpoint every N iterations')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--dtype', type=str, default='torch.float32')
    parser.add_argument('--n_valid_batches', type=int, default=10, help='Number of validation batches')
    parser.add_argument('--valid_every', type=int, default=100, help='Validate every N iterations')
    parser.add_argument('--log_every', type = int, default=100, help='Log every N iterations')
    # parser.add_argument('--alpha_max', type=float, default=1e-3, help='Maximum learning rate for cosine annealing')
    # parser.add_argument('--alpha_min', type=float, default=1e-4, help='Minimum learning rate for cosine annealing')
    # parser.add_argument('--T_w', type=int, default=500, help='Warmup period for cosine annealing')
    # parser.add_argument('--T_c', type=int, default=10000, help='Cycle length for cosine annealing')
    parser.add_argument('--dataset', type=str, default='tinystories', help='Dataset name')
    parser.add_argument('--run_name', type=str, default='default', help='Run name for wandb')
    parser.add_argument('--load_from', type=str, default=None, help='Load checkpoint from file')
    parser.add_argument('--config', type=str, default=None, help='Config file path (overrides command line args)')
    parser.add_argument('--ablation', type=str, default=None, help='Ablation study to run')
    parser.add_argument('--sample', action='store_true', default=True, help='Sample from dataloader?')
    parser.add_argument('--no-sample', action='store_false', dest='sample', help="Don't sample from dataloader")
    parser.add_argument('--precision', default='high', help='torch internal matrix multiplication precision')
    parser.add_argument('--amp', default=False, type=bool,help='use torch AMP?')

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignoring unrecognized arguments: {unknown}")

    # if config file is provided, load it
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = json.load(f)
            # update args with config values
            for key, value in config.items():
                setattr(args, key, value)
    
    return args


def main(args = None):
    if args is None:
        args = parse_arguments()
    
    # set cosine annealing by default
    args.alpha_max = args.lr
    args.alpha_min = args.lr / 10

    if args.n_tokens is not None:
        # default value: 128 * 256 * 10000
        args.n_iter = args.n_tokens // (args.batch_size * args.seq_len)
        args.n_iter = int(args.n_iter)
    
    args.T_w = args.n_iter // 20
    args.T_c = args.n_iter

    if args.d_ff_ratio is not None:
        args.d_ff = args.d_model * args.d_ff_ratio

    # make parameter dictionaries
    transformer_params = {
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "num_layers": args.num_layers,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
    }
    
    adamw_params = {
        "lr": args.lr,
        "betas": (args.beta1, args.beta2),
        "eps": args.eps,
        "weight_decay": args.weight_decay,
    }

    if args.dtype == 'torch.bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    training_params = {
        "n_iter": args.n_iter,
        "checkpoint_every": args.checkpoint_every,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "device": args.device,
        "dtype": dtype,
        "n_valid_batches": args.n_valid_batches,
        "valid_every": args.valid_every,
        "log_every": args.log_every,
        "alpha_max": args.alpha_max,
        "alpha_min": args.alpha_min,
        "T_w": args.T_w,
        "T_c": args.T_c,
        "dataset": args.dataset,
        "run_name": args.run_name,
        "ablation": args.ablation,
        "sample": args.sample,
        "torch_precision": args.precision,
        "amp": args.amp,
    }
    
    print('Training with parameters:')
    print(transformer_params)
    print(adamw_params)
    print(training_params)

    # create trainer instance
    trainer = TransformerTrainer(
        transformer_params=transformer_params,
        adamw_params=adamw_params,
        training_params=training_params,
        load_from=args.load_from
    )
    
    # start training
    # print('Validating')
    # trainer.validate(0)
    
    trainer.train()


if __name__ == "__main__":
    main()
