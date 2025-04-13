
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.95), eps = 1e-8, weight_decay = 1e-2, device = None, dtype = torch.float32):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if any(beta < 0 or beta > 1 for beta in betas):
            raise ValueError(f"Invalid beta values: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        # initialize first and second moments
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p]["m"] = torch.zeros_like(p.data, dtype = dtype, device = device)
                self.state[p]["v"] = torch.zeros_like(p.data, dtype = dtype, device = device)
    
    def step(self):
        for group in self.param_groups:
            alpha = group["lr"]
            beta_1, beta_2 = group["betas"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                prev_m = state["m"]
                prev_v = state["v"]
                grad = p.grad.data
                t = state.get("t", 1)

                # update state
                state["m"] = beta_1 * prev_m + (1 - beta_1) * grad
                state["v"] = beta_2 * prev_v + (1 - beta_2) * grad**2

                alpha_t = alpha * (1 - (beta_2)**t)**(1/2)
                alpha_t /= (1 - (beta_1)**t) 
                # update and decay weights
                p.data -= alpha_t * state["m"] / (state["v"] ** 0.5 + epsilon)
                p.data -= alpha * weight_decay * p.data
                
                # update timestep
                state["t"] = t + 1

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        
        return loss

def SGD_loop(lr, n_iter):
    weights = torch.nn.Parameter(5 * torch.randn((10,10)))
    opt = SGD([weights], lr = lr)

    for t in range(n_iter):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
    
    print('Final loss: {}'.format(loss.cpu().item()))

def SGD_experiment(lrs = [1e1, 1e2, 1e3], n_iter = 10):
    for lr in lrs:
        SGD_loop(lr, n_iter)

if __name__ == "__main__":
    SGD_experiment()