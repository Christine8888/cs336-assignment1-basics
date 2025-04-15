import torch
import cs336_basics.transformer as transformer
import cs336_basics.tokenizer as tokenizer
import cs336_basics.layers as layers
import argparse

device = 'cuda'
dtype = torch.float32

class LanguageModel():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @classmethod
    def from_files(cls, transformer_params, state_filepath, vocab_filepath, merges_filepath, special_tokens = None):
        # read in pickle files
        tokenizer = tokenizer.Tokenizer.from_files(vocab_filepath = vocab_filepath,
                                                    merges_filepath = merges_filepath,
                                                    special_tokens = special_tokens)
        
        model = transformer.TransformerLM(**transformer_params, device = device, dtype = dtype)
        state_dict = torch.load(state_filepath)
        model.load_state_dict(state_dict["model"])

        return cls(model, tokenizer)
    
    def decode(self, prompt = " ", max_tokens = 100, temperature = 1.0, top_p = 1.0):
        """Sample from language model, including prompting, temperature scaling, and top-p sampling.
        Only allow sampling one sequence at a time; batching functionality is slightly more complicated"""
        prompt_str = prompt
        prompt_tokenized = torch.Tensor(self.tokenizer.encode(prompt)).to(device)
        eot_token = self.tokenizer.encode("<|endoftext|>")[0]
        n = 0

        while prompt_tokenized[-1] != eot_token and n < max_tokens:
            last_logits = self.model(prompt_tokenized.unsqueeze(0)).squeeze(0)[-1, :]

            # apply softmax temperature scaling
            last_logits /= temperature
            probs = layers.softmax(last_logits, dim = -1)

            # apply top p
            sorted_values, sorted_indices = torch.sort(probs, descending = True)
            cumsum = torch.cumsum(sorted_values, dim = 0)
            max_ind = torch.searchsorted(cumsum, top_p)
            selected_indices = sorted_indices[:max_ind]
            selected_values = sorted_values[:max_ind]

            if len(selected_indices) <= 1:
                # fall back to greedy sampling
                # use keepdims to ensure we get same-shape tensors every time
                sample = torch.argmax(probs, keepdims = True, dim = -1)
            else:
                # sample from top p
                sample = selected_indices[torch.multinomial(selected_values, 1)]
            
            # append next token
            prompt_tokenized = torch.cat((prompt_tokenized, sample))
            prompt_tokenized = prompt_tokenized[-256:]
            n += 1
            prompt_str += self.tokenizer.decode(sample.tolist())

        return prompt_str

if __name__ == "__main__":
    files = 'openwebtext'
    tokenizer = tokenizer.Tokenizer.from_files(vocab_filepath = f"./models/{files}_vocab.pkl", merges_filepath = f"./models/{files}_merges.pkl", special_tokens = ['<|endoftext|>'])
    
    transformer_params_ts = {
        "d_model": 512,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000,
        "num_layers": 4,
        "vocab_size": 32000,
        "context_length": 256,
    }
    
    checkpoint_path = "/data/c-cye/owt_basic/0/checkpoints/checkpoint_9000.pt"
    #checkpoint_path = "/home/c-cye/assignment1-basics/cs336_basics/multirun/openwebtext/train_args.batch_size=128,train_args.lr=0.005/checkpoints/checkpoint_9000.pt"
    #checkpoint_path = "/home/c-cye/assignment1-basics/cs336_basics/multirun/2025-04-13/lr_sweep/2/checkpoints/checkpoint_9000.pt"
    model = transformer.TransformerLM(**transformer_params_ts, device = device, dtype = dtype)
    state_dict = torch.load(checkpoint_path)#, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict["model"])
    
    lm = LanguageModel(model, tokenizer)
    
    parser = argparse.ArgumentParser(description='sample from a transformer')
    parser.add_argument('--temperature', type=float, default = 1.0)
    parser.add_argument('--topp', type=float, default = 1.0)
    parser.add_argument('--max_tokens', type=int, default = 256)
    args, unknown = parser.parse_known_args()

    while True:
        user_input = input("Enter input (or 'exit' to quit): ")
        output = lm.decode(prompt = user_input, temperature = args.temperature, top_p = args.topp, max_tokens = args.max_tokens)
        print(output)
