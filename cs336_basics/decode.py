from . import transformer
from . import tokenizer
import torch
from . import layers


device = 'cpu'
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

        prompt_tokenized = torch.Tensor(self.tokenizer.encode(prompt), device = device, dtype = dtype)
        eot_token = self.tokenizer.encode("<|endoftext|>")[0]
        n = 0

        while prompt_tokenized[-1] != eot_token and n < max_tokens:
            logits = self.model(prompt_tokenized)

            # apply softmax temperature scaling
            logits /= temperature
            probs = layers.softmax(logits, dim = -1)

            # apply top p
            sorted_values, sorted_indices = torch.sort(probs, descending = True)
            cumsum = torch.cumsum(sorted_values, dim = 0)
            max_ind = torch.searchsorted(cumsum, top_p)
            selected_indices = sorted_indices[:max_ind]
            selected_values = sorted_values[:max_ind]

            if len(selected_indices) == 0:
                # fall back to greedy sampling
                sample = torch.argmax(probs, dim = -1)
            else:
                # sample from top p
                sample = selected_indices[torch.multinomial(selected_values, 1)]

            # append next token
            prompt_tokenized = torch.cat((prompt_tokenized, sample))
            n += 1

