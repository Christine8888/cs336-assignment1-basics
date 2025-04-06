import bpe
import json
from typing import Iterable
import regex as re

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.size = len(vocab)
        # assume they are already in the BPE vocabulary
        self.special_tokens = special_tokens
        
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        # read in json files
        with open(vocab_filepath, 'r') as f:
            vocab = json.load(f)
        with open(merges_filepath, 'r') as f:
            merges = json.load(f)
        
        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        # rough tokenization
        words = self.pattern.finditer(text)
        word_list = [match.group() for match in words]

        # encode words
        encoded = []
        for word in word_list:
            if word in self.special_tokens:
                encoded.append(self.vocab[word])
            else:
                encoded.extend(self.encode_word(word))
        
        return encoded
    
    def encode_word_from_merges(self, word):
        

    def encode_iterable(self, iterable: Iterable[str]):
        pass
    
    def decode(self, ids):
        # first decode ids into bytes
        byte_list = b""
        for id in ids:
            if id in self.vocab:
                byte_list += self.vocab[id]
            else:
                # use unicode replacement character
                byte_list += b"U+FFFD"
        
        # then decode bytes into text
        return byte_list.decode('utf-8', errors='replace')
    