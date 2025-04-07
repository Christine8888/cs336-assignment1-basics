import regex as re
from typing import List, Dict
from collections import Counter, defaultdict
import multiprocessing
from multiprocessing import Pool
import json
import time
import cProfile
import pstats
from pstats import SortKey
import pickle

N_BYTES = 256
BASE_PATH = "/users/christineye/cs336/assignment1-basics"
MULTI = multiprocessing.cpu_count() - 1

class BPE():
    def process_document(self, text):
        # documents split by <|endoftext|> tokens
        # assume all data structures are initialized properly
        # do rough tokenization
        words = self.pattern.finditer(text)

        local_counts = defaultdict(int)
        local_words = defaultdict(bytes)
        
        # construct mapping to counts and mapping to bytes
        word_list = [match.group() for match in words]
        for word in word_list:
            local_counts[word] += 1

            if word not in local_words.keys():
                local_words[word] = self.encode(word)
        
        return local_counts, local_words

    def process_vocab(self, word):
        local_pairs = defaultdict(int)
        word_bytes = self.words[word]
        word_count = self.counts[word]
        local_locations = defaultdict(set)

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            
            # account for all instances of the word
            local_pairs[pair] += word_count
            local_locations[pair].add(word)
        
        return local_pairs, local_locations
        

    def __init__(self, input_path: str, special_tokens: list[str] = None):
        # pre-compile regex
        self.pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.counts = defaultdict(int)
        self.pairs = defaultdict(int)
        self.words = defaultdict(bytes)
        self.locations = defaultdict(set)
        self.pair_strings = defaultdict(str)
        self.merges = []
        
        with open(input_path, 'r') as file:
           all_text = file.read()
        
        # initialize vocabulary and special tokens
        self.vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            self.vocabulary[N_BYTES + i] = token.encode('utf-8')
        self.size = N_BYTES + len(special_tokens)
        self.sorted_vocabulary = sorted(self.vocabulary.items(), key = lambda x: len(x[1]), reverse = True)

        # split on every Nth EOT
        documents_split = all_text.split("<|endoftext|>")
        num_documents = len(documents_split)
        n_batches = MULTI
        documents_batched = [documents_split[i::n_batches] for i in range(n_batches)]
        documents_batched = [''.join(batch) for batch in documents_batched]
        
        # parallelize further steps
        with Pool(MULTI) as p:
            print(f"Processing {len(documents_batched)} documents")
            results = p.map(self.process_document, documents_batched)
        
        for local_counts, local_words in results:
            for word, count in local_counts.items():
                self.counts[word] += count
            
            for word, encoding in local_words.items():
                if word not in self.words:
                    self.words[word] = encoding
        
        with Pool(MULTI) as p:
            all_words = list(self.counts.keys())
            results = p.map(self.process_vocab, all_words)
        
        for local_pairs, local_locations in results:
            for pair, count in local_pairs.items():
                self.pairs[pair] += count
                if pair not in self.pair_strings:
                    self.pair_strings[pair] = self.decode_pair(pair, string = True)
            
            for pair, locations in local_locations.items():
                self.locations[pair] = self.locations[pair].union(locations)

    def encode(self, word: str):
        word_bytes = word.encode('utf-8')
        i = 0
        encoding = []
        while i < len(word_bytes):
            for id, token in self.sorted_vocabulary:
                n = len(token)
                
                # look for matching token
                if word_bytes[i:i + n] == token:
                    i += len(token)
                    encoding.append(id)
                    break
        
        return encoding

    def count_pairs(self, word):
        # only used at the beginning of BPE
        word_bytes = self.words[word]
        word_count = self.counts[word]

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            
            # account for all instances of the word
            self.pairs[pair] += word_count
            self.locations[pair].add(word)
        
    def decode_pair(self, pair, string = True, flattened = False):
        byte_tuple = (self.vocabulary[pair[0]], self.vocabulary[pair[1]])
        if string:
            return str((byte_tuple[0], byte_tuple[1]))

        if flattened:
            byte_tuple = b''.join(byte_tuple)
        
        return byte_tuple
    
    def update(self):
        # select best merge
        merge_pair, count = max(self.pairs.items(), key=lambda x: (x[1], self.pair_strings[x[0]]))

        # update vocabulary
        self.vocabulary[self.size] = self.decode_pair(merge_pair, string=False, flattened=True)
        new_id = self.size
        self.size += 1

        # get all words that we need to modify; recalculate pairs fully
        affected_words = self.locations[merge_pair].copy()

        for word in affected_words:
            word_tokens = self.words[word]

            # remove all old pairs from self.pairs, including the merge
            for i in range(len(word_tokens) - 1):
                old_pair = (word_tokens[i], word_tokens[i + 1])
                self.pairs[old_pair] -= self.counts[word]
                if self.pairs[old_pair] <= 0:
                    del self.pairs[old_pair]
                    self.locations.pop(old_pair, None)
                else:
                    self.locations[old_pair].discard(word)

            # apply the merge (replace all occurrences of the pair)
            i = 0
            new_tokens = []
            # account for multiple occurrences of the pair
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == merge_pair:
                    new_tokens.append(new_id)
                    # jump past pair
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            
            # update tokenization
            self.words[word] = new_tokens

            # add new pairs from the updated word
            for i in range(len(new_tokens) - 1):
                new_pair = (new_tokens[i], new_tokens[i + 1])
                self.pairs[new_pair] += self.counts[word]
                if new_pair not in self.pair_strings:
                    self.pair_strings[new_pair] = self.decode_pair(new_pair, string = True)
                self.locations[new_pair].add(word)

        # track merge
        byte_merge = self.decode_pair(merge_pair, string=False)
        self.merges.append(byte_merge)


    def train(self, vocab_size: int):
        while self.size < vocab_size and self.pairs:
            self.update()
            if self.size % 100 == 0:
                print(self.size)
        
        return self.vocabulary, self.merges

    def save_model(self, output_name):
        serializable_vocab = {}
        for token_id, token_bytes in self.vocabulary.items():
            serializable_vocab[str(token_id)] = list(token_bytes)
        
        # convert to lists
        serializable_merges = []
        # merges is a list of tuples, where each tuple is a pair of bytes
        for (byte1, byte2) in self.merges:
            serializable_merges.append([byte1, byte2])
        
        # make pickles
        with open("./models/" + output_name + "_vocab.pkl", 'wb') as f:
            pickle.dump(serializable_vocab, f)
            
        with open("./models/" + output_name + "_merges.pkl", 'wb') as f:
            pickle.dump(serializable_merges, f)

def analyze_profile(name = 'bpe_stats'):
    # load stats file
    p = pstats.Stats(name)
    
    # filter
    p.strip_dirs()
    
    # sort
    print("\n=== Top functions by cumulative time ===")
    p.sort_stats(SortKey.CUMULATIVE).print_stats(30)
    
    print("\n=== Top functions by total time ===")
    p.sort_stats(SortKey.TIME).print_stats(30)
    
    print("\n=== Top functions by number of calls ===")
    p.sort_stats(SortKey.CALLS).print_stats(30)
    
    # filter to only my functions
    print("\n=== Only functions in your BPE class ===")
    p.sort_stats(SortKey.CUMULATIVE).print_stats("BPE")

def train_tinystories():
    data_path = BASE_PATH + "/data/TinyStoriesV2-GPT4-train.txt"
    tokenizer = BPE(data_path, special_tokens = ["<|endoftext|>"])
    vocab_size = 10000
    vocabulary, merges = tokenizer.train(vocab_size)
    
    # serialize and save
    tokenizer.save_model('tinystories')

def train_openwebtext():
    data_path = BASE_PATH + "/data/owt_train.txt" 
    tokenizer = BPE(data_path, special_tokens = ["<|endoftext|>"])
    vocab_size = 32000
    vocabulary, merges = tokenizer.train(vocab_size)
    tokenizer.save_model('openwebtext')

if __name__=="__main__":
    # also profile memory usage
    # cProfile.run('train_openwebtext()', 'bpe_stats')
    # analyze_profile()  
    train_tinystories()