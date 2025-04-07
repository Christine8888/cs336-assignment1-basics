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
import os
import heapq

N_BYTES = 256
BASE_PATH = "/users/christineye/cs336/assignment1-basics"
MULTI = 32 #multiprocessing.cpu_count() - 1 
CHUNK_SIZE = 1024 * 1024 * 50 # 50MB chunks

def chunk_documents(path: str, n_workers: int):
    # calculate optimal chunk size
    file_size = os.path.getsize(path)
    chunk_size = file_size // n_workers
    
    with open(path, 'r', encoding='utf-8') as f:
        position = 0
        for _ in range(n_workers):
            chunk = f.read(chunk_size)
            # read to next newline
            if position + chunk_size < file_size:
                chunk += f.readline()
            yield chunk
            position = f.tell()

def invert_string(s: str) -> str:
    return ''.join(chr(255 - ord(c)) for c in s)


"""Byte-Pair Encoding (BPE) tokenizer"""
class BPE():
    def process_chunk(self, text, delimiter="<|endoftext|>"):
        # optimized counting
        counts = Counter(m.group() for m in self.pattern.finditer(text))
        return counts

    def process_vocab(self, words):
        local_pairs = defaultdict(int)
        local_locations = defaultdict(set)

        for word in words:
            word_bytes = self.words[word]
            word_count = self.counts[word]

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
        self.pair_heap = []
        
        # initialize vocabulary and special tokens
        self.vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            self.vocabulary[N_BYTES + i] = token.encode('utf-8')
        self.size = N_BYTES + len(special_tokens)
        self.sorted_vocabulary = sorted(self.vocabulary.items(), key = lambda x: len(x[1]), reverse = True)

        # with open(input_path, 'r') as file:
        #    all_text = file.read()
        # split on every Nth EOT
        # documents_split = all_text.split("<|endoftext|>")
        # num_documents = len(documents_split)
        # n_batches = MULTI
        # documents_batched = [documents_split[i::n_batches] for i in range(n_batches)]
        # documents_batched = [''.join(batch) for batch in documents_batched]
        
        # parallelize further steps
        start = time.time()
        with Pool(MULTI) as p:
            print(f"Processing with {MULTI} workers...")
            chunks = chunk_documents(input_path, n_workers = MULTI)
            results = p.imap_unordered(self.process_chunk, chunks, chunksize = 4)

            for local_counts in results:
                for word, count in local_counts.items():
                    self.counts[word] += count
        
        # don't need to parallize?
        for word in self.counts.keys():
            self.words[word] = self.encode(word)

        # gather pairs/locations
        with Pool(MULTI) as p:
            print(f"Processing with {MULTI} workers...")
            # chunk words from all_words
            all_words = list(self.counts.keys())
            all_words_batched = [all_words[i::MULTI] for i in range(MULTI)]
            results = p.imap_unordered(self.process_vocab, all_words_batched, chunksize = 4)

            for local_pairs, local_locations in results:
                for pair, count in local_pairs.items():
                    self.pairs[pair] += count
                
                for pair, locations in local_locations.items():
                    self.locations[pair].update(locations)
        
        for pair, count in self.pairs.items():
            if pair not in self.pair_strings:
                self.pair_strings[pair] = self.decode_pair(pair, string=True)
            heapq.heappush(self.pair_heap, (-count, invert_string(self.pair_strings[pair]), pair))

        end = time.time()
        print(f"Time taken: {end - start} seconds")

    def encode(self, word: str):
        word_bytes = word.encode('utf-8')
        
        return list(word_bytes)
        
    def decode_pair(self, pair, string = True, flattened = False):
        byte_tuple = (self.vocabulary[pair[0]], self.vocabulary[pair[1]])
        if string:
            return str((byte_tuple[0], byte_tuple[1]))

        if flattened:
            byte_tuple = b''.join(byte_tuple)
        
        return byte_tuple
    
    def update(self):
        # select best merge
        # merge_pair, count = max(self.pairs.items(), key=lambda x: (x[1], self.pair_strings[x[0]]))
        while self.pair_heap:
            neg_count, neg_string_priority, merge_pair = heapq.heappop(self.pair_heap)
            count = -neg_count
            
            # check pair validity
            if merge_pair in self.pairs and self.pairs[merge_pair] == count:
                break
            elif merge_pair in self.pairs:
                # update count (lazily)
                heapq.heappush(self.pair_heap, (-self.pairs[merge_pair], 
                                               neg_string_priority, 
                                               merge_pair))
        else:
            # no valid pairs found
            return False
        
        
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
                    # we accounted for all occurrences of this pair
                    del self.pairs[old_pair]
                    self.locations.pop(old_pair, None)
                else:
                    self.locations[old_pair].discard(word)
                
                if old_pair in self.pairs:
                    heapq.heappush(self.pair_heap, (-self.pairs[old_pair], invert_string(self.pair_strings[old_pair]), old_pair))

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
                old_count = self.pairs[new_pair]
                
                self.pairs[new_pair] += self.counts[word]
                self.locations[new_pair].add(word)
                
                if new_pair not in self.pair_strings:
                    self.pair_strings[new_pair] = self.decode_pair(new_pair, string = True)

                heapq.heappush(self.pair_heap, (-self.pairs[new_pair], invert_string(self.pair_strings[new_pair]), new_pair))

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
    cProfile.run('train_openwebtext()', 'bpe_stats')
    analyze_profile()  
    # train_openwebtext()