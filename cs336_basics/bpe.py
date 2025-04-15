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
import pdb

N_BYTES = 256
BASE_PATH = "/users/christineye/cs336/assignment1-basics"
MULTI = 1 #multiprocessing.cpu_count() - 1 
CHUNK_SIZE = 1024 *  50 # 50MB chunks

def chunk_documents_streaming(
    path: str,
    chunk_size: int = CHUNK_SIZE,
    special_token: str = "<|endoftext|>"
):
    """
    Reads 'path' in streaming fashion, yielding chunks of text that
    each end on a '<|endoftext|>' boundary.
    """

    leftover = ""
    token_len = len(special_token)

    with open(path, "r", encoding="utf-8") as f:
        while True:
            # Read one chunk_size block of text
            block = f.read(chunk_size)
            if not block:
                # no more data in file
                break

            # combine leftover from previous iteration + new block
            block = leftover + block
            leftover = ""

            # find the *last* occurrence of the special token in 'block'
            last_eot_idx = block.rfind(special_token)

            if last_eot_idx == -1:
                # no complete document in this chunk
                # keep everything in leftover for the next read
                leftover = block
            else:
                # up through last_eot_idx is a complete set of docs
                yield block[: last_eot_idx + token_len]
                # keep everything after that boundary as leftover
                leftover = block[last_eot_idx + token_len:]

    # yield leftover text
    if leftover:
        yield leftover

# old version, without taking into account EoT
def chunk_documents(path: str, n_workers: int, special_tokens: list[str] = None):
    # calculate optimal chunk size
    # old chunking approach
    file_size = os.path.getsize(path)
    chunk_size = file_size // n_workers
    
    with open(path, 'r', encoding='utf-8') as f:
        position = 0
        for _ in range(n_workers):
            chunk = f.read(chunk_size)
            # read to next newline
            if position + chunk_size < file_size:
                chunk += f.readline()
            # split on special tokens, do not include them in the chunk
            yield chunk
            position = f.tell()

def invert_string(s: bytes) -> bytes:
    return s
    # inv_bytes = bytes([255 - c for c  in s])
    return inv_bytes
    

"""Byte-Pair Encoding (BPE) tokenizer"""
class BPE():
    def process_chunk(self, text):
        # given a chunk of text
        # optimized counting
        text = re.split(self.special_pattern, text)
        text = [re.finditer(self.pattern, t) for t in text]
        # flatten list; doing this somehow fixed the vocabulary thing but not the 
        text = [m.group() for t in text for m in t]
        counts = Counter(text)
        return counts

    def process_chunk_from_boundaries(self, boundaries):
        # optimized counting
        with open(self.input_path, 'rb') as f:
            start, end = boundaries
            text = f.read(end - start).decode("utf-8", errors="ignore")
            # split and rejoin on special tokens
            text = re.split(self.special_pattern, text)
            text = [re.finditer(self.pattern, t) for t in text]
            # flatten list
            text = [m.group() for t in text for m in t]
            counts = Counter(text)
        
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
        # build split pattern
        self.special_pattern = "|".join(re.escape(token) for token in special_tokens)
        
        self.counts = defaultdict(int)
        self.pairs = defaultdict(int)
        self.words = defaultdict(bytes)
        self.locations = defaultdict(set)
        self.pair_strings = defaultdict(str)
        self.merges = []
        self.pair_heap = []
        self.input_path = input_path
        # initialize vocabulary and special tokens
        self.vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            self.vocabulary[N_BYTES + i] = token.encode('utf-8')
        self.size = N_BYTES + len(special_tokens)
        
        # with open(input_path, 'rb') as f:
        #     chunks = chunk_file(f, MULTI, "<|endoftext|>".encode("utf-8"))
        #     boundaries = zip(chunks[:-1], chunks[1:])

        # parallelize further steps
        start = time.time()
        with Pool(MULTI) as p:
            print(f"Processing with {MULTI} workers...")
            chunks = chunk_documents_streaming(input_path)
            results = p.imap_unordered(self.process_chunk, chunks, chunksize = 4)
            # results = p.imap_unordered(self.process_chunk_from_boundaries, boundaries, chunksize = 4)
            
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
                self.pair_strings[pair] = invert_string(self.decode_pair(pair, string = True))
            
            heapq.heappush(self.pair_heap, (-count, self.pair_strings[pair], pair))

        end = time.time()
        print(f"Time taken: {end - start} seconds")

    def encode(self, word: str):
        word_bytes = word.encode('utf-8')
        
        return list(word_bytes)
        
    def decode_pair(self, pair, string = True, flattened = False):
        # the criminal thing is, python stores strings differently depending on its contents
        # so strings cannot be easily "inverted" -- you have to invert the bytes

        byte_tuple = (self.vocabulary[pair[0]], self.vocabulary[pair[1]])

        if string:
            #return byte_tuple
            result = b'\x00'.join(byte_tuple)
            return result

        if flattened:
            byte_tuple = b''.join(byte_tuple)
        
        return byte_tuple
    
    def update(self):
        # select best merge. just doing dumb max works

        merge_pair, counts = max(self.pairs.items(), key = lambda x: (x[1], self.pair_strings[x[0]]))
        # while self.pair_heap:
        #     neg_count, neg_string_priority, merge_pair = heapq.heappop(self.pair_heap)
        #     count = -neg_count
            
        #     # check pair validity
        #     if merge_pair in self.pairs and self.pairs[merge_pair] == count:
        #         break
        #     elif merge_pair in self.pairs:
        #         # update count (lazily)
        #         heapq.heappush(self.pair_heap, (-self.pairs[merge_pair], 
        #                                        neg_string_priority, 
        #                                        merge_pair))
        # else:
        #     # no valid pairs found
        #     return False
        
        
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
                    heapq.heappush(self.pair_heap, (-self.pairs[old_pair], self.pair_strings[old_pair], old_pair))

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
                    self.pair_strings[new_pair] = invert_string(self.decode_pair(new_pair, string = True))

                heapq.heappush(self.pair_heap, (-self.pairs[new_pair], self.pair_strings[new_pair], new_pair))

        # track merge
        byte_merge = self.decode_pair(merge_pair, string = False)
        self.merges.append(byte_merge)
        
        if merge_pair in self.pairs:
            del self.pairs[merge_pair]


    def train(self, vocab_size: int):
        while self.size < vocab_size and self.pairs:
            # if len(self.merges) == 69:
            #     pdb.set_trace()
            self.update()
        
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

def analyze_profile(name = 'bpe_stats', classname = 'BPE'):
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
    print(f"\n=== Only functions in {classname} class ===")
    p.sort_stats(SortKey.CUMULATIVE).print_stats(classname)

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
    cProfile.run('train_tinystories()', 'bpe_stats')
    analyze_profile()  
    # train_openwebtext()