import json
from typing import Iterable
import regex as re
import pickle
import random
import time
import numpy as np
import multiprocessing
from functools import partial
import os
import cProfile
from . import bpe

MULTI = 32 #max(multiprocessing.cpu_count() - 1, 1)
CHUNK_SIZE = 10_000_000

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens = None):
        self.id_to_token = vocab
        self.token_to_id = {bytes(v): int(k) for k, v in vocab.items()}
        
        self.merges = {tuple(k): i for i, k in enumerate(merges)}
        self.size = len(vocab)
        # assume they are already in the BPE vocabulary
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
        
        # sort special tokens by length
        self.special_tokens.sort(key=len, reverse=True)
        
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        # read in pickle files
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        
        return cls(vocab, merges, special_tokens)

    def split_by_special_tokens(self, text):
        if not self.special_tokens:
            return [text]
        
        # escape tokens for regex and join them
        token_pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        # split and keep delimiters
        split = re.split(f"({token_pattern})", text)
        return split

    def encode(self, text):
        chunks = self.split_by_special_tokens(text)
        word_list = []
        
        for piece in chunks:
            if piece in self.special_tokens:
                word_list.append(piece)
            else:
                word_list.extend(match.group() for match in re.finditer(self.pattern, piece))

        # print(word_list)
        # encode words
        encoded = []
        n_words = len(word_list)
        
        for i, word in enumerate(word_list):
            if word in self.special_tokens:
                encoded.append(self.token_to_id[word.encode('utf-8')])
            else:
                merged = self.encode_word_from_merges(word)
                encoded.extend([self.token_to_id[b] for b in merged])
            
            if i % 1000000 == 0:
                print(f"encoded {i}/{n_words} words")
        
        return encoded
    
    def encode_word_from_merges(self, word):
        # breakpoint()
        byte_list = word.encode('utf-8')
        byte_list = [bytes([b]) for b in byte_list]

        while len(byte_list) > 1:
            first_merge = None
            first_idx = float('inf')
            first_pos = None

            for i in range(len(byte_list) - 1):
                byte_pair = (byte_list[i], byte_list[i + 1])
                if byte_pair in self.merges:
                    if self.merges[byte_pair] < first_idx:
                        # get earliest merge (from BPE training)
                        first_merge = byte_pair[0] + byte_pair[1]
                        first_idx = self.merges[byte_pair]
                        first_pos = i

            if first_merge is None:
                # no more valid merges to make
                break
            
            byte_list = byte_list[:first_pos] + [first_merge] + byte_list[first_pos + 2:]
        
        return byte_list
    
    def _process_chunk(self, text_chunk):
        return self.encode(text_chunk)
    
    def encode_iterable(self, iterable: Iterable[str], one_at_a_time = True):
        """
        Reads a file in streaming fashion, chunks it by special tokens,
        and encodes each chunk using multiprocessing.
        one_at_a_time: if True, yield one token at a time; memory-efficient
        """
        batch_num = 0
        special_token = "<|endoftext|>"
        token_len = len(special_token)
        
        def generate_chunks(chunk_size = None):
            """Internal generator that reads and chunks the file"""
            if chunk_size is None:
                chunk_size = CHUNK_SIZE
            leftover = ""
            
            f = iterable
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
                    # generators yield result but do not close function
                    yield block[: last_eot_idx + token_len]
                    # keep everything after that boundary as leftover
                    leftover = block[last_eot_idx + token_len :]

            # yield leftover text
            if leftover:
                yield leftover
        
        if one_at_a_time:
            # memory efficient, read small chunks
            chunks = generate_chunks(chunk_size = 1000)
        else:
            chunks = generate_chunks()
        
        # yield one at a time, most memory efficient
        if one_at_a_time:
            for chunk in chunks:
                for token in self.encode(chunk):
                    yield token
        
        # yield chunks at a time (up to all at once)
        else:
            all_tokens = []
            
            with multiprocessing.Pool(processes=MULTI) as pool:
                process_func = partial(self._process_chunk)
                
                while True:
                    print(f"Processing batch {batch_num}", flush=True)
                    batch_num += 1

                    # collect a batch of chunks
                    batch = []
                    for _ in range(MULTI):
                        try:
                            chunk = next(chunks)
                            batch.append(chunk)
                        except StopIteration:
                            break
                    
                    if not batch:
                        break

                    # process in batches
                    results = pool.map(process_func, batch)
                    for result in results:
                        all_tokens.extend(result)
            
            yield all_tokens
    
    def decode(self, ids):
        # first decode ids into bytes
        byte_list = b""
        for id in ids:
            if str(id) in self.id_to_token:
                byte_list += bytes(self.id_to_token[str(id)])
            elif id in self.id_to_token:
                byte_list += bytes(self.id_to_token[id])
            else:
                # use unicode replacement character
                byte_list += b"U+FFFD"
        
        # then decode bytes into text
        return byte_list.decode('utf-8', errors='replace')


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
                leftover = block[last_eot_idx + token_len :]

    # yield leftover text
    if leftover:
        yield leftover
    
def chunked_text_generator(filepath):
    with open(filepath, 'r') as f:
        buffer = []
        total_chars = 0
        for line in f:
            buffer.append(line)
            total_chars += len(line)
            if total_chars >= CHUNK_SIZE:
                yield ''.join(buffer)
                buffer = []
                total_chars = 0
        if buffer:
            yield ''.join(buffer)

def test_tokenizer(files = 'openwebtext', data_path = '../data/owt_valid.txt'):
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath = f"./models/{files}_vocab.pkl", merges_filepath = f"./models/{files}_merges.pkl", 
                                     special_tokens = special_tokens)
    
    text = open(data_path, "r").read()
    text = text.split("<|endoftext|>")
    # set random seed for reproducibility
    random.seed(42)
    sampled_texts = random.sample(text, 10)

    compression_ratios = []
    for text in sampled_texts:
        text_bytes = text.encode('utf-8')
        encoded_ids = tokenizer.encode(text)
        print([tokenizer.decode([id]) for id in encoded_ids])
        compression_ratios.append(len(text_bytes) / len(encoded_ids))
    
    print(f"Average compression ratio: {sum(compression_ratios) / len(compression_ratios)}")

    print('Longest token:')
    longest_token = max(tokenizer.token_to_id.keys(), key=len)
    print(bytes(longest_token).decode('utf-8'))

def tokenize_corpus(files = 'openwebtext', data_path = '../data/owt_', split = 'train'):
    # so stupid i forgot to split on special tokens 
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath = f"./models/{files}_vocab.pkl", merges_filepath = f"./models/{files}_merges.pkl", 
                                     special_tokens = special_tokens)
    
    start_time = time.time()

    corpus_path = f"{data_path}{split}.txt"
    # corpus_path = "../tests/fixtures/tinystories_sample.txt"
    
    memory_save = False
    
    if memory_save:
        with open(corpus_path, 'r') as f:
            for id in tokenizer.encode_iterable(f, one_at_a_time = True):
                tokens.append(id)
        
    else:
        with open(corpus_path, 'r') as f:
            tokens = list(tokenizer.encode_iterable(f, one_at_a_time = False))[0]
    
    # save as numpy array
    tokenized_text = np.array(tokens, dtype=np.uint16)

    print([tokenizer.decode([id]) for id in tokenized_text[:1000]])

    # save file
    with open(f"../data/{files}_tokenized-{split}.npy", "wb") as f:
        np.save(f, tokenized_text)
    
    end_time = time.time()
    print(f"total time: {end_time - start_time} seconds")
    # total_bytes = len(tokenizer.decode(tokenized_text).encode('utf-8'))
    # get file size
    total_bytes = os.path.getsize(f"../data/{files}_tokenized-{split}.npy")
    print(f"total bytes: {total_bytes}")
    print(f"throughput: {total_bytes / (end_time - start_time)} bytes per second")
    return tokenized_text

if __name__ == "__main__":
    # test_tokenizer(files = 'tinystories', data_path = '../data/TinyStoriesV2-GPT4-valid.txt')
    # test_tokenizer(files = 'openwebtext', data_path = '../data/owt_valid.txt')
    tokenize_corpus(files = 'tinystories', data_path = '../data/TinyStoriesV2-GPT4-', split = 'valid')
    tokenize_corpus(files = 'tinystories', data_path = '../data/TinyStoriesV2-GPT4-', split = 'train')
    tokenize_corpus(files = 'openwebtext', data_path = '../data/owt_', split = 'valid')
    tokenize_corpus(files = 'openwebtext', data_path = '../data/owt_', split = 'train')

    # # run with cprofile
    # cProfile.run('tokenize_corpus(files = "tinystories", split = "valid")', 'tokenizer_stats')
    # bpe.analyze_profile(name = 'tokenizer_stats', classname = 'Tokenizer')