import json
from typing import Iterable
import regex as re
import pickle
import random
import time
import numpy as np

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens = None):
        self.id_to_token = vocab
        self.token_to_id = {bytes(v): k for k, v in vocab.items()}
        
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
        return re.split(f"({token_pattern})", text)

    def encode(self, text):
        chunks = self.split_by_special_tokens(text)
        word_list = []
        
        for piece in chunks:
            if piece in self.special_tokens:
                word_list.append(piece)
            else:
                word_list.extend(match.group() for match in re.finditer(self.pattern, piece))

        # encode words
        encoded = []
        n_words = len(word_list)
        for i, word in enumerate(word_list):
            if word in self.special_tokens:
                encoded.append(self.token_to_id[word.encode('utf-8')])
            else:
                merged = self.encode_word_from_merges(word)
                encoded.extend([self.token_to_id[b] for b in merged])
            
            if i % 100000 == 0:
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

    def token_crosses_boundary(self, buffer, current):
        # check if a token crosses a boundary by applying regex to both chunks separately and together
        buffer_words = re.finditer(self.pattern, buffer)
        buffer_words = [match.group() for match in buffer_words]
        current_words = re.finditer(self.pattern, current)
        current_words = [match.group() for match in current_words]
        separated_words = buffer_words + current_words

        combined_chunk = buffer + current
        combined_words = re.finditer(self.pattern, combined_chunk)
        combined_words = [match.group() for match in combined_words]

        # splitting chunks is not necessary?

        return False
    
        if len(separated_words) != len(combined_words):
            return True
        
        for i in range(len(separated_words)):
            if separated_words[i] != combined_words[i]:
                return True
       
        return False

    def encode_iterable(self, iterable: Iterable[str]):
        def process():
            # make iterator
            iterator = iter(iterable)
            
            # try to get first item
            try:
                buffer = next(iterator)
            except StopIteration:
                # empty iterable, throw error
                return []
                
            # stream through items, holding a buffer
            for current in iterator:    
                if self.token_crosses_boundary(buffer, current):
                    # boundary case - combine current and buffer chunk
                    combined = buffer + current
                    encoded_result = self.encode(combined)
                    for id in encoded_result:
                        yield id
                    
                    # reset buffer for next iteration
                    try:
                        buffer = next(iterator)
                    except StopIteration:
                        # no more items
                        return
                else:
                    # no boundary issues; encode buffer and update
                    encoded_result = self.encode(buffer)
                    for id in encoded_result:
                        yield id
                    buffer = current
            
            # last item
            encoded_result = self.encode(buffer)
            for id in encoded_result:
                yield id

        return list(process())
    
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

def chunked_text_generator(filepath, chunk_size=1_000_000):
    with open(filepath, 'r') as f:
        buffer = []
        total_chars = 0
        for line in f:
            buffer.append(line)
            total_chars += len(line)
            if total_chars >= chunk_size:
                yield ''.join(buffer)
                buffer = []
                total_chars = 0
        if buffer:
            yield ''.join(buffer)

def test_tokenizer(files = 'tinystories', data_path = '../data/owt_valid.txt'):
    tokenizer = Tokenizer.from_files(vocab_filepath = f"./models/{files}_vocab.pkl", merges_filepath = f"./models/{files}_merges.pkl")
    # print(tokenizer.merges)
    
    text = open(data_path, "r").read()
    text = text.split("<|endoftext|>")
    sampled_texts = random.sample(text, 10)

    compression_ratios = []
    for text in sampled_texts:
        text_bytes = text.encode('utf-8')
        encoded_ids = tokenizer.encode(text)
        print([tokenizer.decode([id]) for id in encoded_ids])
        compression_ratios.append(len(text_bytes) / len(encoded_ids))
    
    print(f"Average compression ratio: {sum(compression_ratios) / len(compression_ratios)}")

def tokenize_corpus(files = 'tinystories', data_path = '../data/TinyStoriesV2-GPT4', split = 'valid'):
    tokenizer = Tokenizer.from_files(vocab_filepath = f"./models/{files}_vocab.pkl", merges_filepath = f"./models/{files}_merges.pkl")
    
    start_time = time.time()
    tokenized_text = tokenizer.encode_iterable(chunked_text_generator(f"{data_path}-{split}.txt"))
    
    # save tokenized text to numpy array, dtype = uint16
    tokenized_text = np.array(tokenized_text, dtype=np.uint16)
    with open(f"../data/{files}_tokenized-{split}.npy", "wb") as f:
        np.save(f, tokenized_text)
    
    end_time = time.time()
    print(f"total time: {end_time - start_time} seconds")
    total_bytes = len(tokenizer.decode(tokenized_text).encode('utf-8'))
    print(f"total bytes: {total_bytes}")
    print(f"throughput: {total_bytes / (end_time - start_time)} bytes per second")
    return tokenized_text

if __name__ == "__main__":
    test_tokenizer()