import regex as re
from typing import List, Dict
from collections import Counter, defaultdict

N_BYTES = 256

class BPE():
    def __init__(self, input_path: str, special_tokens: list[str] = None):
        # read in all text
        with open(input_path, 'r') as file:
           all_text = file.read()
        
        # initialize vocabulary and special tokens
        self.vocabulary = {i: bytes([i]) for i in range(N_BYTES)} # every byte
        for i, token in enumerate(special_tokens):
            self.vocabulary[N_BYTES + i] = token.encode('utf-8')
        
        self.size = N_BYTES + len(special_tokens)

        # do rough tokenization
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        words = re.findall(PAT, all_text)

        # construct mapping to counts and mapping to bytes
        self.counts = Counter(words)
        # greedy, start from longest tokens and work backwards
        sorted_vocabulary = sorted(self.vocabulary.items(), key = lambda x: len(x[1]), reverse = True)
        self.words = {word: self.encode(word, sorted_vocabulary) for word in self.counts.keys()}

        # count initial byte pairs, indexed by tuple (ind1, ind2), recording location of 1st byte
        self.pairs = defaultdict(int)
        self.locations = defaultdict(list)
        self.count_pairs()

        self.merges = []
    
    def encode(self, word: str, sorted_vocabulary):
        word_bytes = word.encode('utf-8')
        i = 0
        encoding = []

        while i < len(word_bytes):
            for id, token in sorted_vocabulary:
                n = len(token)
                
                # look for matching token
                if word_bytes[i:i + n] == token:
                    i += len(token)
                    encoding.append(id)
                    break
        
        return encoding

    def count_pairs(self):
        # only used at the beginning of BPE
        for word in self.words.keys():
            word_bytes = self.words[word]
            word_count = self.counts[word]

            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                
                # account for all instances of the word
                self.pairs[pair] += word_count
                self.locations[pair].append((word, i))
        
    def decode_pair(self, pair, string = True):
        byte_list = []
        for p in pair:
            token = self.vocabulary[p]
            if isinstance(token, str):
                print(token)
            
            byte_list.append(token)
        
        byte_list = b''.join(byte_list)
        
        if string:
            try:
                return byte_list.decode('utf-8')
            except UnicodeDecodeError:
                return byte_list.decode('utf-8', errors='replace')
        else:
            return byte_list
    
    def update(self):
        # select best merge
        pairs_ranked = sorted(self.pairs.items(), key = lambda x: (x[1], self.decode_pair(x[0])), reverse = True)
        merge_pair, count = pairs_ranked[0]

        # update self.vocabulary
        self.vocabulary[self.size] = self.decode_pair(merge_pair, string = False)
        new_id = self.size
        self.size += 1

        # update self.counts
        pair_locations = self.locations[merge_pair]

        # create (word, loc) index
        word_loc_pairs = defaultdict(list)
        for pair, loc_list in self.locations.items():
            for word, loc in loc_list:
                word_loc_pairs[(word, loc)].append(pair)

        for word, loc in pair_locations:
            # look for prev token (left)
            if loc > 0:
                prev_loc = (word, loc - 1)
                if prev_loc in word_loc_pairs:
                    for left_pair in word_loc_pairs[prev_loc]:
                        # subtract all instances
                        self.pairs[left_pair] -= self.counts[word]

                        # add to new pair
                        new_pair = (left_pair[1], new_id)
                        self.pairs[new_pair] += self.counts[word]
                        if (word, loc - 1) not in self.locations[new_pair]:
                            self.locations[new_pair].append((word, loc - 2))
            
            # look for next token (right)
            if loc < len(word) - 1:
                next_loc = (word, loc + 1)
                if next_loc in word_loc_pairs:
                    for right_pair in word_loc_pairs[next_loc]:
                        # subtract all instances
                        self.pairs[right_pair] -= self.counts[word]

                        # add to new pair
                        new_pair = (new_id, right_pair[1])
                        self.pairs[new_pair] += self.counts[word]
                        if (word, loc + 1) not in self.locations[new_pair]:
                            self.locations[new_pair].append((word, loc))

            # update word tokenization
            self.words[word][loc:loc + 2] = [new_id]
        
        self.merges.append(merge_pair)
        print(self.size, merge_pair)

        # update pair bookkeeping
        del self.pairs[merge_pair]
        del self.locations[merge_pair]

    def train(self, vocab_size: int):
        while self.size < vocab_size and self.pairs:
            self.update()
        
        return self.vocabulary, self.merges

if __name__ == "__main__":
    tokenizer = BPE("", ["<|endoftext|>"])
    tokenizer.train(270)