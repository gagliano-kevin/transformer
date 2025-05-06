"""
Tokenizer class for training and encoding/decoding text using a byte pair encoding (BPE) algorithm.
This class allows for the training of a tokenizer on a given text, encoding text into tokens, and decoding tokens back into text.
"""

"""
To do:
- Add support for special tokens
- Add support for training from file
- Add support for saving and loading the tokenizer
- Add support for encoding and decoding with special tokens
- add logging logic
"""

import unicodedata

class tokenizer:
    def __init__(self, vocab_size=256):
        # check if vocab_size is greater than 256
        if vocab_size <= 256:
            raise ValueError("vocab_size should be greater than 256")
        # check if vocab_size is integer
        if not isinstance(vocab_size, int):
            raise ValueError("vocab_size should be an integer")
        # check if vocab_size is positive
        if vocab_size <= 0:
            raise ValueError("vocab_size should be positive")
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.vocab = self.build_vocab_T2B()
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}


    def get_freq_dict(self, tokens):
        """
        Function to get frequency dictionary of token pairs.
        The function takes a list of tokens and returns a dictionary with token pairs as keys and their frequencies as values.
        Arguments:
        tokens: list of tokens (list of integers)
        Returns:
        freq_dict: dictionary with token pairs as keys (tuple of integers) and their frequencies as values (integers)
        """
        freq_dict = {}
        for idx in range(len(tokens)-1):
            pair = tokens[idx], tokens[idx+1]       
            freq_dict[pair] = freq_dict.get(pair, 0) + 1
        return freq_dict

    def sort_by_freq(self, freq_dict):
        """
        Function to sort the frequency dictionary by frequency.
        Arguments:
        freq_dict: dictionary with token pairs as keys (tuple of integers) and their frequencies as values (integers)
        Returns:
        sorted_freq: list of tuples  with frequency as first element and token pair as second element, sorted by frequency in descending order
        """
        return sorted(((v,k) for k, v in freq_dict.items()), reverse=True)
    
    def merge(self, token_list, pair, new_token_id):
        """
        Function to merge token pairs in a list of tokens.
        The function takes a list of tokens and a pair of tokens to be merged, and returns a new list of tokens with the pair replaced by a new token id.
        Arguments:
        token_list: list of tokens (list of integers)
        pair: tuple of two tokens to be merged (tuple of integers)
        new_token_id: new token id to replace the pair (integer)
        Returns:
        new_token_list: list of tokens with the pair replaced by the new token id (list of integers)
        """
        new_token_list = []
        idx = 0
        while idx < len(token_list):
            if idx < len(token_list) - 1 and pair[0] == token_list[idx] and pair[1] == token_list[idx+1]:
                new_token_list.append(new_token_id)
                idx += 2
            else:
                new_token_list.append(token_list[idx])
                idx += 1
        return new_token_list

    """token id two bytes"""
    def build_vocab_T2B(self):
        """
        Function to build vocab dictionary, a mapping from token id to bytes.
        The function takes a list of token ids and returns a dictionary with token ids as keys (integers) and their corresponding bytes as values (bytes).
        """
        self.vocab = {token_id : bytes([token_id]) for token_id in range(256)}
        for (p0, p1), token_id in self.merges.items():
            self.vocab[token_id] = self.vocab[p0] + self.vocab[p1]


    def train(self, text):
        """
        Function to train the tokenizer on a given text.
        The function takes a string of text and builds the vocabulary and merges based on the frequency of token pairs.
        Tokens variable is a list of integers representing the bytes of the text (so in the range 0-255).
        Frequency dictionary is built from the tokens, and the most frequent pair is merged into a new token id.
        Arguments:
        text: string of text to train on
        """
        # train from file check
        # check if text is a string
        tokens = list(text.encode("utf-8"))
        for i in range(self.num_merges):
            freq_dict = self.get_freq_dict(tokens)
            most_frequent_pair = max(freq_dict, key = freq_dict.get)
            new_token_id = 256 + i
            # check if log in enabled to print merges
            print(f"merge {i+1}/{self.num_merges}:", most_frequent_pair, "->", new_token_id)
            tokens = self.merge(tokens, most_frequent_pair, new_token_id)
            self.merges[most_frequent_pair] = new_token_id

        self.build_vocab_T2B()

    
    def encode(self, text):
        """
        Function to encode a given text into tokens.
        The function takes a string of text and returns a list of tokens (list of integers) representing the bytes of the text.
        The function first encodes the text into bytes, then iterates through the bytes and merges the most frequent pairs until no more merges are possible.
        Arguments:
        text: string of text to encode
        Returns:
        tokens: list of tokens (list of integers) representing the bytes of the text
        """
        # check for string(text) or forced cast 
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            freq_dict = self.get_freq_dict(tokens)
            pair = min(freq_dict, key = lambda pair: self.merges.get(pair, float("inf")))
            if pair not in self.merges:
                break
            new_token_id = self.merges[pair]
            tokens =self.merge(tokens, pair, new_token_id)
        return tokens


    def decode(self, tokens):
        """
        Function to decode a list of tokens into text.
        The function takes a list of tokens (list of integers) and returns a string of text.
        The function first checks if the tokens are in the vocabulary, and if not, it replaces them with a special token.
        Tokens variable is the concatenation of the bytes associated with the token ids in the vocabulary.
        The function then decodes the bytes into a string using utf-8 encoding.
        Arguments:
        tokens: list of tokens (list of integers) to decode
        Returns:
        text: string of text decoded from the tokens
        """
        tokens = b"".join([self.vocab[token] for token in tokens])
        text = tokens.decode("utf-8", errors="replace")
        return text


    def replace_control_characters(self,text):
        chars = []
        for ch in text:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch)
            else:
                chars.append(f"\\u{ord(ch):04x}")   # unicode code point 4 digit hexadecimal zero left padding
        return "".join(chars)


    def render_token(self, token_byte):
        token_string = token_byte.decode("utf-8", errors="replace")
        token_string = self.replace_control_characters(token_string)
        return token_string


    def save(self, file_prefix):
        model_file = file_prefix + ".model"
        with open(model_file, "w") as f:
            f.write("tokenizer v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for token, token_id in self.special_tokens.items():
                f.write(f"{token} {token_id}\n")
            for tok_id1, tok_id2 in self.merges:
                f.write(f"{tok_id1} {tok_id2}\n")
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {id: pair for pair, id in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for id, token in self.vocab.items():
                token_string = self.render_token(token)
                if id in inverted_merges:
                    id0, id1 = inverted_merges[id]
                    s0 = self.render_token(self.vocab[id0])
                    s1 = self.render_token(self.vocab[id1])
                    f.write(f"[{s0}][{s1}] -> [{token_string}] {id}\n")
                else:
                    f.write(f"[{token_string}] {id}\n")


    def save(self, model_file):
        pass
                




tok = tokenizer(vocab_size=1000)
f = open("train_text.txt", "r")
text = f.read()
tok.train(text)
enc = tok.encode("dog bark at the door")
print(enc)
dec = tok.decode(enc)
print(dec)
#print(tok.vocab)
#print(tok.merges)
