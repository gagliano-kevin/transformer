"""
Tokenizer class for training and encoding/decoding text using a byte pair encoding (BPE) algorithm.
This class allows for the training of a tokenizer on a given text, encoding text into tokens, and decoding tokens back into text.
"""

# Module used for handling unicode characters, explicitly used in the replace_control_characters function
import unicodedata
import json
import os


class CustomBPETokenizer:
    def __init__(self, vocab_size=256, log=False):
        # check if vocab_size is greater than 256 because the first 256 tokens are reserved for single byte tokens
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
        self.vocab = {}
        self.merges = {}

        self.tokenizer_params_dir = "tests/tokenizer_params"

        self.log = log
    

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


    """token id to bytes mapping"""
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
        tokens = list(text.encode("utf-8"))
        for i in range(self.num_merges):
            freq_dict = self.get_freq_dict(tokens)
            most_frequent_pair = max(freq_dict, key = freq_dict.get)        # get the most frequent pair taking the max of the frequency dictionary
            new_token_id = 256 + i
            if self.log:
                print(f"merge {i+1}/{self.num_merges}:", most_frequent_pair, "->", new_token_id)
            tokens = self.merge(tokens, most_frequent_pair, new_token_id)
            self.merges[most_frequent_pair] = new_token_id

        self.build_vocab_T2B()


    def train_from_file(self, file_path):
        """
        Function to train the tokenizer on a given text file.
        The function takes a file path and reads the entire file, and trains the tokenizer on the text in the file.
        Arguments:
        file_path: path to the text file to train on
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.train(text)
        if self.log:
            print(f"Tokenizer trained on file: {file_path}")
 

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
        if not isinstance(text, str):
            raise ValueError("text should be a string")
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            freq_dict = self.get_freq_dict(tokens)
            pair = min(freq_dict, key = lambda pair: self.merges.get(pair, float("inf"))) 
            if pair not in self.merges:
                break
            new_token_id = self.merges[pair]
            tokens =self.merge(tokens, pair, new_token_id)
            if self.log:
                print(f"Encoding: {pair} -> {new_token_id}")
        return tokens


    def decode(self, tokens):
        """
        Function to decode a list of tokens into text.
        The function takes a list of tokens (list of integers) and returns a string of text.
        The function first checks if the tokens are in the vocabulary, and if not, it replaces them with a special token.
        Byte_tokens variable is the concatenation of the bytes associated with the token ids in the vocabulary.
        The function then decodes the bytes into a string using utf-8 encoding.
        Arguments:
        tokens: list of tokens (list of integers) to decode
        Returns:
        text: string of text decoded from the tokens
        """
        byte_tokens = b"".join([self.vocab[token] for token in tokens])
        text = byte_tokens.decode("utf-8", errors="replace")
        return text


    def replace_control_characters(self,text):
        """
        Function to replace control characters (newline, tab, etc.) in a string with their unicode code points.
        The function takes a string and returns a new string with control characters replaced by their unicode code points (represented as a 4 digit hexadecimal number).
        The function iterates through the string and checks if each character is a control character using the unicodedata module.
        Arguments:
        text: string to replace control characters in
        Returns:
        new_text: string with control characters replaced by their unicode code points
        """
        chars = []
        for ch in text:
            # check if the character is not a control character, if it is not a control character, append it to the list
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch)
            else:
                # if it is a control character, replace it with its unicode code point (represetented as a 4 digit hexadecimal number)
                chars.append(f"\\u{ord(ch):04x}")   # unicode code point 4 digits hexadecimal zero left padding (format: \uXXXX)
        return "".join(chars)


    def render_token_all_chars(self, token_byte):
        """
        Enhanced token rendering function that properly displays all characters including
        the extended ASCII range (128-254).
        
        Arguments:
        token_byte: token (bytes) to render
        
        Returns:
        token_string: descriptive string representation of the token
        """
        # For single byte tokens, use a special representation based on the byte value
        if len(token_byte) == 1:
            byte_val = token_byte[0]
            
            # ASCII control characters (0-31) and DEL (127)
            if byte_val < 32 or byte_val == 127:
                return f"\\u{byte_val:04x}"
                
            # Extended ASCII (128-255)
            elif byte_val >= 128:
                # Try different encodings to get the best representation
                try:
                    # Try Latin-1 which has a 1:1 mapping for all byte values 0-255
                    char_repr = token_byte.decode('latin-1')
                    # Still escape control characters
                    if unicodedata.category(char_repr)[0] == "C":
                        return f"\\x{byte_val:02x}"
                    return char_repr
                except:
                    return f"\\x{byte_val:02x}"
                    
            # Printable ASCII (32-126)
            else:
                return token_byte.decode('utf-8')
        
        # For multi-byte tokens, use the standard rendering
        else:
            token_string = token_byte.decode("utf-8", errors="replace")
            token_string = self.replace_control_characters(token_string)
            return token_string


    def display_vocab(self):
        """
        Display a complete representation of the vocabulary with all characters.
        
        Returns:
        display_dict: dictionary mapping token IDs to readable representations
        """
        display_dict = {}
        for token_id, token_bytes in sorted(self.vocab.items()):
            display_dict[token_id] = self.render_token_all_chars(token_bytes)
        return display_dict


    def save_model(self, file_prefix):
        """
        Save the tokenizer vocabulary and merges to separate files.
        
        Arguments:
        file_prefix: prefix for the output files
        
        Outputs:
        - {file_prefix}_vocab.json: JSON file containing the vocabulary
        - {file_prefix}_merges.txt: Text file containing the merges
        """
        # Create tokenizer parameters directory if it doesn't exist
        if not os.path.exists(self.tokenizer_params_dir):
            os.makedirs(self.tokenizer_params_dir)

        # Construct file paths
        vocab_file = os.path.join(self.tokenizer_params_dir, f"{file_prefix}_vocab.json")
        merges_file = os.path.join(self.tokenizer_params_dir, f"{file_prefix}_merges.txt")

        # Check if the files already exist
        if os.path.exists(vocab_file) or os.path.exists(merges_file):
            raise FileExistsError(f"Files with prefix {file_prefix} already exist. Please choose a different prefix.")

        # Build vocab dictionary for saving
        vocab_dict = {}
        for token_id, token_bytes in self.vocab.items():
            # Convert bytes to string representation
            #token_string = self.render_token(token_bytes)
            token_string = self.render_token_all_chars(token_bytes)
            vocab_dict[token_string] = token_id

        # Save vocab dictionary to JSON file
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
        
        # Save merges to text file
        with open(merges_file, "w", encoding="utf-8") as f:
            f.write(f"# Total merges: {len(self.merges)}\n")
            
            for (id1, id2), new_id in self.merges.items():
                # Get string representations of the token pair (integers -> bytes -> strings)
                # (id1, id2) is a tuple of integers
                # self.vocab[id1] and self.vocab[id2] are bytes
                # self.render_token(self.vocab[id1]) and self.render_token(self.vocab[id2]) are strings
                
                #s1 = self.render_token(self.vocab[id1])
                s1 = self.render_token_all_chars(self.vocab[id1])
                #s2 = self.render_token(self.vocab[id2])
                s2 = self.render_token_all_chars(self.vocab[id2])
                
                # Write to file string representations of the token pair and the new token id
                f.write(f"\"{s1}\" , \"{s2}\" -> {new_id}\n")
        if self.log:
            print(f"Vocabulary saved to {vocab_file}")
            print(f"Merges saved to {merges_file}")


    def load_model(self, file_prefix):
        """
        Load the tokenizer vocabulary and merges from the saved files.

        Arguments:
        file_prefix: prefix of the vocabulary and merges files

        Outputs:
        None. Updates the self.vocab and self.merges attributes.
        """
        # Check if the tokenizer_params_dir exists
        if not os.path.exists(self.tokenizer_params_dir):
            raise FileNotFoundError(f"Tokenizer parameters directory not found: {self.tokenizer_params_dir}")
        
        # Construct file paths
        vocab_file = os.path.join(self.tokenizer_params_dir, f"{file_prefix}_vocab.json")
        merges_file = os.path.join(self.tokenizer_params_dir, f"{file_prefix}_merges.txt")

        # Check if the files exist
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
        if not os.path.exists(merges_file):
            raise FileNotFoundError(f"Merges file not found: {merges_file}")

        # Load vocabulary from JSON file
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            self.vocab = {int(token_id): token_string.encode('utf-8', errors='replace') for token_string, token_id in vocab_data.items()}

        # Load merges from text file
        self.merges = {}
        with open(merges_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(" -> ")
                if len(parts) == 2:
                    merge_tokens_str, new_id_str = parts
                    merge_tokens = merge_tokens_str.strip().split(" , ")
                    if len(merge_tokens) == 2:
                        s1_quoted, s2_quoted = merge_tokens
                        s1_rendered = s1_quoted.strip().strip('"')
                        if s1_rendered == "": s1_rendered = "\""        # handling for string: """ (to avoid resulting in empty string)
                        s2_rendered = s2_quoted.strip().strip('"')
                        if s2_rendered == "": s2_rendered = "\""        # handling for string: """ (to avoid resulting in empty string)
                        try:
                            new_id = int(new_id_str)
                            id1 = next((token_id for token_id, token_bytes in self.vocab.items() if token_bytes.decode('utf-8', errors='replace') == s1_rendered), None)
                            id2 = next((token_id for token_id, token_bytes in self.vocab.items() if token_bytes.decode('utf-8', errors='replace') == s2_rendered), None)
                            if id1 is not None and id2 is not None:
                                self.merges[(id1, id2)] = new_id
                            else:
                                print(f"Warning: Could not find token IDs for merge: '{s1_rendered}', '{s2_rendered}' in the loaded vocabulary.")
                                print(f"Original line: {line}")
                        except ValueError:
                            print(f"Warning: Invalid new_id in merges file: {new_id_str}")
                    else:
                        print(f"Warning: Invalid merge format in line: {line.strip()}")
                else:
                    print(f"Warning: Invalid line format in merges file: {line.strip()}")
        if self.log:
            print(f"Vocabulary loaded from {vocab_file}")
            print(f"Merges loaded from {merges_file}")
            
