"""
File to test the tokenizer module.
"""

import sys
import os

# Get the absolute path to the parent directory 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path if it is not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from BPE_tokenizer import CustomBPETokenizer as BPE_tokenizer


tokenizer = BPE_tokenizer(vocab_size=300, log=True)
tokenizer.train_from_file("../datasets/dracula-stoker.txt")
tokenizer.save_model("dracula_tokenizer")

tokenizer = BPE_tokenizer(vocab_size=1000)
tokenizer.load_model("dracula_tokenizer")
print(tokenizer.encode("This is a test sentence."))
print(tokenizer.decode(tokenizer.encode("This is a test sentence.")))