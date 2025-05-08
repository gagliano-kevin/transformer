"""
File to test the tokenizer module.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from BPE_tokenizer import CustomBPETokenizer as BPE_tokenizer


tokenizer = BPE_tokenizer(vocab_size=260)
tokenizer.train_from_file("../datasets/dracula-stoker.txt")
tokenizer.save_model("test")

tokenizer = BPE_tokenizer(vocab_size=1000)
tokenizer.load_model("test")
print(tokenizer.encode("This is a test sentence."))
print(tokenizer.decode(tokenizer.encode("This is a test sentence.")))