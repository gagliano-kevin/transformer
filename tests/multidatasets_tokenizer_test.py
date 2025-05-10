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

# Path to the datasets
DATASETS_DIR = "../datasets"
filenames = os.listdir(DATASETS_DIR)

# Remove the file datasets_source.txt if from the filenames list
if "datasets_source.txt" in filenames:
    filenames.remove("datasets_source.txt")
# Concatenate all the files in the datasets directory into a single string
text = ""
for filename in filenames:
    with open(os.path.join(DATASETS_DIR, filename), "r", encoding="utf-8") as f:
        text += f.read()


tokenizer = BPE_tokenizer(vocab_size=260, log=True)
tokenizer.train(text)
tokenizer.tokenizer_params_dir = "./multidatasets_tokenizer_params"
tokenizer.save_model("multidatasets_tokenizer")

tokenizer = BPE_tokenizer(vocab_size=260)
tokenizer.tokenizer_params_dir = "./multidatasets_tokenizer_params"
tokenizer.load_model("multidatasets_tokenizer")
text_len = len(text)
print(f"\n\nlen of thext: {text_len}\n\n")

# Test on 1/10 of the text
print(tokenizer.encode(text[:text_len//10]))
print(tokenizer.decode(tokenizer.encode(text[:text_len//10])))