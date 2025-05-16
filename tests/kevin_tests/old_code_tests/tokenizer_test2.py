import os
import sys

# Get the absolute path to the parent directory 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the parent directory to the system path if it is not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from train import init_tokenizer


if __name__ == "__main__":

    tokenizer = init_tokenizer(vocab_size=1000, pretrained=True, tokenizer_name="bpe_tok_5k", log=True)