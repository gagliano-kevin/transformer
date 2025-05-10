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

"""
tokenizer = BPE_tokenizer(vocab_size=1000, log=True)
tokenizer.train_from_file("../datasets/dracula-stoker.txt")
tokenizer.save_model("dracula_tokenizer")
"""

tokenizer = BPE_tokenizer(vocab_size=1000, log=True)
tokenizer.load_model("dracula_tokenizer")

text = " This is a test sentence, can you tokenize it? , yes, I can! , thank you! , I am happy to help! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! "
"""
with open("../datasets/dracula-stoker.txt", "r", encoding="utf-8") as f:
    text += f.read()
"""
#print(tokenizer.encode(text))
#print(tokenizer.decode(tokenizer.encode("This is a test sentence.")))

encoded_text = tokenizer.encode(text)

#save encoded text to a file separated by spaces
with open('encoded_text.txt', 'w') as f:
    f.write(' '.join(map(str, encoded_text)))
    
# Load the encoded text from the file
with open('encoded_text.txt', 'r') as f:
    loaded_encoded_text = list(map(int, f.read().split()))

print(loaded_encoded_text)
print(tokenizer.decode(loaded_encoded_text))