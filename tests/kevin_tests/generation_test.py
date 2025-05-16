import torch
import os
import sys


# Get the absolute path to the parent directory 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the parent directory to the system path if it is not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from train import init_tokenizer, load_model, stream_text
from nano_transformer_class import transformerConfig


if __name__ == "__main__":

    tokenizer = init_tokenizer(vocab_size=6000, pretrained=True, tokenizer_name="j_tokenizer", log=True)


    config = transformerConfig(
    num_layers=4,
    num_heads=4,
    embedding_dim=512,
    feed_forward_dim=1024,
    max_seq_len=256,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1
    )

    model = load_model(config, model_name="no_monte_cristo_6k_transformer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    


    prompt = "This is a test sentence, can you tokenize it? , yes, I can! , thank you! , I am happy to help! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! "

    prompt = "how are you"
    #print(prompt, end="", flush=True)   # live stream output

    for token in stream_text(prompt=prompt, max_len=128, model=model, bpe_tokenizer=tokenizer, device=device):
        print(token, end="", flush=True)