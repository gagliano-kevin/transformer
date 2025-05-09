import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time


# Get the absolute path to the parent directory 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Add the parent directory to the system path if it is not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from train import init_tokenizer, load_data_from_directory, CustomDataset, load_model, train_model, stream_text
from nano_transformer_class import transformer, transformerConfig


if __name__ == "__main__":

    tokenizer = init_tokenizer(vocab_size=5000, tokenizer_name="bpe_tok_5k", log=True)

    text = load_data_from_directory()

    encoded_text = tokenizer.encode(text)

    #save encoded text to a file separated by spaces
    with open('bpe_5k_encoded_text.txt', 'w') as f:
        f.write(' '.join(map(str, encoded_text)))
    
    """
    # Load the encoded text from the file
    with open('bpe_5k_encoded_text.txt', 'r') as f:
        loaded_encoded_text = list(map(int, f.read().split()))
    """

    torch_tokens = torch.tensor(encoded_text, dtype=torch.long)

    config = transformerConfig(
    num_layers=4,
    num_heads=4,
    embedding_dim=512,
    feed_forward_dim=1024,
    max_seq_len=256,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1
    )

    batch_size = 2

    dataset = CustomDataset(torch_tokens, seq_len=config.max_seq_len)

    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Build the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #model = load_model(config, model_name="test_load")
    model = transformer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    

    optimizer = model.init_optimizers(weight_decay=0.01, learning_rate=6e-4, device=device)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=1,
        log_freq=100,
        model_name="test_transofrmer",
        checkpoints_per_epoch=1000,
    )

    prompt = "This is a test sentence, can you tokenize it? , yes, I can! , thank you! , I am happy to help! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! "

    # stream_text
    stream_text(prompt=prompt, model=model, bpe_tokenizer=tokenizer, device=device)