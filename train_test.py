import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time


from nano_transformer_class import transformer, transformerConfig
from train import CustomDataset, reg_train_model, init_tokenizer, load_data_from_directory

if __name__ == "__main__":

    exclude_files = ["datasets_source.txt", "Psychology-and-Pedagogy-of-Anger.txt", "The-Count-of-Monte-Cristo.txt"]

    tokenizer = init_tokenizer(vocab_size=10000, exclude_files=exclude_files,pretrained=False, tokenizer_name="bpe_10k", log=True)

    text = load_data_from_directory(exclude_files=exclude_files)

    encoded_text = tokenizer.encode(text)
    #"""
    #save encoded text to a file separated by spaces
    with open('bpe_10k_encoded_text.txt', 'w') as f:
        f.write(' '.join(map(str, encoded_text)))
    #"""
    """
    # Load the encoded text from the file
    with open('bpe_6k_no_monte_cristo_encoded_text.txt', 'r') as f:
        encoded_text = list(map(int, f.read().split()))
    """

    torch_tokens = torch.tensor(encoded_text, dtype=torch.long)

    "milli_transformer"
    config = transformerConfig(
    num_layers=8,
    num_heads=8,
    embedding_dim=1024,
    feed_forward_dim=1024,
    max_seq_len=256,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1
    )

    batch_size = 8

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

    optimizer = model.init_optimizers(weight_decay=0.01, learning_rate=1e-5, device=device)

    reg_train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=1,
        log_freq=250,
        model_name="milli_transformer",
        checkpoints_per_epoch=50,
    )

    #prompt = "This is a test sentence, can you tokenize it? , yes, I can! , thank you! , I am happy to help! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! , I am a large language model trained by OpenAI. , I can help you with many things, such as writing code, answering questions, and providing information on a wide range of topics. , I am here to assist you in any way I can! "

    # stream_text
    #stream_text(prompt=prompt, model=model, bpe_tokenizer=tokenizer, device=device)