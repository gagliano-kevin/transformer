import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time



from train import simple_init_tokenizer, load_data_from_files, CustomDataset, load_model, train_model, stream_text
from nano_transformer_class import transformer, transformerConfig


if __name__ == "__main__":

    tokenizer = simple_init_tokenizer(vocab_size=6000, pretrained=True, log=False)

    #text = load_data_from_files()

    #encoded_text = tokenizer.encode(text)

    """
    #save encoded text to a file separated by spaces
    with open('encoded_text.txt', 'w') as f:
        f.write(' '.join(map(str, encoded_text)))
    """
    
    # Load the encoded text from the file
    with open('bpe_6k_no_monte_cristo_encoded_text.txt', 'r') as f:
        encoded_text = list(map(int, f.read().split()))
    
    
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

    batch_size = 64

    dataset = CustomDataset(torch_tokens, seq_len=config.max_seq_len)

    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Build the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = load_model(config, model_name="model_name")
    model = transformer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)    

    optimizer = model.init_optimizers(weight_decay=0.01, learning_rate=6e-4, device=device)

    #"""
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=1,
        log_freq=100,
        model_name="test_j_train",
        checkpoints_per_epoch=1000,
    )
    #"""
    # Test the model
    prompt = "Project Gutenberg volunteers and employees expend considerable effort to identify"
    print(prompt, end="", flush=True)
    for token in stream_text(prompt=prompt, model=model, bpe_tokenizer=tokenizer, device=device, max_len=200):
        print(token, end="", flush=True)
        #time.sleep(0.1)
    # stream_text

    #stream_text(prompt=prompt, model=model, bpe_tokenizer=tokenizer, device=device)