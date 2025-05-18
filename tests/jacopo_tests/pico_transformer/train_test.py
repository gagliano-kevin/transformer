import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time



from train_log import simple_init_tokenizer, load_data_from_files, CustomDataset, load_model, train_model, stream_text_old, stream_text, stream_text_medium
from nano_transformer_class import transformer, transformerConfig


if __name__ == "__main__":

    tokenizer = simple_init_tokenizer(vocab_size=7000, pretrained=False, log=True)

    text = load_data_from_files()

    encoded_text = tokenizer.encode(text)

    
    #save encoded text to a file separated by spaces
    with open('7k_pico_transformer_encoded_text.txt', 'w') as f:
        f.write(' '.join(map(str, encoded_text)))
    """    
    
    # Load the encoded text from the file
    with open('bpe_8k_encoded_text.txt', 'r') as f:
        loaded_encoded_text = list(map(int, f.read().split()))
    
    
    torch_tokens = torch.tensor(encoded_text, dtype=torch.long)

    config = transformerConfig(
    num_layers=2,
    num_heads=2,
    embedding_dim=256,
    feed_forward_dim=512,
    max_seq_len=64,
    vocab_size=tokenizer.vocab_size,
    dropout=0.1
    )

    batch_size = 128

    dataset = CustomDataset(torch_tokens, seq_len=config.max_seq_len)

    val_size = len(dataset) // 10
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Build the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #model = load_model(config, model_name="8k_transformer")
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
        model_name="femto_transformer",
        checkpoints_per_epoch=500,
    )
    
    # Test the model
    prompt = "the Count Dracula is a bad person"
    print(prompt, end="", flush=True)
    for token in stream_text_old(prompt=prompt, model=model, bpe_tokenizer=tokenizer, device=device, max_len=100):
        print(token, end="", flush=True)
        #time.sleep(0.1)
    # stream_text

    #stream_text(prompt=prompt, model=model, bpe_tokenizer=tokenizer, device=device)
    """