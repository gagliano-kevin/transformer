import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time

from BPE_tokenizer import CustomBPETokenizer as BPE_tokenizer
from nano_transformer_class import transformer, transformerConfig


class CustomDataset(Dataset):
    """
    Custom dataset class for loading and processing text data.
    Args:
        data (str): Input text data.
        seq_len (int): Length of each sequence.
    """
    def __init__(self, tokens_tensor, seq_len=256):
        self.data = tokens_tensor
        self.seq_len = seq_len

        # Ensure we only keep valid starting indices
        self.valid_indices = [i for i in range(len(self.data) - self.seq_len - 1)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]                   
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]   
        return x, y


def load_data_from_directory(directory_path="../../datasets", exclude_files=["datasets_source.txt"], file_extension='txt'):
    """
    Load text files from a directory, excluding specified files.
    Args:
        directory_path (str): Path to the directory containing text files.
        exclude_files (list): List of filenames to exclude from loading.
        file_extension (str): File extension of the text files to load.
    Returns:
        str: Concatenated text from all files in the directory, excluding specified files.
    """
    text= ""
    for filename in os.listdir(directory_path):
        if filename.endswith(f".{file_extension}") and filename not in exclude_files:
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                text += file.read() + "\n"
    return text

# function to load data fram files in the current directory
def load_data_from_files(file_paths=["dracula-stoker.txt", "Psychology-and-Pedagogy-of-Anger.txt", "The-Invisible-Man.txt", "The-Killing-Complex.txt"]):
    """
    Load text data from specified files.
    Args:
        file_paths (list): List of file paths to load.
    Returns:
        str: Concatenated text from all specified files.
    """
    text = ""
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            text += file.read() + "\n"
    return text


def simple_init_tokenizer(vocab_size=300, pretrained=False, log=False, log_file=None):
    """
    Initialize a tokenizer, either by loading a pretrained model or training a new one.
    Args:
        vocab_size (int): Size of the vocabulary for the tokenizer.
        pretrained (bool): Whether to load a pretrained tokenizer or train a new one.
        log (bool): Whether to log the training process.
        log_file (file): Optional log file to write to.
    Returns:
        tokenizer (BPE_tokenizer): Initialized tokenizer object.
    """
    # Define log_print function that writes to console and log file if provided
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        if log_file:
            print(*args, **kwargs, file=log_file)
            log_file.flush()
    
    # Create log file if not provided
    if log and log_file is None:
        log_path = f"tokenizer_{vocab_size}_training.log"
        log_file = open(log_path, 'w')
        log_print(f"Tokenizer log saved to: {log_path}")
        own_log_file = True
    else:
        own_log_file = False
    
    if pretrained:
        log_print(f"Loading pretrained tokenizer with vocab size {vocab_size}...")
        tokenizer = BPE_tokenizer(vocab_size=vocab_size, log=log)
        tokenizer.load_model(file_prefix="tokenizer")
        log_print("Tokenizer loaded successfully.")
    else:
        log_print(f"Training new tokenizer with vocab size {vocab_size}...")
        text = load_data_from_files(["dracula-stoker.txt", "Psychology-and-Pedagogy-of-Anger.txt", "The-Invisible-Man.txt", "The-Killing-Complex.txt"])
        tokenizer = BPE_tokenizer(vocab_size=vocab_size, log=log)
        tokenizer.train(text)
        tokenizer.save_model(file_prefix="tokenizer")
        log_print(f"Tokenizer trained and saved successfully.")
    
    # Close log file if we opened it
    if log and own_log_file:
        log_file.close()
        
    return tokenizer

def train_model(model, train_loader, val_loader, optimizer, num_epochs=10, log_freq=20, model_name="", checkpoints_per_epoch=100):
    """
    Train a transformer model.
    Args:
        model (transformer): Transformer model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train the model.
        log_freq (int): Frequency of logging training progress.
        model_name (str): Name of the model to save.
        checkpoints_per_epoch (int): Number of checkpoints to save per epoch.
    """
    if model_name == "":
        raise ValueError("Model name cannot be empty.")
    model_path = model_name + ".pth"
    log_path = model_name + "_training.log"
    
    # Create a log file
    log_file = open(log_path, 'w')
    
    # Define a custom print function that writes to both console and log file
    def log_print(*args, **kwargs):
        # Print to console
        print(*args, **kwargs)
        # Print to log file
        print(*args, **kwargs, file=log_file)
        # Ensure log is written immediately
        log_file.flush()
    
    log_print("Model path:", model_path)
    log_print("Log path:", log_path)

    config_path = model_name + "_config.txt"

    with open(config_path, 'w') as f:
        f.write(f"model_name={model_name}\n")
        f.write(f"num_layers={model.config.num_layers}\n")
        f.write(f"num_heads={model.config.num_heads}\n")
        f.write(f"embedding_dim={model.config.embedding_dim}\n")
        f.write(f"feed_forward_dim={model.config.feed_forward_dim}\n")
        f.write(f"max_seq_len={model.config.max_seq_len}\n")
        f.write(f"vocab_size={model.config.vocab_size}\n")
        f.write(f"dropout={model.config.dropout}\n")
        f.write(f"num_epochs={num_epochs}\n")
    log_print("Model configuration saved to:", config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print("Using device:", device)

    dtype = torch.float32
    log_print("Using dtype:", dtype)

    if torch.cuda.is_available():
        model = model.to(device)

    checkpoint_batches = len(train_loader) // checkpoints_per_epoch
    log_print(f"Training model for {num_epochs} epochs with {len(train_loader)} batches per epoch.")

    log_print(f"Checkpoints will be saved every {checkpoint_batches} batches.")
    for epoch in range(num_epochs):
        total_loss = 0
        batch_idx = 0
        model.train()       
        avg_batch_time = 0
        token_efficiency = 0
        for x, y in train_loader:
            t0 = time.time()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)  # Direct computation on CPU
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_batch_time += time.time() - t0
            token_efficiency += x.size(0) * x.size(1)
            if batch_idx % log_freq == 0:
                token_efficiency /= avg_batch_time
                avg_batch_time /= log_freq if batch_idx > 0 else 1
                log_print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Time/batch: {avg_batch_time:.4f} s, Avg Token Efficiency: {token_efficiency:.2f} tokens/s")
                avg_batch_time = 0
                token_efficiency = 0

            if (batch_idx+1) % checkpoint_batches == 0:
                torch.save(model.state_dict(), model_path)
                log_print(f"Model checkpoint saved at batch {batch_idx+1} in epoch {epoch+1}.")

            batch_idx += 1

        log_print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        batch_idx = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _, loss = model(x, y)
                else:
                    _, loss = model(x, y)  # Direct computation on CPU
                val_loss += loss.item()
                if batch_idx % log_freq == 0:
                    log_print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(val_loader)}, Validation Loss: {loss.item():.4f}")
                batch_idx += 1
        log_print(f"Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader):.4f}")

    # Save model at the end of training
    torch.save(model.state_dict(), model_path)
    log_print("Model saved successfully.")
    
    # Close log file
    log_file.close()


def load_model(transformer_config, model_name="", device=None, log=False):  
    """
    Load a pretrained transformer model.
    Args:
        transformer_config (transformerConfig): Configuration object for the transformer model.
        model_name (str): Name of the pretrained model to load.
        device (torch.device): Device to load the model on (CPU or GPU).
        log (bool): Whether to log the loading process.
    Returns:
        model (transformer): Loaded transformer model.
    """
    if model_name == "":
        raise ValueError("Model name cannot be empty.")
    model_path = model_name + ".pth"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log: print("Using device:", device)

    if log: print("Loading model from:", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    model = transformer(transformer_config)
    state_dict = torch.load(model_path, weights_only=True, map_location=device)

    # Remove the "_orig_mod." prefix from keys (useful check when loading a torch.compile model)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)
    if log: print("Model loaded successfully.")
    return model



# Text generation function
def generate_text(prompt, max_len=200, model=None, tokenizer=None, device=None, log=False):
    """
    Generate text using a pretrained transformer model.
    Args:
        prompt (str): Input text prompt for generation.
        max_len (int): Maximum length of generated text.
        model (transformer): Pretrained transformer model.
        tokenizer (BPE_tokenizer): Tokenizer for encoding and decoding text.
        device (torch.device): Device to run the model on (CPU or GPU).
    Returns:
        str: Generated text.
    """
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be provided for text generation.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log:
        print("Using device:", device)
        print("Generating text with prompt:", prompt)
        print("Max length:", max_len)

    model.eval()
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device) #unsqueeze because we need a batch dimension
    
    with torch.no_grad():
        for _ in range(max_len):
            if tokens.size(1) >= model.config.max_seq_len:
                break
            output, _ = model(tokens)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)
    
    return tokenizer.decode(tokens.squeeze(0).tolist())


#function that creates a stream of text, one token at a time untill it reaches the end of the sequence or the max_len>256
def stream_text(prompt, max_len=256, model=None, bpe_tokenizer=None, device=None, log=False):
    """
    Stream text generation using a pretrained transformer model.
    Args:
        prompt (str): Input text prompt for generation.
        max_len (int): Maximum length of generated text.
        model (transformer): Pretrained transformer model.
        bpe_tokenizer (BPE_tokenizer): Tokenizer for encoding and decoding text.
        device (torch.device): Device to run the model on (CPU or GPU).
    Yields:
        str: Generated text piece by piece.
    """
    if model is None or bpe_tokenizer is None:
        raise ValueError("Model and tokenizer must be provided for text generation.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if log:
        print("Using device:", device)
        print("Generating text with prompt:", prompt)
        print("Max length:", max_len)

    model.eval()
    tokens = torch.tensor(bpe_tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)       # used as a token sliding window of context size dimension
    all_tokens = tokens.clone()  # Keep track of ALL tokens
    generated_text = prompt
    
    with torch.no_grad():
        for _ in range(max_len):
            if tokens.size(1) >= model.config.max_seq_len:
                # Slide window only for model input
                tokens = tokens[:, -model.config.max_seq_len:]
            
            output, _ = model(tokens)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(1)
            
            # Update both collections
            tokens = torch.cat((tokens, next_token), dim=1)                     # tokens is the sliding window of context size dimension
            all_tokens = torch.cat((all_tokens, next_token), dim=1)             # all_tokens is the whole sequence of tokens generated so far (prompt + all generated tokens)
            
            # Decode from all tokens for consistency
            new_text = bpe_tokenizer.decode(all_tokens[0].tolist())             # new_text is equivalent to all_text (prompt + all generated tokens)
            new_piece = new_text[len(generated_text):]                          # new piece is equivalent to new_text[-1] -> so isn't it the last generated token ??
            generated_text = new_text                                           # generated_text is now the whole text (prompt + all the generated tokens)                
            yield new_piece