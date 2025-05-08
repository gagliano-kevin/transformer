import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import time
from nano_transformer_class import transformer, transformerConfig

import tiktoken

from BPE_tokenizer import CustomBPETokenizer

# Suppress errors from the dynamo library (devices with CUDA capability < 7.0 will throw an error, e.g. GTX 980m)
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # Suppress TorchDynamo errors

# Download the Dracula dataset if not already present
data_path = "dracula_piece.txt"

# Load the dataset
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Check if tokenizer already exists
tokenizer_files_exist = os.path.exists("dracula_custom_tokenizer_vocab.json") and os.path.exists("draculacustom__tokenizer-merges.txt")

if not tokenizer_files_exist:
    # Initialize a tokenizer
    bpe_tokenizer = CustomBPETokenizer(vocab_size=500)  # Adjust vocab_size and min_frequency as needed
    
    # Train it on Dracula (adjust vocab_size as needed)
    bpe_tokenizer.train(text)    #min frequency means that the token must appear at least twice in the training data to be included in the vocabulary
    
    # Save the tokenizer
    bpe_tokenizer.save_model("dracula_custom_tokenizer")
    print("BPE tokenizer trained and saved.")
else:
    print("Loading existing BPE tokenizer.")
    # Load the trained tokenizer

bpe_tokenizer.load_model("dracula_custom_tokenizer")

# Tokenize the text
encodings = bpe_tokenizer.encode(text)
tokenized_text = torch.tensor(encodings.ids, dtype=torch.long) #tensor 
vocab_size = bpe_tokenizer.get_vocab_size()

# Define a custom dataset class
class DraculaDataset(Dataset):
    def __init__(self, data, seq_len=100):
        self.data = data
        self.seq_len = seq_len

        # Ensure we only keep valid starting indices
        self.valid_indices = [i for i in range(len(self.data) - self.seq_len - 1)]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        idx = self.valid_indices[idx]                   # Get a valid index
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]   
        return x, y



# Create dataset and dataloader
seq_len = 256
batch_size = 64                                              # Batch size for training reduced to fit GPU 980m memory 4GB (3.35 GiB used)
dataset = DraculaDataset(tokenized_text, seq_len)

# Create a validation set
val_size = len(dataset) // 10
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Build the training dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Dataset and Dataloaders created successfully.")
print("Vocab size:", vocab_size)
print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))



# Transformer configuration
config = transformerConfig(
    num_layers=4,
    num_heads=4,
    embedding_dim=512,
    feed_forward_dim=1024,
    max_seq_len=seq_len,
    vocab_size=vocab_size,
    dropout=0.1
)

# Initialize model, loss, and optimizer
model_path = "dracula_model_256_64_BPE.pth"

#def load_model():
#    model = transformer(config)
#    #model = torch.compile(model)

#    model.load_state_dict(torch.load(model_path, weights_only=True))
#    print("Model loaded successfully.")
#    return model


def load_model():
    model = transformer(config)
    state_dict = torch.load(model_path, weights_only=True, map_location=device)

    # Fix the keys: remove "_orig_mod." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    model.load_state_dict(new_state_dict)
    print("Model loaded successfully.")
    return model


# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#torch.set_float32_matmul_precision('high')

if os.path.exists(model_path):
    model = load_model()
else:
    model = transformer(config)
    #model = torch.compile(model)

model.to(device)

# CPU: avg time per batch ~ 3.05s without torch.compile, token efficiency ~ 1050 tokens/s

# Torch compile the model
model = torch.compile(model)        # CPU: avg time per batch ~ 2.8s with torch.compile, token efficiency ~ 1100 tokens/s

criterion = nn.CrossEntropyLoss()
#optimizer = optim.AdamW(model.parameters(), lr=1e-3)
optimizer = model.init_optimizers(weight_decay=0.01, learning_rate=6e-4, device=device)                         # Initialize the optimizer

# Precision settings
dtype = torch.float32                   # Set the default data type to float32 for old GPUs (comment out for newer GPUs)
#dtype = torch.bfloat16                  # Set the default data type to bfloat16 for newer GPUs (comment out for older GPUs)

# Training loop method with validation
def train_model(model, train_loader, val_loader, optimizer, num_epochs=10, log_freq=20, model_path=model_path, checkpoints_per_epoch=10):
    checkpoint_batches = len(train_loader) // checkpoints_per_epoch
    print(f"Training model for {num_epochs} epochs with {len(train_loader)} batches per epoch.")
    print(f"Checkpoints will be saved every {checkpoint_batches} batches.")
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
                avg_batch_time /= log_freq if batch_idx > 0 else avg_batch_time
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Time/batch: {avg_batch_time:.4f}, Avg Token Efficiency: {token_efficiency:.2f} tokens/s")
                avg_batch_time = 0
                token_efficiency = 0

            if (batch_idx+1) % checkpoint_batches == 0:
                torch.save(model.state_dict(), model_path)
                print(f"Model checkpoint saved at batch {batch_idx+1} in epoch {epoch+1}.")

            batch_idx += 1

        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_loader):.4f}")

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
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(val_loader)}, Validation Loss: {loss.item():.4f}")
                batch_idx += 1
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader):.4f}")

    # Save model at the end of training
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully.")


# Text generation function
def generate_text(prompt, max_len=200):
    model.eval()
    tokens = torch.tensor(bpe_tokenizer.encode(prompt).ids, dtype=torch.long).unsqueeze(0).to(device) #unsqueeze because we need a batch dimension
    
    with torch.no_grad():
        for _ in range(max_len):
            if tokens.size(1) >= config.max_seq_len:
                #let the window slide take the last 256 tokens of the sequence
                #tokens = tokens[:, -config.max_seq_len:]
                break
            output, _ = model(tokens)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)
    
    return bpe_tokenizer.decode(tokens.squeeze(0).tolist())

#function that creates a stream of text, one token at a time untill it reaches the max_len=256 or the end of the sequence
def stream_text(prompt, max_len=200):
    model.eval()
    tokens = torch.tensor(bpe_tokenizer.encode(prompt).ids, dtype=torch.long).unsqueeze(0).to(device)
    generated_text = prompt
    
    with torch.no_grad():
        for _ in range(max_len):
            if tokens.size(1) >= config.max_seq_len:
                #let the window slide take the last 256 tokens of the sequence
                #tokens = tokens[:, -config.max_seq_len:]
                break
            output, _ = model(tokens)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(1)
            tokens = torch.cat((tokens, next_token), dim=1)
            
            # Get full text and return just the new piece
            new_text = bpe_tokenizer.decode(tokens[0].tolist())
            new_piece = new_text[len(generated_text):]
            generated_text = new_text
            yield new_piece

#function that creates a stream of text, one token at a time untill it reaches the end of the sequence or the max_len>256
def stream_text2(prompt, max_len=200):
    model.eval()
    tokens = torch.tensor(bpe_tokenizer.encode(prompt).ids, dtype=torch.long).unsqueeze(0).to(device)
    all_tokens = tokens.clone()  # Keep track of ALL tokens
    generated_text = prompt
    
    with torch.no_grad():
        for _ in range(max_len):
            if tokens.size(1) >= config.max_seq_len:
                # Slide window only for model input
                tokens = tokens[:, -config.max_seq_len:]
            
            output, _ = model(tokens)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(1)
            
            # Update both collections
            tokens = torch.cat((tokens, next_token), dim=1)
            all_tokens = torch.cat((all_tokens, next_token), dim=1)
            
            # Decode from all tokens for consistency
            new_text = bpe_tokenizer.decode(all_tokens[0].tolist())
            new_piece = new_text[len(generated_text):]
            generated_text = new_text
            yield new_piece

# main entry point
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, optimizer, num_epochs=3, log_freq=10, model_path=model_path, checkpoints_per_epoch=10)


    # Example usage of the generate_text function
    #prompt = "dracula"
    #generated_text = generate_text(prompt, max_len=100)
    #print("Generated text:")
    #print(generated_text)


    prompt = "Vienna il mattino dopo presto"
    print(prompt, end="", flush=True)   # live stream output
    for token in stream_text2(prompt, max_len=100):
        print(token, end="", flush=True)   # live stream output