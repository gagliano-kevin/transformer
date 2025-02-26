import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os
import time
from nano_transformer_class import transformer, transformerConfig

import tiktoken 

# Download the Tiny Shakespeare dataset if not already present
DATA_PATH = "tiny_tiny_shakespeare.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Check if the dataset exists
if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare dataset...")
    urllib.request.urlretrieve(URL, DATA_PATH)

# Load the dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Use the tokenizer from the tiktoken library
#gpt_encoding = tiktoken.encoding_for_model("gpt2")
gpt_encoding = tiktoken.get_encoding("cl100k_base")  # Correct tokenizer for GPT-4

# Tokenize the text
tokenized_text = torch.tensor(gpt_encoding.encode(text), dtype=torch.long)
vocab_size = gpt_encoding.n_vocab

# Define a custom dataset class
class ShakespeareDataset(Dataset):
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
seq_len = 100
batch_size = 32
dataset = ShakespeareDataset(tokenized_text, seq_len)

# Create a validation set
val_size = len(dataset) // 10
train_size = len(dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Build the training dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Build the validation dataloader
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print("Dataset and Dataloaders created successfully.")
print("Vocab size:", vocab_size)
print("Number of training samples:", len(train_dataset))
print("Number of validation samples:", len(val_dataset))

# Transformer configuration
config = transformerConfig(
    num_layers=4,
    num_heads=4,
    embedding_dim=128,
    feed_forward_dim=256,
    max_seq_len=seq_len,
    vocab_size=vocab_size,
    dropout=0.1
)

# Initialize model, loss, and optimizer
model_path = "tiny_tiny_shakespeare_model.pth"

def load_model():
    model = transformer(config)
    model = torch.compile(model)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("Model loaded successfully.")
    return model

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.set_float32_matmul_precision('high')

if os.path.exists(model_path):
    model = load_model()
else:
    model = transformer(config)
    model = torch.compile(model)

model.to(device)

# CPU: avg time per batch ~ 3.05s without torch.compile, token efficiency ~ 1050 tokens/s

# Torch compile the model
#model = torch.compile(model)        # CPU: avg time per batch ~ 2.8s with torch.compile, token efficiency ~ 1100 tokens/s

criterion = nn.CrossEntropyLoss()
#optimizer = optim.AdamW(model.parameters(), lr=1e-3)
optimizer = model.init_optimizers(weight_decay=0.01, learning_rate=6e-4, device=device)                         # Initialize the optimizer



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
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_batch_time += time.time() - t0
            token_efficiency += x.size(0) * x.size(1)
            if batch_idx % log_freq == 0:
                token_efficiency /= avg_batch_time
                avg_batch_time /= log_freq if batch_idx > 0 else avg_batch_time
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Time/batch: {avg_batch_time:.4f}, Avg Token Efficiency: {token_efficiency:.2f} tokens/s")
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
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
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
    tokens = torch.tensor(gpt_encoding.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            if tokens.size(1) >= config.max_seq_len:
                break
            output, _ = model(tokens)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)
    
    return gpt_encoding.decode(tokens.squeeze(0).tolist())


# main entry point
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, optimizer, num_epochs=10, log_freq=10, model_path=model_path, checkpoints_per_epoch=10)
    print("Generated Text:", generate_text("ROMEO:"))
    print("Generated Text:", generate_text("JULIET:"))
