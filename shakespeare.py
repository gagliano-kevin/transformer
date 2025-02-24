import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import urllib.request
import os
from nano_transformer_class import transformer, transformerConfig

# Download the Tiny Shakespeare dataset if not already present
DATA_PATH = "tiny_shakespeare.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

if not os.path.exists(DATA_PATH):
    print("Downloading Tiny Shakespeare dataset...")
    urllib.request.urlretrieve(URL, DATA_PATH)

# Load the dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Character-level vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

# Tokenize the text
tokenized_text = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

# Define a dataset class
class ShakespeareDataset(Dataset):
    def __init__(self, data, seq_len=100):
        self.data = data
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y

# Create dataset and dataloader
seq_len = 100
batch_size = 32
dataset = ShakespeareDataset(tokenized_text, seq_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset_len = len(dataset)
print(f"Dataset length: {dataset_len}")
print(f"Number of batches: {len(dataloader)}")
print(f"{dataset_len//batch_size} batches of size {batch_size} and 1 batch of size {dataset_len % batch_size}")
print(f"Number of unique characters: {vocab_size}")
print(f"Sample input shape: {next(iter(dataloader))[0].shape}")

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer(config).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    batch_idx = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, loss = model(x, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        batch_idx += 1
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Text generation function
def generate_text(prompt, max_len=200):
    model.eval()
    tokens = torch.tensor([char_to_idx[c] for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            if tokens.size(1) >= config.max_seq_len:
                break
            output, _ = model(tokens)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)
    
    return "".join([idx_to_char[idx] for idx in tokens.squeeze(0).tolist()])

# Generate text
print("Generated Text:", generate_text("ROMEO:"))
