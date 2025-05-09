import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time


# Get the absolute path to the parent directory 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path if it is not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from BPE_tokenizer import CustomBPETokenizer as BPE_tokenizer
from nano_transformer_class import transformer, transformerConfig

# Path to the datasets
DATASETS_DIR = "../datasets"
filenames = os.listdir(DATASETS_DIR)

# Remove the file datasets_source.txt if from the filenames list
if "datasets_source.txt" in filenames:
    filenames.remove("datasets_source.txt")
# Concatenate all the files in the datasets directory into a single string
text = ""
for filename in filenames:
    with open(os.path.join(DATASETS_DIR, filename), "r", encoding="utf-8") as f:
        text += f.read()


# Check if a pre-trained tokenizer exists
pretrained_tokenizer = os.path.exists("./multidatasets_tokenizer_params/multi_datasets_test_tokenizer_vocab.json") and os.path.exists("./multidatasets_tokenizer_params/multi_datasets_test_tokenizer_merges.txt")
print("Pre-trained tokenizer exists:", pretrained_tokenizer)

if pretrained_tokenizer:
    # Load the pre-trained tokenizer
    bpe_tokenizer = BPE_tokenizer(vocab_size=300)
    bpe_tokenizer.tokenizer_params_dir = "./multidatasets_tokenizer_params"
    bpe_tokenizer.load_model("multi_datasets_test_tokenizer")
    print("Pre-trained BPE tokenizer loaded.")
else:
    # Create a new untrained tokenizer
    bpe_tokenizer = BPE_tokenizer(vocab_size=260, log=True)
    bpe_tokenizer.tokenizer_params_dir = "./multidatasets_tokenizer_params"
    bpe_tokenizer.train(text)  # Train the tokenizer on the datasets
    bpe_tokenizer.save_model("multi_datasets_test_tokenizer")
    print("Pre-trained BPE tokenizer not found. A new one has been trained and saved.")

# Tokenize the text
encoded_text = bpe_tokenizer.encode(text)
tokenized_text = torch.tensor(encoded_text, dtype=torch.long) 
vocab_size = bpe_tokenizer.vocab_size

# Define a custom dataset class
class MultipleBooksDataset(Dataset):
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
batch_size = 2  #<-------- BATCH SIZE LIMITED TO VALUE = 2 FOR GTX980M, IF GREATER IT WILL SHUT DOWN THE COMPUTER!!!!! ( **** REMINDER FOR KEVIN **** )                                             
dataset = MultipleBooksDataset(tokenized_text, seq_len)

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

# Path to save/load the model
model_path = "../pretrained_models/multidatasets_training_test.pth"

if os.path.exists(model_path):
    print("Model checkpoint found. Loading model...")
else:
    print("No model checkpoint found. A new model will be created.")


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


if os.path.exists(model_path):
    model = load_model()
else:
    model = transformer(config)

model = transformer(config)
model.to(device)


# Torch compile the model
#model = torch.compile(model)        

criterion = nn.CrossEntropyLoss()
optimizer = model.init_optimizers(weight_decay=0.01, learning_rate=6e-4, device=device)                      

# Precision settings
dtype = torch.float32                   # Set the default data type to float32 for old GPUs 

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
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Avg Time/batch: {avg_batch_time:.4f} s, Avg Token Efficiency: {token_efficiency:.2f} tokens/s")
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
    tokens = torch.tensor(bpe_tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device) #unsqueeze because we need a batch dimension
    
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
    tokens = torch.tensor(bpe_tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
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
    tokens = torch.tensor(bpe_tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
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

    train_model(model, train_loader, val_loader, optimizer, num_epochs=1, log_freq=10, model_path=model_path, checkpoints_per_epoch=891)

    # Example usage of the text generation function
    """
    prompt = "Once upon a time"
    generated_text = generate_text(prompt, max_len=100)
    print("Generated text:")
    print(generated_text)
    """