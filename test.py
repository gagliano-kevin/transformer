import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nano_transformer_class import transformer, transformerConfig

# Define a simple real dataset
class TinyTextDataset(Dataset):
    def __init__(self):
        self.sentences = [
            "hello how are you",
            "the cat sat on the mat",
            "where is the nearest store",
            "this is a test sentence",
            "can you help me with this task"
        ]
        self.vocab = {word: idx for idx, word in enumerate(set(" ".join(self.sentences).split()))}
        self.vocab_size = len(self.vocab)
        self.data = [torch.tensor([self.vocab[word] for word in sentence.split()]) for sentence in self.sentences]
        self.max_len = max(len(seq) for seq in self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        padded_seq = torch.cat([seq, torch.zeros(self.max_len - len(seq), dtype=torch.long)])
        return padded_seq[:-1], padded_seq[1:]

# Create dataset and dataloader
dataset = TinyTextDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Transformer configuration
config = transformerConfig(
    num_layers=2,
    num_heads=2,
    embedding_dim=32,
    feed_forward_dim=64,
    max_seq_len=dataset.max_len,
    vocab_size=dataset.vocab_size,
    dropout=0.1
)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = transformer(config).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output, loss = model(x, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Test generation function
def generate_text(prompt, max_len=10):
    model.eval()
    tokens = torch.tensor([dataset.vocab.get(word, 0) for word in prompt.split()]).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            if tokens.size(1) >= config.max_seq_len:  # Prevent exceeding max_seq_len
                break
            output, _ = model(tokens)
            next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)

    return " ".join([list(dataset.vocab.keys())[idx] for idx in tokens.squeeze(0).tolist()])


# Generate text
print("Generated Text:", generate_text("hello"))
print("Generated Text:", generate_text("the cat"))
print("Generated Text:", generate_text("where is"))
print("Generated Text:", generate_text("this is"))
print("Generated Text:", generate_text("can you"))

