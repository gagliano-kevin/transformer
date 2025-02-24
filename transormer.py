import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
from hellaswag import render_example, iterate_examples

"""
Configurations for the transformer architecture
"""
@dataclass
class transformerConfig:
    num_layers: int = 6             # Number of layers in the transformer
    num_heads: int = 8              # Number of heads in the multiheadattention blocks
    embedding_dim: int = 512        # Embedding dimension of the model
    feed_forward_dim: int = 2048    # Feed forward dimension of the model
    max_seq_len: int = 512          # Maximum length of the input sequence
    vocab_size: int = 50257         # Number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    dropout: float = 0.1            # Dropout rate


"""
Feed forward neural network with GELU activation
"""
class mlp(nn.Module):
    def __init__(self, config: transformerConfig):
        super(mlp, self).__init__()
        self.linear1 = nn.Linear(config.embedding_dim, config.feed_forward_dim)         # First linear layer
        self.linear2 = nn.Linear(config.feed_forward_dim, config.embedding_dim)         # Second linear layer
        self.linear2.SCALE = True                                                       # Flag to scale the weights of the second linear layer
        #self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = nn.GELU(self.linear1(x))                                                    # Apply GELU activation to the first linear layer
        #x = self.dropout(x)
        x = self.linear2(x)                                                             
        return x


"""
Multihead attention block
"""
class multiheadAttention(nn.Module):
    def __init__(self, config: transformerConfig):
        super(multiheadAttention, self).__init__()
        assert config.embedding_dim % config.num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.head_dim = config.embedding_dim // config.num_heads                                    # Dimension of each head
        self.num_heads = config.num_heads                                                           # Number of heads
        self.scale = self.head_dim ** 0.5                                                           # Scaling factor for the dot product
        self.query_layer = nn.Linear(config.embedding_dim, config.embedding_dim)                    # Query layer
        self.key_layer = nn.Linear(config.embedding_dim, config.embedding_dim)                      # Key layer
        self.value_layer = nn.Linear(config.embedding_dim, config.embedding_dim)                    # Value layer
        self.attention_output_layer = nn.Linear(config.embedding_dim, config.embedding_dim)         # Output layer
        self.dropout = nn.Dropout(config.dropout)                                                   # Dropout layer
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(1, 1, config.max_seq_len, config.max_seq_len))
    
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.size()
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)               # Reshape the query tensor to (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)               # Reshape the key tensor to (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)               # Reshape the value tensor to (batch_size, num_heads, seq_len, head_dim)
        scores = torch.matmul(Q, K.transpose(2,3)) / self.scale                                     # Reshape the key tensor to (batch_size, num_heads, head_dim, seq_len) and perform dot product for the attention scores
        scores = scores.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))        # Apply mask to the attention scores
        attention = F.softmax(scores, dim=-1)                                                       # Apply softmax to the attention scores 
        context = torch.matmul(self.dropout(attention), V)                                          # Apply dropout to the attention scores and multiply with the value tensor
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, embedding_dim)      # Reshape the context tensor to (batch_size, seq_len, embedding_dim)
        context = self.attention_output_layer(context)                                              # Apply the output layer to the context tensor
        return context


"""
Core transformer block
"""
class transformerBlock(nn.Module):
    def __init__(self, config: transformerConfig):
        super(transformerBlock, self).__init__()
        self.norm1_layer = nn.LayerNorm(config.embedding_dim)           # Pre-normalization layer
        self.attention_layer = multiheadAttention(config)               # Multihead attention layer
        self.norm2_layer = nn.LayerNorm(config.embedding_dim)           # Post-normalization layer
        self.mlp_layer = mlp(config)                                    # Feed forward neural network
    
    def forward(self, x):
        x = x + self.attention_layer(self.norm1_layer(x))               # Residual connection through the attention
        x = x + self.mlp_layer(self.norm2_layer(x))                     # Residual connection through the feed forward neural network
        return x


"""
Transformer model
"""
class transformer(nn.Module):
    def __init__(self, config: transformerConfig):
        super(transformer, self).__init__()
        self.config = config                                                                                            # Configuration for the transformer   

        self.token_embedding_layer = nn.Embedding(config.vocab_size, config.embedding_dim)                              # Token embedding layer
        self.positional_embedding_layer = nn.Parameter(torch.randn(config.max_seq_len, config.embedding_dim))           # Positional embedding layer
        self.transformer_blocks = nn.ModuleList([transformerBlock(config) for _ in range(config.num_layers)])           # List of transformer blocks
        self.norm_layer = nn.LayerNorm(config.embedding_dim)                                                            # Normalization layer
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)                              # Output layer

        self.token_embedding.weight = self.output_layer.weight                                                          # Tie the weights of the token embedding and output layer
    
        self.apply(self.init_weights)                                                                                   # Initialize the weights of the model
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02                                                                        # Standard deviation for initializing weights of output linear layer
            if hasattr(module, "SCALE") and module.SCALE:                                   
                std *= (2 * self.config.num_layers ** -0.5)                                   # std = std * sqrt(1/#residual_connection) - #residual_connection = 2*num_transformerBlocks
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)                           # Initialize the weights of the linear layer with normal distribution
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)                                       # Initialize the bias of the linear layer with constant value 0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)                           # Initialize the weights of the embedding layer with normal distribution
        """
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)                                           # Initialize the bias of the layer normalization layer with constant value 0
            torch.nn.init.constant_(module.weight, 1)                                         # Initialize the weights of the layer normalization layer with constant
        """

    def forward(self, x, targets=None):
        batch_size, seq_len = x.size()                                                          # Get the batch size and sequence length
        assert seq_len <= self.config.max_seq_len, f"Input sequence length = {seq_len} exceeds the maximum sequence length = {self.config.max_seq_len}"

        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)          # Generate positions tensor
        positional_embedding = self.positional_embedding_layer(positions)                       # Get positional embeddings for the input sequence (batch_size, seq_len, embedding_dim)
        token_embedding = self.token_embedding_layer(x)                                         # Get token embeddings for the input sequence (batch_size, seq_len, embedding_dim)
        x = token_embedding + positional_embedding                                              # Add token and positional embeddings 
        for block in self.transformer_blocks:
            x = block(x)                                                                        # Pass through the transformer blocks
        x = self.norm_layer(x)                                                                  # Apply normalization
        x = self.output_layer(x)                                                                # Apply output layer
        loss = None                                                                             # Initialize loss to None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))                    # Calculate the cross entropy loss
        return x, loss                                                                          # Return the output(logits) and loss

    def init_optimizers(self, weight_decay, learning_rate, device):
        parameter_dict = {param_name : param for param_name, param in self.named_parameters() if param.requires_grad}   # Get the parameters that require gradients
        weight_decay_parameters = [param for name, param in parameter_dict.items() if param.dim >= 2]                   # Get the parameters that require weight decay (only for bidiemensional tensors)
        no_weight_decay_parameters = [param for name, param in parameter_dict.items() if param.dim < 2]                 # Get the parameters that do not require weight decay
        optim_param_group = [
            {"params": weight_decay_parameters, "weight_decay": weight_decay},
            {"params": no_weight_decay_parameters, "weight_decay": 0.0}
        ]
        decay_param_tensor_num = len(weight_decay_parameters)                                                           # Number of tensors that require weight decay
        decay_param_num = sum(param.numel() for param in weight_decay_parameters)                                       # Number of parameters that require weight decay
        no_decay_param_tensor_num = len(no_weight_decay_parameters)                                                     # Number of tensors that do not require weight decay
        no_decay_param_num = sum(param.numel() for param in no_weight_decay_parameters)                                 # Number of parameters that do not require weight decay
        
        if master_process:
            print(f"number of decayed parameter tensors: {decay_param_tensor_num}, with {decay_param_num} parameters")
            print(f"number of non-decayed parameter tensors: {no_decay_param_tensor_num}, with {no_decay_param_num} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_param_group, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


"""
Data loader for the transformer model - GPT2 version
"""
class dataLoader:
    def __init__(self, batch_size, seq_len, process_rank, num_processes, split):
        self.barch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ["train", "val"], "Split must be either 'train' or 'val'"

        data_root = "edu_fineweb10B"                                        # Path to the data
        shards = os.listdir(data_root)                                      # List of shards
        shards = [shard for shard in shards if split in shards]             # Filter the shards based on the split (train/val)
        shards = sorted(shards)                                             # Sort the shards
        shards = [os.path.join(data_root, shard) for shard in shards]       # Get the full path of the shards
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"        # Check if any shards are found for the split
        if master_process:                                                  
            print(f"Found {len(shards)} shards for split {split}")  
        self.reset()

    def load_tokens(filename):
        tokens = np.load(filename).astype(np.int32)                         # Load the tokens from the file in numpy format
        return torch.tensor(tokens, dtype=torch.long)                       # Convert the tokens to torch tensor

    def reset(self):
        self.current_shard = 0                                                          # Initialize the current shard to 0
        self.tokens = self.load_tokens(self.shards[self.current_shard])                 # Load the tokens from the current shard
        self.current_position = self.barch_size * self.seq_len * self.process_rank      # Initialize the current position to the start of the shard based on the process rank

    def next_batch(self):
        B, T = self.barch_size, self.seq_len
        token_buf = self.tokens[self.current_position : (self.current_position + B * T) + 1]   # Get the tokens for the current batch (+1 ensures that the last target token is included)
        x = token_buf[:-1].view(B, T)                                                          # Get the input tokens
        y = token_buf[1:].view(B, T)                                                           # Get the target tokens
        self.current_position += B * T * self.num_processes                                    # Update the current position
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):        # If loading the next batch would be out of bounds, advance to next shard
            self.current_shard = (self.current_shard + 1) % len(self.shards)                   # Move to the next shard
            self.tokens = self.load_tokens(self.shards[self.current_shard])                    # Load the tokens from the next shard
            self.current_position = B * T * self.process_rank                                  # Update the current position
        return x, y


"""
Helper function for HellaSwag evaluation
takes tokens, mask and logits as input and returns the index of the completion with the lowest loss
"""
def get_most_likely_row(tokens, mask, logits):
    shift_logits = logits[..., :-1, :].contiguous()                                             # Shifted logits for the target tokens (excluding the last logits, used for last target token +1)
    shift_labels = tokens[..., 1:].contiguous()                                                 # Shifted labels for the target tokens (excluding the first token, used for first logits -1)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))                            # Flatten the shifted logits
    flat_shift_labels = shift_labels.view(-1)                                                   # Flatten the shifted labels
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_labels, reduction="none")      # Calculate the cross entropy loss for the shifted logits and labels
    shift_losses = shift_losses.view(tokens.size(0), -1)                                        
        
    # Get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = mask[..., 1:].contiguous()                                                      # Mask == 1 for the  tokens completion region
    masked_shift_losses = shift_losses * shift_mask                                              # Apply mask to the shifted losses
    # Sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)                                                    # Sum the losses in each row
    avg_loss = sum_loss / shift_mask.sum(dim=1)                                                  # Calculate the average loss in each row
    
    # Now we have a loss for each of the 4 completions, the one with the lowest loss should be the most likely
    best_completion = avg_loss.argmin().item()                                                         # Get the index of the completion with the lowest loss
    return best_completion

# -----------------------------------------------------------------------------------------------------------------------------------------------------------

# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1                                            # check if we are running in DDP mode (DistributedDataParallel)
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "torch.cuda is not available"
    init_process_group(backend='nccl')                                                  # initialize the process group
    ddp_rank = int(os.environ['RANK'])                                                  # global rank
    ddp_local_rank = int(os.environ['LOCAL_RANK'])                                      # local rank within the node
    ddp_world_size = int(os.environ['WORLD_SIZE'])                                      # number of processes
    device = f'cuda:{ddp_local_rank}'                                                   # device to use
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0                                                      # This process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

gpt2_encoding = tiktoken.get_encoding("gpt2")

total_batch_size = 524288                                                                                                       # 2**19, ~0.5M, in number of tokens
B = 64                                                                                                                          # Batch size per process (micro batch size)
T = 1024                                                                                                                        # Sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)                                                                 # Number of gradient accumulation steps before optimizer step

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_data_loader = dataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")                    # Data loader for training data
val_data_loader = dataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")                        # Data loader for validation data

torch.set_float32_matmul_precision('high')                                                                                      # Set the precision for matmul operations to high

model = transformer(transformerConfig(vocab_size=50304))                                                                        # Initialize the transformer model
model = model.to(device)                                                                                                        # Move the model to the device

use_compile = False                                                                                                             # Flag to use torchscript for the model
if use_compile:
    model = torch.jit.script(model)                                                                                             # Compile the model using torchscript

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], output_device=device)                                                       # Wrap the model with DistributedDataParallel

raw_model = model.module if ddp else model                                                                                      # Get the raw model (without DDP)

# Learning rate schedule
warmup_steps = 715                                                                                                              # Number of warmup steps
max_steps = 19073                                                                                                               # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

max_lr = 6e-4
min_lr = max_lr * 0.1

def get_lr(iteration):
    if iteration < warmup_steps:
        return max_lr * (iteration + 1) / warmup_steps                                                                          # 1) linear warmup for warmup_iters steps
    if iteration > max_steps:                                                                                                   # 2) if beyond the max_steps, return min_lr
        return min_lr
    decay_factor = (iteration - warmup_steps) / (max_steps - warmup_steps)                                                      # 3) if iteration is between warmup_steps and max_steps, use cosine decay
    assert 0.0 <= decay_factor <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_factor))                                                                      # Coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.init_optimizers(weight_decay=0.01, learning_rate=max_lr, device_type=device_type)                         # Initialize the optimizer

log_dir = "log"                                                                                                                 # Directory to save the logs
os.makedirs(log_dir, exist_ok=True)                                                                                             # Create the log directory
log_file = os.path.join(log_dir, f"log.txt")                                                                                    # Path to the log file
with open(log_file, "w") as f:                                                                                                  # Open the log file 
    pass

for step in range(max_steps):                                                                                                       # Training loop
    t0 = time.time()                                                                                                                # Start time for the step
    last_step = step == max_steps - 1                                                                                               # Flag to check if it is the last step

    if step % 250 == 0 or last_step:                                                                                                # Every 250 steps evaluate the model on the validation data
        model.eval()                                                                                                                # Set the model to evaluation mode
        val_data_loader.reset()                                                                                                     # Reset the validation data loader
        with torch.no_grad():   
            val_loss_accum = 0.0                                                                                                    # Initialize the validation loss accumulator 
            val_loss_steps = 20                                                                                                     # Number of steps to evaluate the model
            for _ in range(val_loss_steps):
                x, y = val_data_loader.next_batch()                                                                                 # Get the next batch of validation data
                x, y = x.to(device), y.to(device)                                                                                   # Move the data to the device
                with torch.autocast(device_type=device_type, dtype = torch.bfloat16 if device_type == "cuda" else torch.float32):   # Use autocast for mixed precision training (mixed precision allows faster training)
                    logits, loss = model(x, y)                                                                                      # Get the logits and loss
                loss = loss / val_loss_steps                                                                                        # Mean loss over the steps
                val_loss_accum += loss.detach()                                                                                     # Accumulate the loss
        
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)                                                                   # Average the loss across all processes
        
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):                                                                        # Save the model checkpoint every 5000 steps or at the last step
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),                                                                                # Save the model state
                    'optimizer': optimizer.state_dict(),                                                                            # Save the optimizer state
                    'config': raw_model.config,                                                                                     # Save the model configuration
                    'step': step,                                                                                                   # Save the current step
                    'val_loss': val_loss_accum.item(),                                                                              # Save the validation loss 
                    'seed': 1337,                                                                                                   # Save the seed
                    'max_steps': max_steps,                                                                                         # Save the maximum steps
                    'warmup_steps': warmup_steps,                                                                                   # Save the warmup steps
                    'max_lr': max_lr,                                                                                               # Save the maximum learning rate
                    'min_lr': min_lr,                                                                                               # Save the minimum learning rate
                    'grad_accum_steps': grad_accum_steps,                                                                           # Save the gradient accumulation steps
                    'log_file': log_file,                                                                                           # Save the log file
                    'log_dir': log_dir,                                                                                             # Save the log directory
                    'total_batch_size': total_batch_size,                                                                           # Save the total batch size
                    'B': B,                                                                                                         # Save the batch size
                    'T': T,                                                                                                         # Save the sequence length
                    'device': device,                                                                                               # Save the device
                    'device_type': device_type,                                                                                     # Save the device type
                    'ddp': ddp,                                                                                                     # Save the flag for DDP
                    'use_compile': use_compile,                                                                                     # Save the flag for using torchscript
                }

                torch.save(checkpoint, checkpoint_path)                                                                             # Save the checkpoint

    if (step % 250 == 0 or last_step) and (not use_compile):                                                                # Every 250 steps or at the last step, evaluate the model on the HellaSwag dataset
        model.eval()                                                                                                        # Set the model to evaluation mode
        correct_completion_num = 0                                                                                          # Initialize the number of correct completions
        completion_num = 0                                                                                                  # Initialize the number of completions
        for i, example in enumerate(iterate_examples("val")):                                                               # Iterate over the validation examples
            if i % ddp_world_size != ddp_rank:                                                                              # Skip examples that are not processed by this process
                continue
            _, tokens, mask, label = render_example(example)                                                                # Get the tokens, mask and label for the example
            token = token.to(device)                                                                                        # Move the token to the device
            mask = mask.to(device)                                                                                          # Move the mask to the device
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype = torch.bfloat16 if device_type == "cuda" else torch.float32):
                    logits, loss = model(tokens)                                                                            # Get the logits and loss
                completion = get_most_likely_row(tokens, mask, logits)                                                      # Get the most likely completion
            completion_num += 1                                                                                             # Increment the completion number
            correct_completion_num += int(completion == label)                                                              # Increment the correct completion number if the completion is correct

        if ddp:                                                                                                             # Statistics reduction across all processes
            completion_num_tensor = torch.tensor(completion_num, device=device, dtype=torch.long)                           # Convert the completion number to tensor
            correct_completion_num_tensor = torch.tensor(correct_completion_num, device=device, dtype=torch.long)           # Convert the correct completion number to tensor
            dist.all_reduce(completion_num_tensor, op=dist.ReduceOp.SUM)                                                    # Sum the completion number across all processes
            dist.all_reduce(correct_completion_num_tensor, op=dist.ReduceOp.SUM)                                            # Sum the correct completion number across all processes
            completion_num = completion_num_tensor.item()                                                                   # Get the completion number
            correct_completion_num = correct_completion_num_tensor.item()                                                   # Get the correct completion number
        completion_accuracy = correct_completion_num / completion_num                                                       # Calculate the completion accuracy

        if master_process:  
            print(f"HellaSwag accuracy: {correct_completion_num}/{completion_num}={completion_accuracy:.4f}")               # Print the completion accuracy
            with open(log_file, "a") as f:  
                f.write(f"{step} hella {completion_accuracy:.4f}\n")                                                        # Write the completion accuracy to the log file
        
         
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):                                                 # Every 250 steps or at the last step, generate text from the model
        model.eval()
        sequences_num = 4                                                                                                   # Number of sequences to generate
        seq_max_length = 32                                                                                                 # Maximum length of the generated sequence
        tokens = gpt2_encoding.encode("Hello, I'm a language model,")                                                       # Encode the input text
        tokens = torch.tensor(tokens, dtype=torch.long)                                                                     # Convert the tokens to tensor
        tokens = tokens.unsqueeze(0).repeat(sequences_num, 1)                                                               # Repeat the tokens for the number of sequences
        input_seq = tokens.to(device)                                                                                       # Move the input sequence tokens to the device
        sample_rng = torch.Generator(device=device)                                                                         # Random number generator for sampling
        sample_rng.manual_seed(42 + ddp_rank)                                                                               # Set the seed for the random number generator
        while input_seq.size(1) < seq_max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(input_seq)                                                                         # forward input sequence to get logits (B, T, vocab_size)
                logits = logits[:, -1, :]                                                                                   # Take the logits at the last position (B, vocab_size)
                probs = F.softmax(logits, dim=-1)                                                                           # Apply softmax to the logits to get the probabilities
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)                                                    # Get the top-k probabilities and indices (k = 50)
                idxs = torch.multinomial(topk_probs, 1, generator=sample_rng)                                               # Sample from the top-k probabilities for each sequence (B, 1)
                next_tokens = torch.gather(topk_indices, -1, idxs)                                                          # Gather the corresponding indices (B, 1) to get the next token
                input_seq = torch.cat((input_seq, next_tokens), dim=1)                                                      # Concatenate the next token to the input sequence

        for i in range(sequences_num):                                                                                      # Print the generated text for each sequence
            tokens = input_seq[i, :seq_max_length].tolist()
            decoded = gpt2_encoding.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
            
    model.train()                                                                                                           # Set the model to training mode
    optimizer.zero_grad()                                                                                                   # Zero the gradients
    loss_accum = 0.0                                                                                                        # Initialize the loss accumulator
    for micro_step in range(grad_accum_steps):                                                                              # Gradient accumulation loop
        x, y = train_data_loader.next_batch()                                                                               # Get the next batch of training data
        x, y = x.to(device), y.to(device)                                                                                   # Move the data to the device
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)                                         # Require backward gradient synchronization for the last step
        with torch.autocast(device_type=device_type, dtype = torch.bfloat16 if device_type == "cuda" else torch.float32):
            logits, loss = model(x, y)                                                                                      # Get the logits and loss
        loss = loss / grad_accum_steps                                                                                      # Mean loss over the steps
        loss_accum += loss.detach()                                                                                         # Accumulate the loss
        loss.backward()                                                                                                     # Backward pass
    
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)                                                                    # Average the loss across all processes
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)                                                  # Clip the gradients to avoid exploding gradients (L2 norm with max_norm = 1.0)

    lr = get_lr(step)                                                                                                       # Get the learning rate for the step
    for param_group in optimizer.param_groups:                                                                              # Update the learning rate in the optimizer
        param_group['lr'] = lr
    optimizer.step()                                                                                                        # Update the optimizer
    if device_type == "cuda":
        torch.cuda.synchronize()                                                                                            # Synchronize the CUDA device (wait for all kernels in the current stream to complete)
    t1 = time.time()                                                                                                        # End time

    dt = t1 - t0                                                                                                            # Time taken for the step
    tokens_processed = train_data_loader.barch_size * train_data_loader.seq_len * grad_accum_steps * ddp_world_size         # Number of tokens processed in the step
    tokens_per_sec = tokens_processed / dt                                                                                  # Tokens processed per second

    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()                                                                                                 # Destroy the process group
                      
    