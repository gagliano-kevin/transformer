"""
Test class for regularization of the NanoTransformer model.
"""
#------------------------------------------------------------------------------------------------------------------
"""
Main fixes to try in train loop:

X)gradient clipping before the optimizer step -> torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

X)drecrease the learning rate -> from 6e-4 to 1e-5

X)check for nan inputs in the model ->
if torch.isnan(loss):
    print("NaN loss encountered. Saving batch.")
    torch.save((x, y), "bad_batch.pt")
    break

X)proper use of autocast -> with torch.autocast(device_type="cuda", dtype=torch.float16):  # instead of float32

X)Summary:
# Before optimizer.step()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Replace dtype if using autocast
with torch.autocast(device_type="cuda", dtype=torch.float16):
    _, loss = model(x, y)

# Optional: Debug NaN loss
if torch.isnan(loss):
    print("NaN loss encountered. Exiting.")
    torch.save((x, y), "nan_input.pt")
    return
"""
#------------------------------------------------------------------------------------------------------------------

"""
Main fixes to try in the model:

X)loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1)) ->
This assumes that x contains well-scaled logits. If your transformer outputs explode 
(which can happen, especially with tied weights and no temperature scaling), cross_entropy() will result in NaN.
--> F.cross_entropy() internally applies log_softmax, so extremely large logits will break it.

X)self.token_embedding_layer.weight = self.output_layer.weight ->
Weight Tying is okay, but it amplifies the chance of instability if the tied weights are not properly initialized or gradients are large.

X)uncomment LayerNorm Initialization ->
torch.nn.init.constant_(module.bias, 0)
torch.nn.init.constant_(module.weight, 1)

X)logit clamping (safer, to prevent extreme logits from blowing up cross-entropy)->
logits = x.view(-1, x.size(-1))
logits = torch.clamp(logits, min=-20, max=20)  # Prevents extreme values
loss = F.cross_entropy(logits, targets.view(-1))

X)uncomment dropout in the MLP (Dropout is essential for regularization, reducing overfitting, preventing activation explosions in deep nets)->
self.dropout = nn.Dropout(config.dropout)
x = self.dropout(x)

X)attention mask check, if there's a bug where the diagonal is ever masked incorrectly, or all scores in a row become -inf, then softmax(-inf, -inf, ...) returns NaN.
->for debug use -> assert torch.any(scores != float("-inf")), "All scores are masked — attention will NaN"

X)layerNorm eps set ->
nn.LayerNorm(config.embedding_dim, eps=1e-5)

"""

import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

"""
Configurations for the transformer architecture
"""
@dataclass
class transformerConfig:
    num_layers: int = 6                                     # Number of layers in the transformer
    num_heads: int = 8                                      # Number of heads in the multiheadattention blocks
    embedding_dim: int = 512                                # Embedding dimension of the model
    feed_forward_dim: int = 2048                            # Feed forward dimension of the model
    max_seq_len: int = 512                                  # Maximum length of the input sequence
    vocab_size: int = 50257                                 # Number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    dropout: float = 0.1                                    # Dropout rate


"""
Feed forward neural network with GELU activation
"""
class mlp(nn.Module):
    def __init__(self, config: transformerConfig):
        super(mlp, self).__init__()
        self.linear1 = nn.Linear(config.embedding_dim, config.feed_forward_dim)             # First linear layer
        self.linear2 = nn.Linear(config.feed_forward_dim, config.embedding_dim)             # Second linear layer
        self.linear2.SCALE = True                                                           # Flag to scale the weights of the second linear layer
        self.dropout = nn.Dropout(config.dropout)  

    def forward(self, x):   
        x = F.gelu(self.linear1(x))                                                         # Apply GELU activation to the first linear layer
        x = self.dropout(x)
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
        assert torch.any(scores != float("-inf")), "All scores are masked — attention will NaN"     # Check if all scores are masked
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
        self.norm1_layer = nn.LayerNorm(config.embedding_dim, eps=1e-5)             # Pre-normalization layer
        self.attention_layer = multiheadAttention(config)                           # Multihead attention layer
        self.norm2_layer = nn.LayerNorm(config.embedding_dim, eps=1e-5)             # Post-normalization layer
        self.mlp_layer = mlp(config)                                                # Feed forward neural network
    
    def forward(self, x):
        x = x + self.attention_layer(self.norm1_layer(x))                           # Residual connection through the attention
        x = x + self.mlp_layer(self.norm2_layer(x))                                 # Residual connection through the feed forward neural network
        return x


"""
Transformer model
"""
class transformer(nn.Module):
    def __init__(self, config: transformerConfig):
        super(transformer, self).__init__()
        self.config = config                                                                                            # Configuration for the transformer   

        self.token_embedding_layer = nn.Embedding(config.vocab_size, config.embedding_dim)                              # Token embedding layer
        self.positional_embedding_layer = nn.Embedding(config.max_seq_len, config.embedding_dim)                        # Positional embedding layer
        self.transformer_blocks = nn.ModuleList([transformerBlock(config) for _ in range(config.num_layers)])           # List of transformer blocks
        self.norm_layer = nn.LayerNorm(config.embedding_dim)                                                            # Normalization layer
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)                              # Output layer

        self.token_embedding_layer.weight = self.output_layer.weight                                                    # Tie the weights of the token embedding and output layer
    
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
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)                          # Initialize the weights of the embedding layer with normal distribution
        
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)                                           # Initialize the bias of the layer normalization layer with constant value 0
            torch.nn.init.constant_(module.weight, 1)                                         # Initialize the weights of the layer normalization layer with constant
        

    def forward(self, x, targets=None):
        batch_size, seq_len = x.size()                                                                                  # Get the batch size and sequence length
        assert seq_len <= self.config.max_seq_len, f"Input sequence length = {seq_len} exceeds the maximum sequence length = {self.config.max_seq_len}"
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).expand(batch_size, seq_len)             # Generate positions tensor for the input sequence
        positional_embedding = self.positional_embedding_layer(positions)                                               # Get positional embeddings for the input sequence (batch_size, seq_len, embedding_dim)
        token_embedding = self.token_embedding_layer(x)                                                                 # Get token embeddings for the input sequence (batch_size, seq_len, embedding_dim)
        x = token_embedding + positional_embedding                                                                      # Add token and positional embeddings 
        for block in self.transformer_blocks:                       
            x = block(x)                                                                                                # Pass through the transformer blocks
        x = self.norm_layer(x)                                                                                          # Apply normalization
        x = self.output_layer(x)                                                                                        # Apply output layer
        loss = None                                                                                                     # Initialize loss to None
        if targets is not None:
            logits = x.view(-1, x.size(-1))
            logits = torch.clamp(logits, min=-20, max=20)                                                               # Prevents extreme values
            loss = F.cross_entropy(logits, targets.view(-1))                                                            # Compute the loss using cross entropy      
        return logits, loss                                                                                             # Return the output(logits) and loss

    def init_optimizers(self, weight_decay, learning_rate, device):
        parameter_dict = {param_name : param for param_name, param in self.named_parameters() if param.requires_grad}   # Get the parameters that require gradients
        weight_decay_parameters = [param for name, param in parameter_dict.items() if param.dim() >= 2]                   # Get the parameters that require weight decay (only for bidiemensional tensors)
        no_weight_decay_parameters = [param for name, param in parameter_dict.items() if param.dim() < 2]                 # Get the parameters that do not require weight decay
        optim_param_group = [
            {"params": weight_decay_parameters, "weight_decay": weight_decay},
            {"params": no_weight_decay_parameters, "weight_decay": 0.0}
        ]
        decay_param_tensor_num = len(weight_decay_parameters)                                                           # Number of tensors that require weight decay
        decay_param_num = sum(param.numel() for param in weight_decay_parameters)                                       # Number of parameters that require weight decay
        no_decay_param_tensor_num = len(no_weight_decay_parameters)                                                     # Number of tensors that do not require weight decay
        no_decay_param_num = sum(param.numel() for param in no_weight_decay_parameters)                                 # Number of parameters that do not require weight decay
        
        print(f"number of decayed parameter tensors: {decay_param_tensor_num}, with {decay_param_num} parameters")
        print(f"number of non-decayed parameter tensors: {no_decay_param_tensor_num}, with {no_decay_param_num} parameters")
        
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_param_group, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer