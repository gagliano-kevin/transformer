from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

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
        self.config = config                                                                                    # Configuration for the transformer   

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)                            # Token embedding layer
        self.positional_embedding = nn.Parameter(torch.randn(config.max_seq_len, config.embedding_dim))         # Positional embedding layer
        self.transformer_blocks = nn.ModuleList([transformerBlock(config) for _ in range(config.num_layers)])   # List of transformer blocks
        self.norm_layer = nn.LayerNorm(config.embedding_dim)                                                    # Normalization layer
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)                      # Output layer

        self.token_embedding.weight = self.output_layer.weight                                                  # Tie the weights of the token embedding and output layer
    
        self.apply(self.init_weights)                                                                           # Initialize the weights of the model
    
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

    def forward(self, x):
        batch_size, seq_len = x.size()                                                          # Get the batch size and sequence length
        assert seq_len <= self.config.max_seq_len, f"Input sequence length = {seq_len} exceeds the maximum sequence length = {self.config.max_seq_len}"

        #---
        x = self.token_embedding(x) + self.positional_embedding[:seq_len]                       # Add token and positional embeddings
        for block in self.transformer_blocks:
            x = block(x)                                                                        # Pass through the transformer blocks
        x = self.norm_layer(x)                                                                  # Apply normalization