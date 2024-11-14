import torch
import torch.nn as nn
import math
from typing import Optional

from src.utils.transformer_helper import mask_expansion, combine_encoding

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout_rate=0.1, need_weights=False):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            *([nn.Dropout(dropout_rate)] if dropout_rate > 0 else []),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.need_weights = need_weights
        self.heads = heads

    def forward(self, x, mask):
        # Pre-LN for attention
        norm_x = self.norm1(x)
        # value, key, query: (N, seq_length, embed_size), mask: (seq_length, seq_length)
        attention_out, _ = self.attention(norm_x, norm_x, norm_x, attn_mask=mask, need_weights=self.need_weights, average_attn_weights=False, is_causal=False) # (N, seq_length, embed_size)
        x = self.dropout(attention_out) + x if self.dropout else attention_out + x  # (N, seq_length, embed_size)

        # Pre-LN for feed-forward
        out = self.feed_forward(self.norm2(x)) + x  # (N, seq_length, embed_size)

        return out  # (N, seq_length, embed_size)
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, *, use_grid_encoder: bool = False, progressive_head: int = 1, max_length = 2048, dropout_rate=0.05, jupyter_debug=False):
        super().__init__()
        self.use_grid_encoder = use_grid_encoder
        assert embed_size % 8 == 0
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.max_grid_size = 64 # 30 is the max edge size from the data. add 1 for ROW_SEPARATOR and 1 for buffer, charts id requires more
        self.grid_embedding_size = embed_size // 4
        self.register_buffer("positional_encoding", self.generate_positional_encoding(max_length, embed_size))
        self.register_buffer("grid_encoding", self.generate_positional_encoding(self.max_grid_size, self.grid_embedding_size, True))

        assert progressive_head < 4 # it doesn't make sense
        self.layers = nn.ModuleList([TransformerBlock(embed_size, min(heads, 2 ** (layer_index // progressive_head)), dropout_rate=dropout_rate) for layer_index in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Add learnable grid_scale parameter
        self.grid_scale = nn.Parameter(torch.tensor([2.4002, 2.6549, 1.2614])) # previously tuned

        # self.embedding.weight = self.fc_out.weight # NOT GOOD! see special_checkpoints/weight_tying

        self.dropout = nn.Dropout(dropout_rate)
        self.heads = heads

    def activate_attention_weights_output(self, need_weights):
        for layer in self.layers:
            layer.need_weights = need_weights

    def set_dropout_rate(self, dropout_rate):
        """
        Set the dropout rate for all dropout layers in the model.
        
        Args:
            dropout_rate (float): New dropout probability between 0 and 1
        """
        # Update main dropout layer
        self.dropout.p = dropout_rate
        
        # Update dropout in transformer blocks
        for layer in self.layers:
            # Update dropout in feed forward network
            for module in layer.feed_forward:
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate
            
            # Update dropout after attention
            if layer.dropout is not None:
                layer.dropout.p = dropout_rate        

    def generate_positional_encoding(self, max_length, embed_size, reverse = False):
        pe = torch.zeros(max_length, embed_size) # (max_length, embed_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1) # (max_length, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)) #  (embed_size // 2,)
        pe[:, 0::2] = torch.sin(position * div_term) # (max_length, embed_size // 2)
        pe[:, 1::2] = torch.cos(position * div_term) # (max_length, embed_size // 2)
        if reverse:
            pe = torch.flip(pe, [1])  # Flip the positional encoding along the sequence dimension
        return pe.unsqueeze(0) # (1, max_length, embed_size)
    
    def forward(self, x, mask):
        batch_size, seq_length, cell_size = x.shape  # x: (N, seq_length, 5)
        assert cell_size == 5 

        combined_encodings = combine_encoding(x, batch_size, self.use_grid_encoder, seq_length, self.max_grid_size, self.grid_scale, self.grid_encoding, self.positional_encoding)
            
        initial_tensor = self.embedding(x[:, :, 0]) + combined_encodings # (N, seq_length, embed_size)

        x = self.dropout(initial_tensor)  # (N, seq_length, embed_size)

        if mask.dim() == 3: # per sample mask in the batch
            mask = mask_expansion(mask, self.heads)

        for layer in self.layers:
            # x = layer(x, mask[:layer.heads * batch_size, ...])  # The real bug, shitty bug
            mask_slice_ending_index = self.heads * batch_size
            mask_slice_step = self.heads // layer.heads
            x = layer(x, mask[:mask_slice_ending_index:mask_slice_step, ...])  # (N, seq_length, embed_size)


        out = self.fc_out(x)  # (N, seq_length, vocab_size)
        return out  # (N, seq_length, vocab_size)
        
