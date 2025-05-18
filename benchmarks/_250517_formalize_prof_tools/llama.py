import torch
import torch.nn as nn
import torch.nn.functional as F

class LlamaLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        # parallelism
        tp: int,
        cp: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.tp = tp
        self.cp = cp

        # Attention layers
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim // tp, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim // tp, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim // tp, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim // tp, hidden_size, bias=False)

        # MLP layers
        self.gate_proj = nn.Linear(hidden_size, intermediate_size // tp, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size // tp, bias=False)
        self.down_proj = nn.Linear(intermediate_size // tp, hidden_size, bias=False)

        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, hidden_states, attention_mask=None):
        # Input layernorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention computation
        query_states = query_states.view(-1, self.num_attention_heads // self.tp, self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads // self.tp, self.head_dim)
        value_states = value_states.view(-1, self.num_key_value_heads // self.tp, self.head_dim)

        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=True
        )

        # Project back to hidden size
        attn_output = attn_output.reshape(hidden_states.shape[0], -1)
        attn_output = self.o_proj(attn_output)
        hidden_states = residual + attn_output

        # Post attention layernorm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        gate_output = F.silu(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        mlp_output = gate_output * up_output
        mlp_output = self.down_proj(mlp_output)

        hidden_states = residual + mlp_output

        return hidden_states

