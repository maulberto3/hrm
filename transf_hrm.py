from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

# --- Utility Functions ---
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Optional, Tuple

def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    """Truncated normal initialization (JAX/Flax style)."""
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor

def rms_norm(x: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """Root mean square normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + epsilon)

# --- Embedding Layer ---
class Embedding(nn.Module):
    """Token embedding layer."""
    def __init__(self, vocab_size, dim, init_std=1.0):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dim)
        with torch.no_grad():
            trunc_normal_init_(self.embeddings.weight, init_std)
    def forward(self, x):
        return self.embeddings(x)

# --- Linear Layer ---
class Linear(nn.Module):
    """Linear layer with truncated normal initialization."""
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        with torch.no_grad():
            trunc_normal_init_(self.linear.weight, std=1.0 / math.sqrt(in_dim))
            if bias:
                self.linear.bias.zero_()
    def forward(self, x):
        return self.linear(x)

# --- Rotary Embedding ---
class RotaryEmbedding(nn.Module):
    """Rotary position embedding."""
    def __init__(self, dim, max_length=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_length).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())
    def forward(self, x):
        # x shape: [batch, seq_len, num_heads, head_dim]
        seq_len = x.size(1)
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        return (x * cos) + (self.rotate_half(x) * sin)
    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

# --- Attention Layer ---
class Attention(nn.Module):
    """Multi-head attention with rotary embedding."""
    def __init__(self, dim, head_dim, num_heads, key_value_heads_per_head=1):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.key_value_heads_per_head = key_value_heads_per_head
        self.num_key_value_heads = num_heads * key_value_heads_per_head
        self.qkv_proj = Linear(dim, (num_heads + 2 * self.num_key_value_heads) * head_dim, bias=False)
        self.out_proj = Linear(head_dim * num_heads, dim, bias=False)
    def forward(self, x, rotary_emb=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]
        if rotary_emb is not None:
            query = rotary_emb(query)
            key = rotary_emb(key)
        # TODO: Implement grouped query attention logic as in Swift
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        combined = torch.matmul(attn_weights, value)
        combined = combined.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        return self.out_proj(combined)

# --- SwiGLU ---
class SwiGLU(nn.Module):
    """SwiGLU activation block."""
    def __init__(self, dim, expansion=4.0):
        super().__init__()
        inter_dim = int(dim * expansion)
        self.gate_proj = Linear(dim, inter_dim * 2, bias=False)
        self.down_proj = Linear(inter_dim, dim, bias=False)
    def forward(self, x):
        gate, up = torch.chunk(self.gate_proj(x), 2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

# --- Model Config ---
class ModelConfig:
    """Configuration for the hierarchical reasoning model."""
    def __init__(self, seq_len, vocab_size, high_level_cycles, low_level_cycles, num_layers, hidden_size, num_heads, expansion, norm_epsilon=1e-5, rope_theta=10000.0, halt_max_steps=16, halt_exploration_prob=0.1):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.high_level_cycles = high_level_cycles
        self.low_level_cycles = low_level_cycles
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.expansion = expansion
        self.norm_epsilon = norm_epsilon
        self.rope_theta = rope_theta
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob

# --- Reasoning Block ---
class ReasoningBlock(nn.Module):
    """A single reasoning block: attention, MLP, normalization."""
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(
            dim=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads
        )
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm_epsilon = config.norm_epsilon
    def forward(self, x, rotary_emb=None):
        x = rms_norm(x + self.attention(x, rotary_emb), epsilon=self.norm_epsilon)
        x = rms_norm(x + self.mlp(x), epsilon=self.norm_epsilon)
        return x

# --- Reasoner Module ---
class ReasonerModule(nn.Module):
    """Stacked reasoning blocks."""
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            ReasoningBlock(config)
            for _ in range(config.num_layers)
        ])
    def forward(self, hidden_state, input_injection, rotary_emb=None):
        # Ensure hidden_state matches input_injection shape for addition
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(1).expand(-1, input_injection.size(1), -1)
        elif hidden_state.dim() == 4:
            # If 4D, reshape to 3D by collapsing extra dimensions
            hidden_state = hidden_state.squeeze(1)
        elif hidden_state.shape[1] != input_injection.shape[1]:
            hidden_state = hidden_state.unsqueeze(1).expand(-1, input_injection.size(1), -1)
        x = hidden_state + input_injection
        for block in self.blocks:
            x = block(x, rotary_emb)
        return x

# --- Hierarchical Reasoner Model ---
class HierarchicalReasonerModel(nn.Module):
    """Hierarchical Reasoning Model with ACT halting logic."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(trunc_normal_init_(torch.empty(config.hidden_size), std=1.0 / math.sqrt(config.hidden_size)))
        self.input_embedding = Embedding(config.vocab_size, config.hidden_size, init_std=1.0 / math.sqrt(config.hidden_size))
        self.output_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.q_act_head = Linear(config.hidden_size, 2, bias=True)
        with torch.no_grad():
            self.q_act_head.linear.weight.zero_()
            self.q_act_head.linear.bias.fill_(-5)
        self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.seq_len + 1, config.rope_theta)
        self.high_level_reasoner = ReasonerModule(config)
        self.low_level_reasoner = ReasonerModule(config)
        self.register_buffer('H_init', trunc_normal_init_(torch.empty(config.hidden_size), std=1))
        self.register_buffer('L_init', trunc_normal_init_(torch.empty(config.hidden_size), std=1))

    def initial_hidden_states(self, batch_size):
        device = self.cls_token.device
        return {
            'high_level': self.H_init.unsqueeze(0).repeat(batch_size, 1),
            'low_level': self.L_init.unsqueeze(0).repeat(batch_size, 1)
        }

    def encode_inputs(self, inputs):
        """Encode input tokens and prepend CLS token."""
        batch_size = inputs.size(0)
        input_emb = torch.cat([
            self.cls_token.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1),
            self.input_embedding(inputs)
        ], dim=1) * math.sqrt(self.config.hidden_size)
        return input_emb

    def forward(self, hidden_states, inputs):
        """Run hierarchical reasoning cycles and compute outputs."""
        input_emb = self.encode_inputs(inputs)
        high_level_state = hidden_states['high_level']
        low_level_state = hidden_states['low_level']

        # Fix: expand high_level_state to match input_emb shape
        batch_size, seq_len_plus1, hidden_size = input_emb.shape

        # Run reasoning cycles
        for cycle in range(self.config.high_level_cycles * self.config.low_level_cycles - 1):
            low_level_state = self.low_level_reasoner(
                low_level_state, high_level_state.unsqueeze(1).expand(-1, seq_len_plus1, -1) + input_emb, rotary_emb=self.rotary_emb
            )
            if (cycle + 1) % self.config.low_level_cycles == 0:
                high_level_state = self.high_level_reasoner(
                    high_level_state, low_level_state, rotary_emb=self.rotary_emb
                )

        # Final step
        low_level_state = low_level_state.detach()
        high_level_state = high_level_state.detach()
        low_level_state = self.low_level_reasoner(
            low_level_state, high_level_state.unsqueeze(1).expand(-1, seq_len_plus1, -1) + input_emb, rotary_emb=self.rotary_emb
        )
        high_level_state = self.high_level_reasoner(
            high_level_state, low_level_state, rotary_emb=self.rotary_emb
        )

        output_logits = self.output_head(high_level_state[:, 1:])
        q_act_logits = self.q_act_head(high_level_state[:, 0])
        return {
            'hidden_states': {
                'high_level': high_level_state.detach(),
                'low_level': low_level_state.detach()
            },
            'output': output_logits,
            'q_act_halt': q_act_logits[:, 0],
            'q_act_continue': q_act_logits[:, 1]
        }

def generate_reasoning_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """
    Autoregressive generation for hierarchical reasoning model.
    Args:
        model: HierarchicalReasonerModel instance
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input string
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
    Returns:
        Decoded output string
    """
    model.eval()
    device = next(model.parameters()).device

    # Encode the prompt
    input_ids = torch.tensor(tokenizer.encode(prompt).ids).unsqueeze(0).to(device)
    batch_size = input_ids.size(0)
    hidden_states = model.initial_hidden_states(batch_size)

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(hidden_states, input_ids)
            next_token_logits = outputs['output'][:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            hidden_states = outputs['hidden_states']
            if next_token[0].item() == tokenizer.token_to_id("[eos]"):
                break

    return tokenizer.decode(input_ids[0].tolist())