import math
import torch
import torch.nn.functional as F
from torch import nn


# --- Utility Functions ---
def trunc_normal_init_(
    tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0
):
    """
    Truncated normal initialization (LeCun Normal, truncated at 2 stddev).
    Used for all weights, including initial hidden states z_H0, z_L0.
    Paper: "The initial hidden states z_0 are initialized by sampling from a truncated normal distribution..."
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


def rms_norm(x: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    RMSNorm: Post-Norm normalization for stability and bounded parameters.
    Used in all transformer blocks.
    Paper: "Our model satisfies these conditions through its Post-Norm architecture that employs RMSNorm..."
    """
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + epsilon)


# --- Input Network (f_I) ---
class Embedding(nn.Module):
    """
    Input embedding network f_I(x; θ_I).
    Converts discrete tokens to vector representations.
    Paper: "The model includes an embedding layer f_I that converts discrete tokens into vector representations..."
    """

    def __init__(self, vocab_size, dim, init_std=1.0):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dim)
        with torch.no_grad():
            trunc_normal_init_(self.embeddings.weight, init_std)

    def forward(self, x):
        return self.embeddings(x)


# --- Output Network (f_O) ---
class Linear(nn.Module):
    """
    Output head f_O(z; θ_O) = softmax(θ_O z).
    Used for token prediction and Q-head for ACT.
    Paper: "An output head f_O(z; θ_O) = softmax(θ_O z) transforms hidden states into token probability distributions..."
    """

    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        with torch.no_grad():
            trunc_normal_init_(self.linear.weight, std=1.0 / math.sqrt(in_dim))
            if bias:
                self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x)


# --- Rotary Positional Encoding ---
class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Encoding (RoPE).
    Provides relative position info for attention, replacing learned position embeddings.
    Paper: "These improvements include Rotary Positional Encoding..."
    """

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
    """
    Multi-head attention with rotary positional encoding.
    Used in both low-level and high-level modules.
    Implements QKV projections, grouped query attention, and output projection.
    Paper: "Both low-level and high-level recurrent modules f_L and f_H are implemented using encoder-only Transformer blocks..."
    """

    def __init__(self, dim, head_dim, num_heads, key_value_heads_per_head=1):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.key_value_heads_per_head = key_value_heads_per_head
        self.num_key_value_heads = num_heads * key_value_heads_per_head
        self.qkv_proj = Linear(
            dim, (num_heads + 2 * self.num_key_value_heads) * head_dim, bias=False
        )
        self.out_proj = Linear(head_dim * num_heads, dim, bias=False)

    def forward(self, x, rotary_emb=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x).view(
            batch_size,
            seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]
        if rotary_emb is not None:
            query = rotary_emb(query)
            key = rotary_emb(key)
        # TODO: Implement grouped query attention logic as in Swift
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        attn_weights = torch.softmax(attn_logits, dim=-1)
        combined = torch.matmul(attn_weights, value)
        combined = combined.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        return self.out_proj(combined)


# --- SwiGLU Activation ---
class SwiGLU(nn.Module):
    """
    SwiGLU: Gated Linear Unit activation for MLP blocks.
    Used in all transformer blocks for non-linear feature transformation.
    Paper: "These improvements include ... Gated Linear Units ..."
    """

    def __init__(self, dim, expansion=4.0):
        super().__init__()
        inter_dim = int(dim * expansion)
        self.gate_proj = Linear(dim, inter_dim * 2, bias=False)
        self.down_proj = Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        gate, up = torch.chunk(self.gate_proj(x), 2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
