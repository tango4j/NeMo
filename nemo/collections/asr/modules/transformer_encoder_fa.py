from torch import nn
import torch
from torch.nn import Conv1d
from dataclasses import dataclass
from torch.nn import GELU as TorchGELU
from flash_attn import flash_attn_func
@dataclass
class GPTConfig():
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: int = 0.1
    qkv_bias: bool = False
    theta_base: int = 10_000

@dataclass
class TransformerEncoderConfig():
    n_mels: int = 80
    d_model: int = 512
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    causal_mask: bool = False
    theta_base: int = 10_000
    context_length: int = 4096

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            TorchGELU(),
            nn.Linear(4*dim, dim)
        )

    def forward(self, x):
        return self.ffn(x)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased = False)
        with torch.autocast('cuda', dtype=torch.float32):
            norm = (x-mean)/ torch.sqrt(var + self.eps)
            output = self.scale * norm + self.shift

        return output

class MultiHeadAttentionWithFA(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, qkv_bias=False, context_length=1024, num_heads=8, causal_mask=False):
        super().__init__()
        self.d_out = dim_out
        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dropout = dropout
        self.causal_mask = causal_mask
        self.out_proj = nn.Linear(self.d_out, self.d_out)


    def forward(self, x, use_cache=False):
        
        B, num_tokens, d_in  = x.shape
        H = self.num_heads

        keys = self.w_key(x).view(B,num_tokens,H, self.head_dim) # Bxnum_tokens x Hx head_dim
        queries = self.w_query(x).view(B,num_tokens,H, self.head_dim)
        values = self.w_value(x).view(B,num_tokens,H, self.head_dim)
        
        dropout = 0 if self.training == False else self.dropout        
        output = flash_attn_func(queries,keys,values, dropout_p=dropout, causal=self.causal_mask)

        # Bxnum_tokens x Hx head_dim

        output = output.contiguous().view(B,num_tokens,self.d_out)

        output = self.out_proj(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_norm = LayerNorm(self.cfg.d_model)
        self.mha = MultiHeadAttentionWithFA(
            dim_in=self.cfg.d_model, 
            dim_out=self.cfg.d_model, 
            dropout=self.cfg.drop_rate, 
            qkv_bias=self.cfg.qkv_bias, 
            num_heads=self.cfg.n_heads,
            causal_mask=self.cfg.causal_mask
            )
        self.dropout = nn.Dropout(self.cfg.drop_rate)
        self.post_norm = LayerNorm(self.cfg.d_model)
        self.ffn = FeedForward(self.cfg.d_model)

    def forward(self, x, use_cache=False):
        pre_norm = self.pre_norm(x)

        attn_output = self.mha(pre_norm, use_cache=use_cache)
        attn_output = x + self.dropout(attn_output)

        post_norm = self.post_norm(attn_output)
        ffn = self.ffn(post_norm)
        output = attn_output + self.dropout(ffn)
        return output

class ConvSubsampling(nn.Module):
    def __init__(self, n_mels: int = 80, d_model: int = 512):
        super().__init__()
        self.conv1 = Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1) # Decreases the temporal dimension by 2
        self.conv3 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1) # Decreases the temporal dimension by 2
        self.gelu = TorchGELU()

    def forward(self, x, length):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)
        length = length // 2
        x = self.conv3(x)
        x = self.gelu(x)
        length = length // 2
        return x, length

class NGPTStackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate
    Args:
        subsampling_factor (int): The subsampling factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
    """

    def __init__(
        self,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        use_bias: bool = False
    ):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out, bias=use_bias)
        self.pad_frame = nn.Parameter(torch.ones(feat_in, dtype=torch.float32))

    def forward(self, x, length):
        """
        Args:
            x (torch.Tensor): (B, C, T)
            length (torch.Tensor): (B,)
        Returns:
            x (torch.Tensor): (B, T', D_model)
            length (torch.Tensor): (B,)
        """
        x = x.transpose(1, 2) # BxCxT -> BxTxC
        b, t, h = x.size()
        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        length = torch.div(length + pad_size, self.subsampling_factor, rounding_mode='floor')

        # Pad and fill padding frames (all-zero) with a learnable padding 'embedding'
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        x[(x == 0).all(dim=-1)] = self.pad_frame

        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = self.proj_out(x)

        return x, length
class TransformerEncoderWithFA(nn.Module):
    def __init__(self, 
                n_mels: int = 80,
                d_model: int = 512,
                n_heads: int = 8,
                n_layers: int = 17,
                drop_rate: float = 0.1,
                qkv_bias: bool = False,
                causal_mask: bool = False,
                pre_encode: str = "conv", # "conv" or "stacking"
    ):
        super().__init__()
        self.d_model = d_model
        if pre_encode == "conv":
            self.pre_encode = ConvSubsampling(n_mels, d_model)
        elif pre_encode == "stacking":
            self.pre_encode = NGPTStackingSubsampling(subsampling_factor=4, feat_in=n_mels, feat_out=d_model)
        else:
            raise ValueError(f"Invalid pre_encode: {pre_encode}")

        cfg = TransformerEncoderConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, drop_rate=drop_rate, qkv_bias=qkv_bias, causal_mask=causal_mask)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, audio_signal, length): 
        """
        Args:
            audio_signal (torch.Tensor): (B, C, T)
            length (torch.Tensor): (B,)
        Returns:
            x (torch.Tensor): (B, T', D_model)
            length (torch.Tensor): (B,)
        """
        x = audio_signal
        x, length = self.pre_encode(x, length)
        x = self.layer_norm(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        x = x.transpose(1, 2) # BxT'xD_model -> BxD_modelxT'
        return x, length
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self, partial=False):
        for param in self.parameters():
            param.requires_grad = True


