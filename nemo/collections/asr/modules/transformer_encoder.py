from torch import nn
import torch
from torch.nn import Conv1d
from dataclasses import dataclass
from torch.nn import GELU as TorchGELU
from torch.nn.functional import scaled_dot_product_attention
# from flash_attn import flash_attn_func
@dataclass
class GPTConfig():
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: int = 0.1
    qkv_bias: bool = False

@dataclass
class TransformerEncoderConfig():
    n_mels: int = 80
    d_model: int = 512
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    causal_mask: bool = False

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


    def forward(self, x):
        
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
class MultiHeadAttentionWithSDPA(nn.Module):
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

        keys = keys.transpose(1,2) # BxHxnum_tokens,head_dim
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        
        dropout = 0 if self.training == False else self.dropout
        output = scaled_dot_product_attention(queries,keys,values, is_causal=self.causal_mask, dropout_p=dropout)

         # B xH x num_tokens x head_dim

        output = output.transpose(1,2) # Bxnum_tokens x Hx head_dim

        output = output.contiguous().view(B,num_tokens,self.d_out)

        output = self.out_proj(output)

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, qkv_bias=False, context_length=1024, num_heads=8, causal_mask=False):
        super().__init__()
        self.d_out = dim_out
        self.w_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dropout = nn.Dropout(dropout)
        self.causal_mask = causal_mask
        self.out_proj = nn.Linear(self.d_out, self.d_out)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1) if self.causal_mask else torch.zeros(context_length, context_length)
        )
        
        self.register_buffer(
            "cache_k", None, persistent=False
        )
        self.register_buffer(
            "cache_v", None, persistent=False
        )


    def forward(self, x, use_cache=False):
        
        B, num_tokens, d_in  = x.shape
        H = self.num_heads

        keys = self.w_key(x).view(B,num_tokens,H, self.head_dim) # Bxnum_tokens x Hx head_dim
        queries = self.w_query(x).view(B,num_tokens,H, self.head_dim)
        values = self.w_value(x).view(B,num_tokens,H, self.head_dim)

        if use_cache:
            if self.cache_k is None:
                self.cache_k  = keys
                self.cache_v = values
            else:
                self.cache_k = torch.cat((self.cache_k, keys), dim=1)
                self.cache_v = torch.cat((self.cache_v, values), dim=1)
            keys, values = self.cache_k, self.cache_v


        keys = keys.transpose(1,2) # BxHxnum_tokens,head_dim
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = torch.matmul(queries, keys.transpose(-1,-2)) # We need to transpose head_dim and num_tokens for keys. alpha = BxNxTqxTk
        d_k = keys.shape[-1]

        # Masking 
        T = attn_scores.shape[-1]
        mask = self.mask[:num_tokens, :num_tokens]
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

        attn_weights = torch.softmax(masked/d_k**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, values) # B xH x num_tokens x head_dim

        output = output.transpose(1,2) # Bxnum_tokens x Hx head_dim

        output = output.contiguous().view(B,num_tokens,self.d_out)

        output = self.out_proj(output)

        return output
    
    def reset_cache(self,):
        self.cache_k = None
        self.cache_v = None



class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.pre_norm = LayerNorm(self.cfg.d_model)
        self.mha = MultiHeadAttentionWithSDPA(
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

class TransformerEncoder(nn.Module):
    def __init__(self, 
                n_mels: int = 80,
                d_model: int = 512,
                n_heads: int = 8,
                n_layers: int = 17,
                drop_rate: float = 0.1,
                qkv_bias: bool = False,
                causal_mask: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.conv1 = Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1) # Decreases the temporal dimension by 2
        self.conv3 = Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1) # Decreases the temporal dimension by 2

        cfg = TransformerEncoderConfig(d_model=d_model, n_heads=n_heads, n_layers=n_layers, drop_rate=drop_rate, qkv_bias=qkv_bias, causal_mask=causal_mask)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(n_layers)])
        self.gelu = TorchGELU()
        self.layer_norm = LayerNorm(d_model)

    def forward(self, audio_signal, length): 
        x = audio_signal
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        length = length // 2
        x = self.conv3(x)
        x = self.gelu(x)
        x = x.transpose(1, 2) # BxCxT -> BxTxC
        x = self.layer_norm(x)
        length = length // 2
        for idx, layer in enumerate(self.layers):
            x = layer(x)
        x = x.transpose(1, 2) # BxTxC -> BxCxT
        return x, length
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self, partial=False):
        for param in self.parameters():
            param.requires_grad = True