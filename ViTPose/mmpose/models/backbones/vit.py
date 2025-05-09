# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

#base
start_block=2 
end_block=11
start_ratio=0
end_ratio=0.5

"""
# Large
start_block=4
end_block=24-1
start_ratio=0
end_ratio=0.5
"""


# Huge
"""
start_block=5
end_block=32-1
start_ratio=0
end_ratio=0.5
"""
def get_pruning_ratio(this_block, start_block, end_block, start_ratio, end_ratio):
    return start_ratio + ((this_block - start_block)*(start_ratio - end_ratio)/(start_block - end_block))

def softmax_taylor_approximation(x, dim=-1, terms=4):
    """
    Approximates softmax using Taylor series expansion up to specified number of terms.
    
    Args:
        x: Input tensor of attention scores (after q @ k.transpose(-2, -1))
        dim: Dimension along which to compute softmax (default: -1)
        terms: Number of terms in Taylor series expansion (default: 4)
        
    Returns:
        Tensor with approximated softmax applied
    """
    # Step 1: Shift for numerical stability (similar to the regular softmax)
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    x_shifted = x
    
    # Step 2: Compute Taylor series expansion for exp(x)
    # exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4!
    
    # Initialize with first term
    taylor_exp = torch.ones_like(x_shifted)
    
    # Add x term (2nd term)
    taylor_exp = taylor_exp + x_shifted
    
    # Add x²/2! term (3rd term)
    taylor_exp = taylor_exp + (x_shifted**2) / 2
    
    # Add x³/3! term (4th term)
    taylor_exp = taylor_exp + (x_shifted**3) / 6
    
    # Replace any negative values with a small positive number
    # This ensures we don't have negative "probabilities"
    taylor_exp = torch.clamp(taylor_exp, min=1e-10)
    
    # Step 3: Normalize to get a probability distribution
    return taylor_exp / torch.sum(taylor_exp, dim=dim, keepdim=True)

def base2_softmax(x, dim=-1):
    """
    Implements softmax using base 2 instead of base e.
    
    Standard softmax: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    Base-2 softmax: softmax(x_i) = 2^(x_i) / Σ 2^(x_j)
    
    Args:
        x: Input tensor of attention scores
        dim: Dimension along which to compute softmax
    
    Returns:
        Tensor with base-2 softmax applied
    """
    # Shift for numerical stability (same as in regular softmax)
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    
    # Apply 2^x instead of e^x
    # We can compute 2^x as 2^x = e^(x * ln(2))
    # But for better numerical precision, we use the direct power operation
    power_of_2 = torch.pow(2.0, x_shifted)
    print(f"\n\n\n numerator shape: {power_of_2.shape}")

    denominator = torch.sum(power_of_2, dim=dim, keepdim=True)
    print(f"denominator shape: {denominator.shape}")

    out = power_of_2 / denominator
    print(f"out shape: {out.shape}")
    # Normalize to get a probability distribution
    return power_of_2 / denominator

def tensor_approximator(x):
    """
    Approximates each element in x to the form m*2^n,
    where 1 ≤ m < 2 and m is approximated using 3 most significant bits.
    """
    # Handle non-positive values by using a small positive value
    epsilon = 1e-10
    positive_x = torch.clamp(x, min=epsilon)

    
    # Calculate the exponent n where 1 ≤ x*2^(-n) < 2
    log2_x = torch.log2(positive_x)
    n_tensor = torch.floor(log2_x).int()
    
    # Calculate mantissa m where x = m*2^n
    m_tensor = positive_x / (2.0 ** n_tensor)
    
    # Ensure 1 ≤ m < 2 (handling potential precision issues)
    correction = (m_tensor >= 2.0).float()
    m_tensor = m_tensor / (2.0 ** correction)
    n_tensor = n_tensor + correction.int()
    
    # Handle m < 1 (due to numerical issues)
    correction = (m_tensor < 1.0).float()
    m_tensor = m_tensor * (2.0 ** correction)
    n_tensor = n_tensor - correction.int()
    
    # Extract the fractional part
    fractional_part = m_tensor - 1.0
    
    # Calculate the 3-bit approximation
    # 1st bit: 0.5 (2^-1)
    bit1 = (fractional_part >= 0.5).float() * 0.5
    remaining1 = fractional_part - bit1
    
    # 2nd bit: 0.25 (2^-2) 
    bit2 = (remaining1 >= 0.25).float() * 0.25
    remaining2 = remaining1 - bit2
    
    # 3rd bit: 0.125 (2^-3)
    bit3 = (remaining2 >= 0.125).float() * 0.125
    
    # Combine to get the 3-bit approximation of m
    m_approx = 1.0 + bit1 + bit2 + bit3
    
    # Return approximated values
    return m_approx * (2.0 ** n_tensor)

def base2_softmax_approx(x, dim=-1):
    """
    Implements softmax using base 2 instead of base e.
    
    Standard softmax: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    Base-2 softmax: softmax(x_i) = 2^(x_i) / Σ 2^(x_j)
    
    Args:
        x: Input tensor of attention scores
        dim: Dimension along which to compute softmax
    
    Returns:
        Tensor with base-2 softmax applied
    """

    # Shift for numerical stability (same as in regular softmax)
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    
    # Apply 2^x instead of e^x
    # We can compute 2^x as 2^x = e^(x * ln(2))
    # But for better numerical precision, we use the direct power operation
    power_of_2 = torch.pow(2.0, x_shifted)
    # print(f"\n\n\n numerator shape: {power_of_2.shape}")



    denominator = torch.sum(power_of_2, dim=dim, keepdim=True)
    # print(f"denominator shape: {denominator.shape}")
    approx_denominator = tensor_approximator(denominator)
    # print(f"approx_denominator shape: {approx_denominator.shape}")
    out = power_of_2 / approx_denominator
    out = out / out.sum(dim=-1, keepdim=True).clamp(min=1e-10) # Ensure sum is 1, got changed due to approximation

    # print(f"out shape: {out.shape}\n\n\n")

    # Normalize to get a probability distribution
    return out
'''
def squaremax(x, dim=-1):
    """
    Implements a "squaremax" function as an alternative to softmax.
    
    Instead of using exponentiation, squaremax squares the input values
    and then normalizes them to form a probability distribution.
    
    Args:
        x: Input tensor of attention scores
        dim: Dimension along which to compute squaremax
    
    Returns:
        Tensor with squaremax applied
    """
    # Shift for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    
    # Use squared values instead of exponential
    # Ensure all values are >= 0 before squaring
    x_relu = torch.relu(x_shifted)  # Zero out negative values
    print(f"\n\n\n\ x_relu shape : {x_relu.shape} ")
    squares = x_relu ** 2
    print(f"squares shape : {squares.shape} ")

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-5
    sum_squares = torch.sum(squares, dim=dim, keepdim=True) + epsilon
    
    print(f"denominator shape : {sum_squares.shape} \n\n\n")

    # Normalize to get a probability distribution
    return squares / sum_squares
    
def represent_as_m_times_2_to_n(num):
    """
    Represents a positive number in the form m*2^n where 1 ≤ m < 2,
    and approximates m using the three most significant bits after the binary point.
    
    Args:
        num: A positive number to be represented
        
    Returns:
        tuple: (m, n, m_approx, binary_m, approx_binary)
            - m: The mantissa (1 ≤ m < 2)
            - n: The exponent
            - m_approx: Approximation of m using 3 significant bits
            - binary_m: Binary representation of m
            - approx_binary: Approximated binary representation "1.abc"
    """
    if num <= 0:
        raise ValueError("Input must be a positive number")
    
    # Find n such that 1 ≤ num*2^(-n) < 2
    n = math.floor(math.log2(num))
    
    # Calculate m where num = m*2^n
    m = num / (2**n)
    
    # Ensure 1 ≤ m < 2
    if m >= 2:
        m /= 2
        n += 1
    elif m < 1:
        m *= 2
        n -= 1
    
    # Convert m to binary representation
    # Subtract 1 to get just the fractional part (since 1 ≤ m < 2)
    fractional_part = m - 1
    
    # Get binary representation of the fractional part
    binary_fractional = ""
    current = fractional_part
    for _ in range(20):  # Get 20 bits for display purposes
        current *= 2
        bit = int(current)
        binary_fractional += str(bit)
        current -= bit
        if current == 0:
            break
    
    # Full binary representation of m
    binary_m = "1." + binary_fractional
    
    # Extract the first 3 bits for approximation
    first_three_bits = binary_fractional[:3].ljust(3, '0')
    approx_binary = "1." + first_three_bits
    
    # Calculate the approximated m value
    m_approx = 1
    for i, bit in enumerate(first_three_bits):
        if bit == '1':
            m_approx += 2**-(i+1)
    reconstructed = m_approx * (2**n)
    return reconstructed

def squaremax_as_multiplication(x, dim=-1):
    """
    Implements a "squaremax" function as an alternative to softmax.
    
    Instead of using exponentiation, squaremax squares the input values
    and then normalizes them to form a probability distribution.
    
    Args:
        x: Input tensor of attention scores
        dim: Dimension along which to compute squaremax
    
    Returns:
        Tensor with squaremax applied
    """
    # Shift for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    
    # Use squared values instead of exponential
    # Ensure all values are >= 0 before squaring
    x_relu = torch.relu(x_shifted)  # Zero out negative values
    squares = x_relu ** 2
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-5
    sum_squares = torch.sum(squares, dim=dim, keepdim=True) + epsilon
    print(f"\n\n\nshape of (sum_squares): {sum_squares.shape}\n\n\n")
    print(a)
    sum_squares = represent_as_m_times_2_to_n(sum_squares)
    # Normalize to get a probability distribution
    return squares / sum_squares
'''
def pad_with_indices(pruned_tensor, token_indices, n2_tokens):
    """
    Args:
        pruned_tensor: (batch_size, n1_tokens, channels)
        token_indices: (batch_size, n1_tokens) containing target positions
        n2_tokens: final number of tokens (must be >= max indices + 1)
    
    Returns:
        padded_tensor: (batch_size, n2_tokens, channels)
    """
    batch_size, n1_tokens, channels = pruned_tensor.shape
    
    # Initialize output tensor with zeros
    padded_tensor = torch.zeros(batch_size, n2_tokens, channels,
                            dtype=pruned_tensor.dtype,
                            device=pruned_tensor.device)
    
    # Create batch indices [0,0,...0, 1,1,...1, ..., B-1,B-1,...B-1]
    batch_idx = torch.arange(batch_size)[:, None].expand(-1, n1_tokens)
    
    # Scatter pruned tokens into specified positions
    padded_tensor[batch_idx, token_indices] = pruned_tensor
    
    return padded_tensor

def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos
    
    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
 
        # attn = attn.softmax(dim=-1) # Orig softmax)
        

        # attn = base2_softmax(attn, dim=-1) #radix-2 softmax
        attn = base2_softmax_approx(attn, dim=-1)

        attn_weights = attn.clone()
        attn_weights = attn_weights.sum(dim=1).sum(dim=1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_weights

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_attn, attn_weights = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_weights


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@BACKBONES.register_module()
class ViT(BaseBackbone):

    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 ):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}




    def forward_features(self, x):
        B, C, H, W = x.shape

        # # print(f"\n\nx shape: {B, C, H, W}")
        
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        batch_size , n_tokens = x.shape[:-1]
        cumulative_scores = torch.zeros(batch_size , n_tokens).to(x.device)
        cumulative_scores_pruned = torch.zeros(batch_size , n_tokens).to(x.device)

        # print(f"\n\n\n cumulative_scores shape: {cumulative_scores.shape}\n\n\n")
        # print(f"x shape: {x.shape}")
        pruned = False
        n_tokens_retain_last = x.shape[1]
        for block_number, blk in enumerate(self.blocks):
            """    
            if block_number == 2:
                keep_tokens = [i for i in range(7)]
                x = x[: , keep_tokens, :]

            """
            # print(f"block_number, x shape: {block_number, x.shape}")
            if block_number >= start_block:
                n_tokens_retain = int((1 - get_pruning_ratio(block_number, start_block, end_block, start_ratio, end_ratio))*n_tokens)
                # n_tokens_retain = 150
                if n_tokens_retain < n_tokens_retain_last:
                    n_tokens_retain_last = n_tokens_retain
                    # # print(f"attn_weights shape: {block_number, attn_weights.shape}")
                    # # print(f"attn_weights : {block_number, attn_weights}")
                    _, indices = torch.topk(cumulative_scores, k=n_tokens_retain, dim=1)
                    # indices, _ = torch.sort(indices) 
                    cumulative_scores_pruned, retained_indices_for_x = torch.topk(cumulative_scores_pruned, k=n_tokens_retain, dim=1)
                    # retained_indices_for_x, _ = torch.sort(retained_indices_for_x) 
                    # x = x[: , top_k_values_indices[1], :]
                    ## print(f"indices: {indices}")
                    x = x[torch.arange(B)[:, None], retained_indices_for_x]
                    pruned = True


    
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn_weights = blk(x)
            

            if pruned:
                batch_idx = torch.arange(B)[:, None]
                cumulative_scores[batch_idx, indices] += attn_weights
            
            else:
                cumulative_scores += attn_weights
            
            cumulative_scores_pruned += attn_weights
            # print(f"cumulative_scores: {cumulative_scores.shape, cumulative_scores}")
            """
            print(f"cumulative_scores: {cumulative_scores.shape, cumulative_scores}")
            print(f"top 2 cumulative_scores: {torch.topk(cumulative_scores, k=2, dim=1)}")
            
            print(f"cumulative_scores_pruned: {cumulative_scores_pruned.shape, cumulative_scores_pruned}")
            """

            # print(f"block_number, x shape: {block_number, x.shape}")

        # print(a)

        x = pad_with_indices(x, indices, n_tokens)
        # print(f"\n\nx.shape, x: {x.shape, x}")
            
        # print(f"\n\n\nIndices retained finally: \n{indices.shape, indices}")

        # Check if ANY channel is non-zero for each token
        non_zero_mask = (x != 0).any(dim=2)  # shape: (batch_size, n_tokens)

        # For each batch, get indices of tokens with at least one non-zero channel
        non_zero_token_indices = [
            torch.nonzero(non_zero_mask[b], as_tuple=False).squeeze(1)
            for b in range(x.size(0))
        ]

        # print(f"\n\n\nNon-zero indices: \n{non_zero_token_indices}")


        # print(a)

        x = self.last_norm(x)

        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()