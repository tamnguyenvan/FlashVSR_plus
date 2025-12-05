import mlx.core as mx
import mlx.nn as nn
import numpy as np

def sinusoidal_embedding_1d(dim, position):
    # position: (N,)
    half_dim = dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = mx.exp(mx.arange(half_dim) * -emb)
    emb = position[:, None] * emb[None, :]
    emb = mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    return emb

def precompute_freqs_cis(dim, end=1024, theta=10000.0):
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2) / dim))
    t = mx.arange(end)
    freqs = mx.outer(t, freqs) # (end, dim/2)
    # MLX doesn't support complex polar efficiently yet for RoPE in the same way, 
    # we return cos/sin for explicit rotation
    return mx.cos(freqs), mx.sin(freqs)

def apply_rope(x, cos, sin):
    # x: (B, L, num_heads, head_dim)
    # cos, sin: (L, head_dim/2)
    
    # Split into real/imag components (pairs)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    
    # Reshape freqs to broadcast
    # cos: (L, D/2) -> (1, L, 1, D/2)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    
    # Rotate
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    
    # Interleave back
    y = mx.stack([y0, y1], axis=-1)
    y = mx.reshape(y, x.shape)
    return y

def window_partition(x, window_size):
    # x: (B, F, H, W, C)
    B, F, H, W, C = x.shape
    wf, wh, ww = window_size
    
    x = x.reshape(B, F//wf, wf, H//wh, wh, W//ww, ww, C)
    # Permute to (B, nF, nH, nW, wF, wH, wW, C)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    # Flatten windows
    x = x.reshape(-1, wf * wh * ww, C)
    return x

def window_reverse(windows, window_size, original_size):
    F, H, W = original_size
    wf, wh, ww = window_size
    B = windows.shape[0] // ((F//wf) * (H//wh) * (W//ww))
    C = windows.shape[-1]
    
    x = windows.reshape(B, F//wf, H//wh, W//ww, wf, wh, ww, C)
    x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    x = x.reshape(B, F, H, W, C)
    return x

# --- Wavelet Color Correction Utils ---

def wavelet_blur(x, radius):
    # x: (N, H, W, C) -> MLX standard
    # Simple approximations for Gaussian blur using repeated box or predefined kernel
    # Since MLX conv is channel last:
    # kernel: (H, W, In, Out)
    
    vals = np.array([
        [0.0625, 0.125, 0.0625],
        [0.125,  0.25,  0.125 ],
        [0.0625, 0.125, 0.0625],
    ], dtype=np.float32)
    kernel = mx.array(vals) # (3, 3)
    
    # Reshape for depthwise conv: (3, 3, 1, 1) broadcasted manually?
    # MLX Grouped conv: weights (k, k, in/groups, out)
    # here groups=C, in=C, out=C -> weight (k, k, 1, C)
    
    N, H, W, C = x.shape
    weight = mx.broadcast_to(kernel[:, :, None, None], (3, 3, 1, C))
    
    # Pad
    pad = radius
    # MLX pad: ((N_l, N_r), (H_l, H_r), (W_l, W_r), (C_l, C_r))
    x_pad = mx.pad(x, ((0,0), (pad, pad), (pad, pad), (0,0)), mode='edge') # replicate approx
    
    # Dilated conv
    out = nn.conv2d(x_pad, weight, stride=1, padding=0, dilation=radius, groups=C)
    return out

def wavelet_decompose(x, levels=5):
    high = mx.zeros_like(x)
    low = x
    for i in range(levels):
        radius = 2 ** i
        blurred = wavelet_blur(low, radius)
        high = high + (low - blurred)
        low = blurred
    return high, low

def wavelet_reconstruct(content, style, levels=5):
    c_high, _ = wavelet_decompose(content, levels)
    _, s_low = wavelet_decompose(style, levels)
    return c_high + s_low