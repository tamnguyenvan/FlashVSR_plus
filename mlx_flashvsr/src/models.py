import math
import mlx.core as mx
import mlx.nn as nn

# --- Shared Layers ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        # x: (..., dim)
        # MLX RMSNorm is slightly different, implementing explicit calculation
        mean_square = mx.mean(mx.square(x), axis=-1, keepdims=True)
        return x * mx.rsqrt(mean_square + self.eps) * self.weight

class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # MLX Conv3d: input (N, D, H, W, C)
        # Weight shape handled by nn.Conv3d is (out_channels, D, H, W, in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        
        # Calculate Causal Padding for Time dimension (D)
        # PyTorch causal padding: (padding[2], padding[2], padding[1], padding[1], 2*padding[0], 0)
        # This means we pad Time on the LEFT (past) only.
        
        self.kernel_t = kernel_size[0]
        self.pad_t = 2 * padding[0] if isinstance(padding, tuple) else 2 * padding
        self.pad_h = padding[1] if isinstance(padding, tuple) else padding
        self.pad_w = padding[2] if isinstance(padding, tuple) else padding

    def __call__(self, x, cache=None):
        # x: (N, D, H, W, C)
        
        # 1. Handle Temporal Cache/Padding
        if cache is not None:
            # Concat last (k-1) frames from cache to current input
            # cache shape: (N, k-1, H, W, C)
            x = mx.concatenate([cache, x], axis=1)
            # Adjust padding: we have history, so effectively pad=0 on left now logic-wise, 
            # but structurally we just perform valid conv on time if we manage size correctly.
            # However, standard CausalConv implementation implies:
            # Pad Left = Kernel_T - 1 (or specific amount).
            
            # The PyTorch code does:
            # if cache: cat(cache, x); padding[time] -= cache.shape
            # If we cat full cache, we reduce left pad.
            
            # Simplest approach for MLX streaming: 
            # If cache provided, it contains the necessary "left pad" context.
            # We still need spatial padding.
            pad_t_needed = 0 
        else:
            # Initial frame: pad left
            pad_t_needed = self.pad_t

        # 2. Apply Padding
        # MLX Pad: ((N_l, N_r), (D_l, D_r), (H_l, H_r), (W_l, W_r), (C_l, C_r))
        x_padded = mx.pad(x, (
            (0, 0), 
            (pad_t_needed, 0), 
            (self.pad_h, self.pad_h), 
            (self.pad_w, self.pad_w), 
            (0, 0)
        ))

        # 3. Convolution
        out = self.conv(x_padded)

        # 4. Save Cache for next step
        # Cache the last (kernel_t - 1 * stride_t) input frames?
        # Actually, for causal conv with kernel K, we need previous K-1 frames to compute next frame T=0.
        # stride affects this. Assuming stride_t=1 usually for causal context preservation unless downsampling.
        # The PyTorch code uses CACHE_T=2 generally.
        
        # Store last (kernel_t - 1) frames of INPUT x (before padding if we consider continuous stream)
        # But easier to just take the last (kernel_t - 1) of the padded input used.
        cache_len = self.kernel_t - 1
        if cache_len > 0:
            new_cache = x_padded[:, -cache_len:, :, :, :]
        else:
            new_cache = mx.zeros((x.shape[0], 0, x.shape[2], x.shape[3], x.shape[4]))
            
        return out, new_cache

# --- WanModel (DiT) ---

class WanAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

    def __call__(self, x, freqs_cos, freqs_sin, cache=None):
        # x: (N, L, D) - Flattened patches
        N, L, D = x.shape
        
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        # Reshape to heads: (N, L, nH, hD)
        q = q.reshape(N, L, self.num_heads, self.head_dim)
        k = k.reshape(N, L, self.num_heads, self.head_dim)
        v = v.reshape(N, L, self.num_heads, self.head_dim)

        # RoPE
        # freqs: (L, hD/2)
        def apply_rope(x_in, c, s):
            # x_in: (..., hD)
            # split half
            x0 = x_in[..., 0::2]
            x1 = x_in[..., 1::2]
            # broadcast freqs: (1, L, 1, hD/2)
            c = c[None, :, None, :]
            s = s[None, :, None, :]
            
            y0 = x0 * c - x1 * s
            y1 = x0 * s + x1 * c
            
            # stack back
            y = mx.stack([y0, y1], axis=-1)
            return y.reshape(x_in.shape)

        q = apply_rope(q, freqs_cos, freqs_sin)
        k = apply_rope(k, freqs_cos, freqs_sin)

        # KV Cache for DiT Streaming (Block-wise)
        if cache is not None:
            # cache is (k_past, v_past)
            k_past, v_past = cache
            if k_past is not None:
                k = mx.concatenate([k_past, k], axis=1)
                v = mx.concatenate([v_past, v], axis=1)
            new_cache = (k, v)
        else:
            new_cache = None

        # Scaled Dot Product Attention
        # MLX Optimized implementation
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        
        # Reshape back: (N, L, D)
        x = x.reshape(N, L, D)
        
        return self.o(x), new_cache

class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.self_attn = WanAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim)
        )
        self.modulation = mx.zeros((6, dim)) # Placeholder, loaded from weights

    def __call__(self, x, t_mod, freqs_cos, freqs_sin, cache=None):
        # t_mod: (N, 6, dim)
        
        # Modulation parameters
        # self.modulation is (6, dim), broadcast to (N, 6, dim) + t_mod
        mod = self.modulation[None, ...] + t_mod 
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [mod[:, i:i+1, :] for i in range(6)]

        # Attention Block
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa) + shift_msa
        attn_out, new_cache = self.self_attn(x_norm, freqs_cos, freqs_sin, cache)
        x = x + gate_msa * attn_out

        # FFN Block
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp) + shift_mlp
        ffn_out = self.ffn(x_norm)
        x = x + gate_mlp * ffn_out

        return x, new_cache

class WanModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config['dim']
        self.patch_size = config['patch_size'] # (1, 2, 2)
        
        # Patch Embed: Conv3d (C_in, C_out, k, s)
        self.patch_embedding = nn.Conv3d(
            config['in_dim'], self.dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        self.time_embedding = nn.Sequential(
            nn.Linear(config['freq_dim'], self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(self.dim, self.dim * 6)
        )
        
        self.blocks = [
            DiTBlock(self.dim, config['num_heads'], config['ffn_dim'])
            for _ in range(config['num_layers'])
        ]
        
        self.head = nn.Linear(self.dim, config['out_dim'] * math.prod(self.patch_size))
        self.head_norm = nn.LayerNorm(self.dim, eps=1e-6, affine=False)
        self.head_mod = mx.zeros((2, self.dim)) # shift, scale

    def __call__(self, x, t, freqs_cos, freqs_sin, lq_latents=None, caches=None):
        # x: (N, D, H, W, C) - Latent video
        
        # 1. Time Embed
        t_emb = self.time_embedding(t) # (N, dim)
        t_mod = self.time_projection(t_emb)
        t_mod = t_mod.reshape(t_mod.shape[0], 6, self.dim)
        
        # 2. Patchify
        x = self.patch_embedding(x) # (N, D, H/2, W/2, dim)
        N, D, H, W, C = x.shape
        
        # Flatten for DiT blocks: (N, L, dim)
        x_flat = x.reshape(N, -1, C)

        # 3. Blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            # Inject LQ Latents if available (Simple Additive)
            if lq_latents is not None and i < len(lq_latents):
                # lq_latents[i] needs to match shape (N, D, H, W, C) -> (N, L, C)
                lq_res = lq_latents[i].reshape(N, -1, C)
                x_flat = x_flat + lq_res

            cache = caches[i] if caches else None
            x_flat, new_c = block(x_flat, t_mod, freqs_cos, freqs_sin, cache)
            new_caches.append(new_c)
            
        # 4. Head
        # Modulation
        head_mod = self.head_mod[None, ...] + t_emb[:, None, :]
        shift, scale = head_mod[:, 0:1, :], head_mod[:, 1:2, :]
        
        x_flat = self.head_norm(x_flat) * (1 + scale) + shift
        x_flat = self.head(x_flat)
        
        # 5. Unpatchify
        # x_flat: (N, D*H*W, out_dim*patch_vol) -> (N, D*H*W, C_out * 1*2*2)
        out_channels = x_flat.shape[-1] // 4 
        x = x_flat.reshape(N, D, H, W, 1, 2, 2, out_channels)
        x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7) # (N, D, 1, H, 2, W, 2, C_out)
        x = x.reshape(N, D, H*2, W*2, out_channels)
        
        return x, new_caches

# --- LQ Projector ---

class LQProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # PyTorch: in_dim*ff*hh*ww -> 3 * 1 * 16 * 16 = 768
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        
        # PixelShuffle3d factor (1, 16, 16) implied by input dims logic in utils.py
        
        self.conv1 = CausalConv3d(in_dim * 256, self.hidden_dim1, (4,3,3), stride=(2,1,1), padding=(1,1,1))
        self.norm1 = RMSNorm(self.hidden_dim1, eps=1e-5) # images=False in PyTorch utils implies generic RMS
        
        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4,3,3), stride=(2,1,1), padding=(1,1,1))
        self.norm2 = RMSNorm(self.hidden_dim2, eps=1e-5)
        
        self.linear_layers = [nn.Linear(self.hidden_dim2, out_dim) for _ in range(30)] # 30 layers for Wan DiT

    def stream_forward(self, x, caches=None):
        # x: (N, D, H, W, C=3)
        # 1. Pixel Shuffle (Space to Depth) 
        # (N, D, H, W, 3) -> (N, D, H/16, W/16, 3*16*16)
        N, D, H, W, C = x.shape
        x = x.reshape(N, D, H//16, 16, W//16, 16, C)
        x = x.transpose(0, 1, 2, 4, 3, 5, 6)
        x = x.reshape(N, D, H//16, W//16, -1)
        
        c1, c2 = caches if caches else (None, None)
        
        x, nc1 = self.conv1(x, c1)
        x = nn.silu(self.norm1(x))
        
        x, nc2 = self.conv2(x, c2)
        x = nn.silu(self.norm2(x))
        
        # Project to each DiT layer dimension
        # The DiT expects LQ latents injected at blocks. 
        # Return list of tensors.
        outs = [l(x) for l in self.linear_layers]
        
        return outs, (nc1, nc2)

# --- TCDecoder (Tiny AutoEncoder Decoder) ---

class IdentityConv2d(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        # Initialized to Dirac/Identity in PyTorch. 
        # In MLX, we load weights, so init doesn't matter much, just shape.
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2, bias=False)

    def __call__(self, x):
        return self.conv(x)

class MemBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, out_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_dim, out_dim, 1, bias=False) if in_dim != out_dim else nn.Identity()
        self.act = nn.ReLU()

    def __call__(self, x, past):
        # x: (N, H, W, C)
        # past: (N, H, W, C)
        # MLX channel last concatenation
        inp = mx.concatenate([x, past], axis=-1)
        return self.act(self.conv(inp) + self.skip(x))

class TGrow(nn.Module):
    def __init__(self, in_dim, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(in_dim, in_dim * stride, 1, bias=False)

    def __call__(self, x):
        # x: (N, H, W, C)
        x = self.conv(x)
        # Expand time: (N, H, W, C*stride) -> (N, H, W, stride, C) -> (N, stride, H, W, C)
        N, H, W, C_total = x.shape
        C_out = C_total // self.stride
        x = x.reshape(N, H, W, self.stride, C_out)
        x = x.transpose(0, 3, 1, 2, 4) # (N, T, H, W, C)
        return x

class TCDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture defined in src_models_TCDecoder.py.txt
        # channels = [256, 128, 64, 64]
        # latent_channels = 16
        
        self.latent_channels = 16
        n_f = [256, 128, 64, 64]
        
        # --- Base Skeleton Layers ---
        self.clamp = True # tanh logic handled in forward
        self.start_conv = nn.Conv2d(self.latent_channels, n_f[0], 3, padding=1)
        self.start_relu = nn.ReLU()
        
        # Helper to create Identity Deepened layers
        # The PyTorch code creates IdentityConv + ReLU after EVERY ReLU in the base model.
        # We must instantiate these to match the weights.
        
        self.layers = []
        
        # Block 0
        self.mem0_0 = MemBlock(n_f[0], n_f[0])
        self.mem0_1 = MemBlock(n_f[0], n_f[0])
        self.mem0_2 = MemBlock(n_f[0], n_f[0])
        # Upsample 0 (Scale 2) -> nn.Upsample in PyTorch
        self.grow0 = TGrow(n_f[0], 1)
        self.conv0_out = nn.Conv2d(n_f[0], n_f[1], 3, padding=1, bias=False)
        
        # Block 1
        self.mem1_0 = MemBlock(n_f[1], n_f[1])
        self.mem1_1 = MemBlock(n_f[1], n_f[1])
        self.mem1_2 = MemBlock(n_f[1], n_f[1])
        # Upsample 1 (Scale 2)
        self.grow1 = TGrow(n_f[1], 2) # decoder_time_upscale[0] = True -> stride 2
        self.conv1_out = nn.Conv2d(n_f[1], n_f[2], 3, padding=1, bias=False)
        
        # Block 2
        self.mem2_0 = MemBlock(n_f[2], n_f[2])
        self.mem2_1 = MemBlock(n_f[2], n_f[2])
        self.mem2_2 = MemBlock(n_f[2], n_f[2])
        # Upsample 2 (Scale 2)
        self.grow2 = TGrow(n_f[2], 2) # decoder_time_upscale[1] = True -> stride 2
        self.conv2_out = nn.Conv2d(n_f[2], n_f[3], 3, padding=1, bias=False)
        
        # Final
        self.final_relu = nn.ReLU()
        self.final_conv = nn.Conv2d(n_f[3], 3, 3, padding=1)

        # Deepening Layers (Identity + ReLU)
        # In PyTorch `_apply_identity_deepen` adds these after every ReLU.
        # Locations of ReLUs in base:
        # 1. start_relu
        # 2. Inside MemBlocks (MemBlock has convs+relu, and act at end)
        #    MemBlock structure: 2*in->mid (ReLU), mid->mid (ReLU), mid->mid, +skip -> ReLU.
        #    So 3 ReLUs per MemBlock.
        # 3. final_relu
        
        # However, looking at `_apply_identity_deepen`: it iterates the sequential. 
        # MemBlock is a single module in the sequential list.
        # The deepening only applies to ReLUs *visible* in the top-level Sequential.
        # Base Decoder Sequential:
        # [Clamp, Conv, ReLU(1), Mem, Mem, Mem, Upsample, TGrow, Conv, Mem, Mem, Mem, Upsample, TGrow, Conv, Mem, Mem, Mem, Upsample, TGrow, Conv, ReLU(2), Conv]
        
        # ReLUs visible in Sequential:
        # 1. After start_conv
        # 2. Before final_conv
        
        # Wait, does it recurse? The code says `for b in decoder: if isinstance(b, nn.ReLU): ...`
        # So it ONLY adds deepening layers at the top level.
        # So we strictly need deepening layers after `start_relu` and `final_relu`.
        
        self.deepen_start = nn.Sequential(IdentityConv2d(n_f[0]), nn.ReLU())
        self.deepen_final = nn.Sequential(IdentityConv2d(n_f[3]), nn.ReLU())

    def upsample(self, x):
        # Nearest Neighbor Upsample Scale 2
        N, H, W, C = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (N, H, 2, W, 2, C))
        x = x.reshape(N, H*2, W*2, C)
        return x

    def decode_video(self, latents, cond=None, mems=None):
        # latents: (N, D, H, W, C=16)
        # cond: (N, D, H, W, C=3) - LQ Video for pixel shuffle injection?
        # PyTorch code: `if cond is not None: x = torch.cat([self.pixel_shuffle(cond), x], dim=2)`
        
        # Handle Cond (Pixel Shuffle 3D)
        if cond is not None:
            # cond (N, D, H, W, 3) -> Pixel Shuffle (4, 8, 8) -> (N, D/4, H/8, W/8, 3*4*8*8)
            # Actually PixelShuffle3d implementation in utils.py: 
            # (B, C, F, H, W) -> (B, C*ff*hh*ww, F/ff, H/hh, W/ww)
            # Here F=4, H=8, W=8.
            N, D, H, W, C = cond.shape
            # Assuming D is divisible by 4, H,W by 8
            if D % 4 != 0: 
                # Pad first frame repetition logic from utils if needed, or assume caller handles
                pass
            
            c_shuffled = cond.reshape(N, D//4, 4, H//8, 8, W//8, 8, C)
            c_shuffled = c_shuffled.transpose(0, 1, 3, 5, 7, 2, 4, 6) # (N, D', H', W', C, 4, 8, 8)
            c_shuffled = c_shuffled.reshape(N, D//4, H//8, W//8, -1)
            
            # Latents match this spatial resolution?
            # Latents are VAE latents.
            latents = mx.concatenate([c_shuffled, latents], axis=-1)

        # Main Loop over Time (D)
        # TCDecoder processes (N, H, W, C) chunks recurrently
        
        outputs = []
        
        # Initialize Memories if None
        # Mems structure: list of list of tensors for TPool/MemBlocks?
        # TCDecoder uses MemBlocks (recurrence) and TGrow (buffer).
        # We need a stateful runner.
        
        # Since we are implementing `inference only`, we can run frame-by-frame (or chunk).
        # latents is (N, D, H, W, C).
        
        if mems is None:
            # 9 MemBlocks -> 9 states
            mems = [mx.zeros((latents.shape[0], latents.shape[2], latents.shape[3], 256)) for _ in range(3)] + \
                   [mx.zeros((latents.shape[0], latents.shape[2]*2, latents.shape[3]*2, 128)) for _ in range(3)] + \
                   [mx.zeros((latents.shape[0], latents.shape[2]*4, latents.shape[3]*4, 64)) for _ in range(3)]
        
        # TGrow buffers
        # Grow1 (stride 2), Grow2 (stride 2)
        # We need to buffer inputs to grow layers until we have enough to output?
        # Or TGrow outputs multiple frames at once.
        # "TGrow: conv(x) -> split channels -> output multiple frames". 
        # So 1 input frame -> 2 output frames. This expands time.
        
        # Flow:
        # T=0: In -> Mem0..2 -> Up -> Grow0(1) -> Conv -> Mem1..2 -> Up -> Grow1(2->2fr) -> ...
        
        # Let's process input latents one by one
        
        final_frames = []
        
        for t in range(latents.shape[1]):
            x = latents[:, t] # (N, H, W, C)
            
            # Layer 0
            x = mx.tanh(x / 3) * 3
            x = self.start_conv(x)
            x = self.start_relu(x)
            x = self.deepen_start(x)
            
            x = self.mem0_0(x, mems[0]); mems[0] = x
            x = self.mem0_1(x, mems[1]); mems[1] = x
            x = self.mem0_2(x, mems[2]); mems[2] = x
            
            x = self.upsample(x)
            # Grow0 is stride 1, so 1->1
            x = self.grow0(x)[:, 0] # Remove T dim
            x = self.conv0_out(x)
            
            # Layer 1
            x = self.mem1_0(x, mems[3]); mems[3] = x
            x = self.mem1_1(x, mems[4]); mems[4] = x
            x = self.mem1_2(x, mems[5]); mems[5] = x
            
            x = self.upsample(x)
            # Grow1 is stride 2. Output (N, 2, H, W, C)
            xs_1 = self.grow1(x)
            
            # Process the 2 grown frames
            for i in range(2):
                x2 = xs_1[:, i]
                x2 = self.conv1_out(x2)
                
                # Layer 2
                x2 = self.mem2_0(x2, mems[6]); mems[6] = x2
                x2 = self.mem2_1(x2, mems[7]); mems[7] = x2
                x2 = self.mem2_2(x2, mems[8]); mems[8] = x2
                
                x2 = self.upsample(x2)
                # Grow2 is stride 2
                xs_2 = self.grow2(x2)
                
                for j in range(2):
                    x3 = xs_2[:, j]
                    x3 = self.conv2_out(x3)
                    
                    x3 = self.final_relu(x3)
                    x3 = self.deepen_final(x3)
                    x3 = self.final_conv(x3)
                    
                    final_frames.append(x3)

        # Stack T
        # (N*T_out, H, W, C) -> (N, T_out, H, W, C)
        if len(final_frames) > 0:
            out = mx.stack(final_frames, axis=1)
            return out, mems
        else:
            return None, mems