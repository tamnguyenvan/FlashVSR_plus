import mlx.core as mx
import numpy as np
import imageio
from tqdm import tqdm
from .models import WanModel, LQProjector, TCDecoder
from .utils import precompute_freqs_cis, wavelet_reconstruct

class FlashVSRPipeline:
    def __init__(self, model_path):
        print(f"Loading weights from {model_path}...")
        self.weights = mx.load(model_path)
        
        # 1. Configs
        self.dit_config = {
            'dim': 1536, 'in_dim': 16, 'ffn_dim': 8960, 
            'out_dim': 16, 'num_heads': 12, 'num_layers': 30,
            'patch_size': (1, 2, 2), 'freq_dim': 256
        }
        
        # 2. Initialize Models
        self.dit = WanModel(self.dit_config)
        self.lq_proj = LQProjector(3, 1536) # Input 3 channels -> DiT Dim
        self.tc_decoder = TCDecoder()
        
        # 3. Load Weights
        # Helper to filter and strip prefixes
        def load_part(model, prefix):
            part_weights = {
                k.replace(f"{prefix}.", ""): v 
                for k, v in self.weights.items() 
                if k.startswith(f"{prefix}.")
            }
            if not part_weights:
                print(f"Warning: No weights found for {prefix}")
            else:
                # strict=False allows missing buffers/unused keys logic if architectures slightly mismatch
                model.load_weights(list(part_weights.items()), strict=False)

        print("Loading DiT...")
        load_part(self.dit, "dit")
        print("Loading LQ Projector...")
        load_part(self.lq_proj, "lq_proj")
        print("Loading TCDecoder...")
        load_part(self.tc_decoder, "tc_decoder")
        
        # 4. Context & Utils
        self.context = mx.array(self.weights["posi_prompt"]) # (L, D)
        # Precompute RoPE for max expected length
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(self.dit_config['dim'] // self.dit_config['num_heads'])

    def generate(self, lq_video_path, output_path, num_frames=81, height=480, width=832, color_fix=True):
        
        reader = imageio.get_reader(lq_video_path)
        writer = imageio.get_writer(output_path, fps=30)
        
        # Initialize Latents (Noise)
        # Time dimension calculation: (T-1)//4 + 1 for DiT latents
        latent_time = (num_frames - 1) // 4 + 1
        latents = mx.random.normal((1, latent_time, height//8, width//8, 16))
        
        # Timestep (t=1000 for generation start in Flow Matching)
        t = mx.array([1000.0])
        
        # State Caches
        lq_caches = None
        dit_caches = [None] * self.dit_config['num_layers']
        tc_mems = None
        
        # We process in chunks of 4 frames (matching DiT temporal patch 1 and LQ Proj downsample)
        # LQ Proj consumes 4 frames -> produces 1 latent step
        
        # Buffer input frames
        frame_buffer = []
        latent_idx = 0
        
        print("Starting Inference...")
        pbar = tqdm(total=num_frames)
        
        for i, im in enumerate(reader):
            if i >= num_frames: break
            
            # Normalize -1 to 1
            im_mx = mx.array(im.astype(np.float32) / 127.5 - 1.0)
            frame_buffer.append(im_mx)
            
            # Process when we have 4 frames (or remainder at end logic)
            # Simplified: FlashVSR usually processes chunks of 4 LQ frames -> 1 Latent
            
            if len(frame_buffer) == 4:
                # 1. Prepare Chunk (1, 4, H, W, 3)
                lq_chunk = mx.stack(frame_buffer, axis=0)[None, ...]
                
                # 2. LQ Projection -> DiT Condition
                # Returns list of features for each DiT block
                lq_feats, lq_caches = self.lq_proj.stream_forward(lq_chunk, lq_caches)
                
                # 3. Select corresponding Latent
                if latent_idx < latents.shape[1]:
                    curr_latent = latents[:, latent_idx:latent_idx+1]
                    
                    # 4. DiT Denoise Step
                    # RoPE for 1 step
                    f_cos = self.freqs_cos[:1]
                    f_sin = self.freqs_sin[:1]
                    
                    # In FlashVSR/FlowMatch, noise_pred is the vector field v
                    # Euler step: x_prev = x - v * dt (dt=1 for one-step)
                    noise_pred, dit_caches = self.dit(
                        curr_latent, t, f_cos, f_sin, 
                        lq_latents=lq_feats, caches=dit_caches
                    )
                    
                    cleaned_latent = curr_latent - noise_pred
                    
                    # 5. TCDecoder Decode
                    # Pass chunk of LQ video as condition
                    # TCDecoder logic handles time expansion (1 latent -> 4 frames)
                    rec_chunk, tc_mems = self.tc_decoder.decode_video(
                        cleaned_latent, cond=lq_chunk, mems=tc_mems
                    )
                    
                    if rec_chunk is not None:
                        # 6. Color Correction (Wavelet / Adain)
                        if color_fix:
                            rec_chunk = wavelet_reconstruct(rec_chunk, lq_chunk, levels=5)
                        
                        # Clip & Save
                        rec_chunk = mx.clip(rec_chunk, -1.0, 1.0)
                        
                        # Convert back to uint8 image
                        out_frames = np.array(rec_chunk[0] * 127.5 + 127.5).astype(np.uint8)
                        for f in out_frames:
                            writer.append_data(f)
                            pbar.update(1)
                
                frame_buffer = []
                latent_idx += 1
                
                # Memory cleanup prompt
                mx.eval(lq_caches, dit_caches, tc_mems)

        writer.close()
        pbar.close()
        print(f"Saved to {output_path}")