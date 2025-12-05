import argparse
import torch
import numpy as np
import mlx.core as mx
from safetensors.numpy import save_file

def map_weights(key, value):
    # Convert PyTorch tensors to Numpy
    value = value.cpu().float().numpy()
    
    # 1. Linear Layers
    # PyTorch Linear weight: (Out, In)
    # MLX Linear weight: (In, Out)
    if "weight" in key and value.ndim == 2:
        # Check if it's an embedding (usually not named 'weight' in isolation like Linear, but good check)
        # Embeddings in PT: (Num, Dim). MLX: (Num, Dim). No transpose.
        # But for nn.Linear layer weights:
        return value.T
    
    # 2. Conv2d
    # PyTorch: (Out, In, H, W)
    # MLX: (Out, H, W, In) for nn.Conv2d
    if "weight" in key and value.ndim == 4:
        return value.transpose(0, 2, 3, 1)

    # 3. Conv3d
    # PyTorch: (Out, In, D, H, W)
    # MLX: (Out, D, H, W, In) for nn.Conv3d
    if "weight" in key and value.ndim == 5:
        return value.transpose(0, 2, 3, 4, 1)
        
    return value

def convert(args):
    print("Loading PyTorch weights...")
    final_weights = {}

    # --- 1. DiT (WanModel) ---
    print("Converting DiT...")
    try:
        from safetensors.torch import load_file
        dit_sd = load_file(args.dit_path)
    except:
        dit_sd = torch.load(args.dit_path, map_location="cpu", weights_only=True)
        if "model_state" in dit_sd: dit_sd = dit_sd["model_state"]
    
    for k, v in dit_sd.items():
        # WanModel mapping (simplified as names mostly match standard DiT)
        # Ensure 'weight' keys are transposed/permuted
        # Fix specific Wan naming if needed (e.g., norms)
        if "norm" in k and "weight" in k and v.ndim == 1:
            # RMSNorm weight, 1D, no change
            pass
        final_weights[f"dit.{k}"] = map_weights(k, v)

    # --- 2. LQ Projector ---
    print("Converting LQ Projector...")
    lq_sd = torch.load(args.lq_path, map_location="cpu", weights_only=True)
    if "state_dict" in lq_sd: lq_sd = lq_sd["state_dict"]
    
    for k, v in lq_sd.items():
        # Map linear_layers.X.weight/bias
        # Map causal convs
        final_weights[f"lq_proj.{k}"] = map_weights(k, v)

    # --- 3. TCDecoder ---
    print("Converting TCDecoder...")
    tc_sd = torch.load(args.tc_path, map_location="cpu", weights_only=True)
    
    # Dynamic Mapping for TCDecoder
    # The PyTorch code creates `deepen` layers dynamically. We named them `deepen_start` and `deepen_final`.
    # We need to map the keys from the flat sequential used in PyTorch to our named blocks.
    
    # Structure of PyTorch keys likely:
    # decoder.0 (Clamp) - no weights
    # decoder.1 (Conv) -> start_conv
    # decoder.2 (ReLU)
    # decoder.3 (Identity) -> deepen_start.0
    # decoder.4 (ReLU) -> deepen_start.1
    # decoder.5 (MemBlock) -> mem0_0
    # ...
    # We need to manually map these indices based on the list in src_models_TCDecoder.py
    
    # Let's inspect keys roughly.
    # To facilitate robustness, we iterate and check prefixes.
    
    layer_map = {
        "decoder.1": "start_conv",
        "decoder.3": "deepen_start.0",
        "decoder.5": "mem0_0", "decoder.6": "mem0_1", "decoder.7": "mem0_2",
        "decoder.9": "grow0", "decoder.10": "conv0_out",
        "decoder.11": "mem1_0", "decoder.12": "mem1_1", "decoder.13": "mem1_2",
        "decoder.15": "grow1", "decoder.16": "conv1_out",
        "decoder.17": "mem2_0", "decoder.18": "mem2_1", "decoder.19": "mem2_2",
        "decoder.21": "grow2", "decoder.22": "conv2_out",
        "decoder.24": "deepen_final.0",
        "decoder.26": "final_conv"
    }
    
    for k, v in tc_sd.items():
        prefix = ".".join(k.split(".")[:2]) # e.g., decoder.1
        suffix = ".".join(k.split(".")[2:])
        
        if prefix in layer_map:
            new_prefix = layer_map[prefix]
            if suffix:
                new_key = f"tc_decoder.{new_prefix}.{suffix}"
            else:
                new_key = f"tc_decoder.{new_prefix}"
            
            # Special case: MemBlock structure
            # PyTorch: MemBlock.conv (Sequential)
            # MLX: MemBlock.conv (Sequential) match 1:1
            final_weights[new_key] = map_weights(k, v)
        else:
            # Maybe pixel shuffle or head?
            if "pixel_shuffle" in k: pass # Logic handled in forward
            elif "decoder" not in k: # Generic fallback
                final_weights[f"tc_decoder.{k}"] = map_weights(k, v)
            else:
                print(f"Skipping unknown TCDecoder key: {k}")

    # --- 4. Prompt ---
    print("Converting Prompt...")
    prompt = torch.load(args.prompt_path, map_location="cpu")
    final_weights["posi_prompt"] = prompt.cpu().float().numpy()

    print(f"Saving {len(final_weights)} tensors to {args.output_path}...")
    save_file(final_weights, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit_path", type=str, required=True)
    parser.add_argument("--lq_path", type=str, required=True)
    parser.add_argument("--tc_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="flashvsr_mlx.safetensors")
    args = parser.parse_args()
    convert(args)