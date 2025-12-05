import argparse
from src.pipeline import FlashVSRPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="flashvsr_mlx.safetensors")
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_video", type=str, default="output.mp4")
    args = parser.parse_args()
    
    pipe = FlashVSRPipeline(args.model_path)
    pipe.generate(args.input_video, args.output_video)