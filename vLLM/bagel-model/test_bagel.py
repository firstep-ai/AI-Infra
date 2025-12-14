import sys
import os

# Add local source to path to use the latest code supporting Bagel
sys.path.insert(0, "/workspace/vllm-omni")

from vllm_omni.diffusion.omni_diffusion import OmniDiffusion

def main():
    model_path = "/workspace/models/BAGEL-7B-MoT"
    prompt = "A futuristic city skyline at morning, cyberpunk style"
    
    print(f"Initializing OmniDiffusion with model: {model_path}")
    
    try:
        # Initializing without extra cache args to be safe, assuming defaults work
        pipeline = OmniDiffusion(model=model_path)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return

    print(f"Generating image for prompt: '{prompt}'")
    try:
        image = pipeline.generate(prompt)
        # Bagel pipeline returns a PIL Image
        output_file = "/workspace/bagel_output.png"
        image.save(output_file)
        print(f"Generation complete! Saved to {output_file}")
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    main()
