import torch
import torch.nn.functional as F
import os
from run import BDH, BDHConfig  #
import tiktoken
from visualize import (
    generate_neuron_animation, 
    save_gif, 
    add_watermark_to_frames
)

import json

def export_neuron_metadata(model, x_input, x_frames, output_path="neuron_metadata.json"):
    enc = tiktoken.get_encoding("gpt2")
    B, T = x_input.shape
    num_layers = len(x_frames)
    N_total = x_frames[0].shape[-1]
    
    metadata = {}

    print(f"Mapping tokens for {N_total} neurons...")
    
    # We flatten batch and time to treat every token occurrence as a sample
    flat_tokens = x_input.view(-1) # (B*T)
    
    for layer_idx in range(num_layers):
        # Shape: (B*T, N_total)
        acts = x_frames[layer_idx].view(-1, N_total)
        
        for n_idx in range(N_total):
            neuron_acts = acts[:, n_idx]
            # Find the top 5 times this specific neuron fired
            top_vals, top_steps = torch.topk(neuron_acts, k=5)
            
            associations = []
            for val, step in zip(top_vals, top_steps):
                if val > 0.01: # Threshold to ensure it's a real activation
                    # Get the token and two tokens of preceding context
                    start = max(0, step.item() - 2)
                    end = step.item() + 1
                    context_ids = flat_tokens[start:end].tolist()
                    context_str = enc.decode(context_ids)
                    
                    associations.append({
                        "token": context_str,
                        "strength": round(val.item(), 4)
                    })
            
            neuron_id = f"L{layer_idx}_N{n_idx}"
            metadata[neuron_id] = associations

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata exported to {output_path}")

def run_visual_inference(model_path, prompt, output_gif="neuron_dynamics.gif"):
    # 1. Setup Device and Tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = tiktoken.get_encoding("gpt2")

    # 2. Load Model Checkpoint
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['model_args']
    
    # Initialize and load weights
    model = BDH(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 3. Prepare Input
    ids = enc.encode(prompt)
    x_input = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    # 4. Forward Pass with Activation Capture
    # Note: Ensure you updated BDH.forward to support return_activations as shown previously
    print(f"Running inference on: '{prompt}'")
    with torch.no_grad():
        logits, _, x_frames, y_frames = model(x_input, return_activations=True)
        export_neuron_metadata(model, x_input, x_frames, "neuron_metadata.json")
    # 5. Generate Visualization
    print("Generating neuron animation frames...")
    # x_frames/y_frames are lists of (B, T, N) tensors
    # We remove the batch dimension for the visualizer
    x_frames_sq = [f.squeeze(0) for f in x_frames]
    y_frames_sq = [f.squeeze(0) for f in y_frames]

    frames = generate_neuron_animation(
        x_frames=x_frames_sq,
        y_frames=y_frames_sq,
        model=model,
        selection_method='degree'
    )

    # 6. Post-process and Save
    print(f"Saving animation to {output_gif}...")
    frames = add_watermark_to_frames(frames)
    save_gif(frames, output_gif, duration=800)
    print("Success!")

if __name__ == "__main__":
    # Settings - Update these to match your environment
    MODEL_FILE = "ckpt_5000.pt" 
    TEST_PROMPT = "Once upon a time, there was a small dragon"
    
    run_visual_inference(MODEL_FILE, TEST_PROMPT)