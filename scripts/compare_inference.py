#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from dust3r.inference import inference
from dust3r.model import ARCroco3DStereo
from dust3r.utils.image import load_images

def load_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = ARCroco3DStereo.from_pretrained(model_path).to(device)
    model.eval()
    return model

def run_inference_on_frames(model, frames, device, size=512):
    """
    Run inference on a list of frames (numpy arrays).
    Returns list of depth maps (numpy).
    """
    depth_maps = []
    
    # Process batch or frame-by-frame?
    # Monocular inference per frame
    print(f"Running inference on {len(frames)} frames...")
    
    for i, frame in enumerate(tqdm(frames)):
        # Prepare single view input
        # Convert BGR (OpenCV) to RGB for Dust3r
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Helper to create view dict as expected by inference
        # We can use load_images if we had paths, but here we have memory images
        # Construct view manually
        
        # Resize if needed (simplest to let model handle it or input correct size)
        # Dust3r expects tensor input usually via its util
        # Let's use a temporary list of 1 view
        
        # Manual construction of 'views' list for 1 image
        # Needs Image -> Tensor transformation
        # dust3r.image.load_images handles paths
        # Let's simulate:
        
        from PIL import Image
        from torchvision import transforms
        
        img_pil = Image.fromarray(frame_rgb)
        W, H = img_pil.size
        
        # Resize to 'size' (closest 16 multiple?)
        # For simplicity, if input is already 384x512, ensuring divisible by 16
        # 384%16=0, 512%16=0. Fine.
        
        # We need to construct the 'views' dict expected by inference()
        # It's usually: [{'img': tensor, 'true_shape': ..., 'idx': 0, 'instance': '0'}]
        # 'img' tensor should be (1, 3, H, W) normalized -1 to 1?
        # Let's verify dust3r image loading.
        # It calls _load_images -> transforms.Normalize((0.5,),(0.5,))
        
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        img_tensor = tf(img_pil).to(device).unsqueeze(0) # (1, 3, H, W)
        
        view = {
            'img': img_tensor,
            'true_shape': torch.tensor([[H, W]]),
            'idx': 0,
            'instance': '0',
            'img_mask': torch.tensor([True]),
            'ray_mask': torch.tensor([False]), # Required by model
            'ray_map': torch.zeros((1, H, W, 6)).to(device), # Required (dummy)
            'camera_pose': torch.eye(4).unsqueeze(0).to(device), # Identity assumption
            'reset': torch.tensor([False]), # Required
            'update': torch.tensor([True]) # Required
        }
        
        views = [view]
        
        
        # Mock Accelerator for inference function expectation
        class MockAccelerator:
            def __init__(self, dev):
                self.device = dev
        
        mock_accel = MockAccelerator(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                 result, _ = inference(views, model, device, verbose=False, accelerator=mock_accel)

        
        # Extract depth
        # result['pred'][0]['pts3d_in_self_view'] -> (1, H, W, 3)
        pts3d = result['pred'][0]['pts3d_in_self_view'][0].cpu().numpy() # (H, W, 3)
        depth = pts3d[..., 2] # Z coordinate
        
        depth_maps.append(depth)
        
    return depth_maps

def colorize_depth(depth, vmin=None, vmax=None, cmap='magma'):
    # Normalize
    if vmin is None: vmin = np.nanmin(depth)
    if vmax is None: vmax = np.nanmax(depth)
    
    # Handle NaNs
    depth = np.nan_to_num(depth, nan=vmin)
    
    # Clip
    depth = np.clip(depth, vmin, vmax)
    
    # Normalize 0-1
    if vmax - vmin > 1e-6:
        norm = (depth - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(depth)
        
    # Apply colormap
    colormap = plt.get_cmap(cmap)
    colored = colormap(norm)[:, :, :3] # RGBA -> RGB
    
    return (colored * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--finetuned_model", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output_comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_frames", type=int, default=150)
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    vid_name = os.path.basename(args.video_path).replace(".mp4", "")
    
    # Read video
    print(f"Reading video {args.video_path}...")
    cap = cv2.VideoCapture(args.video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
        if len(frames) >= args.max_frames: break
    cap.release()
    print(f"Loaded {len(frames)} frames.")
    
    # 1. Run Base Model
    model = load_model(args.base_model, args.device)
    depths_base = run_inference_on_frames(model, frames, args.device)
    del model
    torch.cuda.empty_cache()
    
    # 2. Run Finetuned Model
    model = load_model(args.finetuned_model, args.device)
    depths_ft = run_inference_on_frames(model, frames, args.device)
    del model
    torch.cuda.empty_cache()
    
    # 3. Combine and Save
    print("Generating comparison video...")
    H, W, _ = frames[0].shape
    
    # Calculate global min/max for depth visualization stability per model?
    # Or per frame? Usually per-frame relative is clearer for structure, 
    # but global helps see consistency. Let's do per-frame for now to highlight details.
    
    out_file = os.path.join(args.output_path, f"{vid_name}_comparison.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Width = W * 3 (RGB, Depth1, Depth2)
    out_vid = cv2.VideoWriter(out_file, fourcc, 30.0, (W*3, H))
    
    for i in range(len(frames)):
        rgb = frames[i]
        d_base = depths_base[i]
        d_ft = depths_ft[i]
        
        # Colorize
        # Note: Dust3r outputs metric depth (meters). 
        # Base model trained on generic might be scale-invariant or different scale.
        # FT model trained on DNA (metric).
        # We should use robust quantile min/max or per-frame range.
        
        d_base_vis = colorize_depth(d_base)
        d_ft_vis = colorize_depth(d_ft)
        
        # Convert RGB to BGR for cv2 (wait, colorize returns RGB, cv2 needs BGR)
        # colorize output is RGB (plt default). 
        # frames[i] is BGR (cv2 read).
        
        d_base_vis_bgr = cv2.cvtColor(d_base_vis, cv2.COLOR_RGB2BGR)
        d_ft_vis_bgr = cv2.cvtColor(d_ft_vis, cv2.COLOR_RGB2BGR)
        
        # Stack horizontal
        combined = np.hstack([rgb, d_base_vis_bgr, d_ft_vis_bgr])
        
        # Add labels
        cv2.putText(combined, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Base Model", (W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Finetuned", (2*W+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out_vid.write(combined)
        
    out_vid.release()
    print(f"Saved comparison to {out_file}")

if __name__ == "__main__":
    main()
