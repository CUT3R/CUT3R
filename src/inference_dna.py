import torch
import torch.nn as nn
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig, inf
from dust3r.datasets.dna import DNAMultiSeqDataset
from dust3r.inference import inference
from dust3r.utils.image import rgb
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.utils.device import to_numpy
from torchvision import transforms
import torchvision.utils as vutils

# Hardcoded config from finetune_dna.yaml
# We use the definition string-like approach or just instantiate directly
def get_model(device):
    # Copying config params from finetune_dna.yaml
    cfg = ARCroco3DStereoConfig(
        freeze='encoder', 
        state_size=768, 
        state_pe='2d', 
        pos_embed='RoPE100', 
        rgb_head=True, 
        pose_head=True, 
        patch_embed_cls='ManyAR_PatchEmbed', 
        img_size=(512, 512), 
        head_type='dpt', 
        output_mode='pts3d+pose', 
        depth_mode=('exp', -inf, inf), 
        conf_mode=('exp', 1, inf), 
        pose_mode=('exp', -inf, inf), 
        enc_embed_dim=1024, 
        enc_depth=24, 
        enc_num_heads=16, 
        dec_embed_dim=768, 
        dec_depth=12, 
        dec_num_heads=12, 
        landscape_only=False
    )
    model = ARCroco3DStereo(cfg)
    return model.to(device)

def load_checkpoint(model, path):
    if not os.path.isfile(path):
        print(f"Error: Checkpoint not found at {path}")
        sys.exit(1)
    
    print(f"Loading checkpoint {path}...")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    
    # Handle possible 'module.' prefix if saved via DDP
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=True)
    print("Checkpoint loaded.")
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize Model
    model = get_model(device)
    
    # 2. Load Checkpoint
    ckpt_path = '/mnt/raid/lin/CUT3R/checkpoints/2026_113_finetune_dna_human/checkpoint-best.pth'
    model = load_checkpoint(model, ckpt_path)
    model.eval()

    # 3. Setup Dataset
    # Using 'test' config style but with training path to verify on training data as requested
    # resolution=[[384, 512]], S=8 (or 2 for quick test)
    # Note: dataset_location matching config
    print("Loading dataset...")
    # Config uses: transform=ImgNorm for test
    from dust3r.utils.image import ImgNorm

    dataset = DNAMultiSeqDataset(
        dataset_location='/mnt/raid/lin/nas-train/Concat_Dataset/DNA', 
        sequence_list=None, 
        S=8, 
        resolution=[[384, 512]], 
        stride=1, 
        transform=ImgNorm
    )
    print(f"Dataset size: {len(dataset)}")

    # 4. Get a sample
    idx = np.random.randint(0, len(dataset))
    print(f"Picking sample index {idx}")
    views = dataset[idx] # Returns a list of dicts (views)
    
    # Collate into batch (list of views -> list of batches of size 1)
    # inference() expects list of lists/batches? 
    # dust3r.inference.inference(groups, ...) where groups is list of (views)
    # views in dataset[idx] is already a list of view dicts.
    
    # Add batch dimension to each tensor in the view dicts
    batch_views = []
    for view in views:
        v = {}
        for key, value in view.items():
            if isinstance(value, torch.Tensor):
                v[key] = value.unsqueeze(0).to(device)
            else:
                v[key] = [value] # List for metadata
        batch_views.append(v)
    
    # 5. Run Inference
    print("Running inference...")
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        results = inference([batch_views], model, device)
    
    # results is a list of dicts (one per scene/batch item)
    # output keys: 'views', 'pred'
    # 'views' is the input batch (modified?)
    # 'pred' contains 'pts3d', 'conf', 'rgb' etc.
    
    pred = results[0]['pred']
    gt_views = results[0]['views']
    
    print("Inference done. Saving visualizations...")
    
    # 6. Visualize
    # Save input RGB, Pred RGB, Confidence, Depth
    
    os.makedirs('inference_vis', exist_ok=True)
    
    num_views = len(gt_views)
    for i in range(num_views):
        # Input RGB
        img_gt = gt_views[i]['img'][0].permute(1, 2, 0).cpu().numpy()
        # Denormalize if dataset normalizes (ImgNorm usually does (x-0.5)*2)
        img_gt = (img_gt + 1) * 0.5
        
        # Pred RGB ?? (Model usually predicts geometry/descriptors, usually not RGB unless head enabled)
        # ARCroco3DStereo has rgb_head=True in config.
        # Check if 'rgb' in pred[i]
        
        plt.figure(figsize=(15, 5))
        
        # GT RGB
        plt.subplot(1, 4, 1)
        plt.imshow(np.clip(img_gt, 0, 1))
        plt.title(f"GT View {i}")
        plt.axis('off')
        
        # Pred RGB (if available)
        idx_offset = 2
        if 'rgb' in pred[i] and pred[i]['rgb'] is not None:
             pred_rgb = pred[i]['rgb'][0].permute(1, 2, 0).detach().cpu().numpy()
             pred_rgb = (pred_rgb + 1) * 0.5
             plt.subplot(1, 4, 2)
             plt.imshow(np.clip(pred_rgb, 0, 1))
             plt.title(f"Pred RGB View {i}")
             plt.axis('off')
             idx_offset = 3
        
        # Confidence
        if 'conf' in pred[i]:
            conf = pred[i]['conf'][0].detach().cpu().numpy()
            plt.subplot(1, 4, idx_offset)
            plt.imshow(conf, cmap='turbo')
            plt.title(f"Confidence")
            plt.axis('off')
            idx_offset += 1
            
        # Depth (pts3d z-channel)
        if 'pts3d' in pred[i]:
            pts3d = pred[i]['pts3d'][0].detach().cpu().numpy()
            depth = pts3d[..., 2]
            plt.subplot(1, 4, idx_offset)
            plt.imshow(depth, cmap='magma')
            plt.title("Pred Depth")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'inference_vis/view_{i}.png')
        plt.close()

    print(f"Visualizations saved to {os.path.abspath('inference_vis')}")

if __name__ == "__main__":
    main()
