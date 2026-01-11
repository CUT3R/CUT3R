import os
import glob
import numpy as np
import cv2
import torch
import PIL.Image
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2, ImgNorm
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

class DNAMultiSeqDataset(BaseMultiViewDataset):
    """
    DNA-Rendering Dataset for MULTIPLE sequences.
    Inherits from BaseMultiViewDataset for compatibility with dust3r training.
    """
    
    def __init__(self,
                 dataset_location,
                 sequence_list=None,
                 moge_depth_dir=None,
                 S=12,
                 resolution=(512, 512),
                 stride=1,
                 transform=None,
                 exclude_seqs=None,
                 view_sample_mode='random',
                 load_moge=True,
                 *args, **kwargs):
                 
        self.dataset_location = dataset_location
        # dust3r config passes S as 'num_views' usually, but here we keep S for compat or pass it up
        # BaseMultiViewDataset takes 'num_views' in __init__ kwargs if passed
        if 'num_views' not in kwargs:
             kwargs['num_views'] = S
        
        super().__init__(resolution=resolution, transform=transform, *args, **kwargs)
        
        self.S = S
        self.stride = stride
        self.exclude_seqs = set(exclude_seqs or [])
        self.view_sample_mode = view_sample_mode
        self.dataset_label = "DNA"
        self.is_metric = True # DNA depth is in meters
        
        # Discover all sequences
        self.sequences = []
        if sequence_list and os.path.exists(sequence_list):
            with open(sequence_list, 'r') as f:
                for line in f:
                    rel_path = line.strip()
                    if rel_path and os.path.isdir(os.path.join(dataset_location, rel_path)):
                         self.sequences.append(rel_path)
        else:
            for seq_name in sorted(os.listdir(dataset_location)):
                seq_dir = os.path.join(dataset_location, seq_name)
                if os.path.isdir(seq_dir) and seq_name not in self.exclude_seqs:
                    self.sequences.append(seq_name)
        
        print(f"DNAMultiSeqDataset: Found {len(self.sequences)} sequences")

        # Build samples (seq, frame)
        self.samples = []
        for seq_name in self.sequences:
            seq_dir = os.path.join(dataset_location, seq_name)
            # Find frames with both rgbs and depth_rendered
            frames = sorted([d for d in os.listdir(seq_dir) if d.isdigit() and os.path.isdir(os.path.join(seq_dir, d))], key=int)
            
            valid_frames = []
            for frame in frames:
                if os.path.exists(os.path.join(seq_dir, frame, 'rgbs')) and \
                   os.path.exists(os.path.join(seq_dir, frame, 'depth_rendered')):
                       # Quick check for view count?
                       # if len(glob.glob(os.path.join(seq_dir, frame, 'rgbs', '*.png'))) >= 48:
                       valid_frames.append(frame)
            
            for i in range(0, len(valid_frames), stride):
                self.samples.append((seq_name, valid_frames[i]))
                
        print(f"DNAMultiSeqDataset: Created {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _crop_resize_dna_specific(self, img, depthmap, K, resolution, rng=None):
        """
        Custom crop/resize because DNA images are 1024x1224 (0.83 AR) 
        and we want 384x512 (0.75 AR) or similar Portrait AR.
        Base class default might not handle the aspect ratio crop ideally.
        """
        if not isinstance(img, PIL.Image.Image):
             img = PIL.Image.fromarray(img)
        
        W_orig, H_orig = img.size
        W_target, H_target = resolution # (384, 512)
        
        # 1. Aspect Ratio Crop (Center)
        target_ar = W_target / H_target
        orig_ar = W_orig / H_orig
        
        if orig_ar > target_ar: # Too wide
            new_w = int(H_orig * target_ar)
            left = (W_orig - new_w) // 2
            img = img.crop((left, 0, left + new_w, H_orig))
            if depthmap is not None: depthmap = depthmap[:, left:left+new_w]
            K[0, 2] -= left
        elif orig_ar < target_ar: # Too tall
            new_h = int(W_orig / target_ar)
            top = (H_orig - new_h) // 2
            img = img.crop((0, top, W_orig, top + new_h))
            if depthmap is not None: depthmap = depthmap[top:top+new_h, :]
            K[1, 2] -= top
            
        # 2. Resize
        W_cropped, H_cropped = img.size
        scale_x = W_target / W_cropped
        scale_y = H_target / H_cropped
        img = img.resize((W_target, H_target), PIL.Image.LANCZOS)
        if depthmap is not None:
            depthmap = cv2.resize(depthmap, (W_target, H_target), interpolation=cv2.INTER_NEAREST)
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        
        # 3. Zoom Augmentation (Optional)
        if rng is not None and rng.rand() < 0.5:
             zoom = rng.uniform(1.0, 1.2)
             if zoom > 1.0:
                 crop_w, crop_h = int(W_target/zoom), int(H_target/zoom)
                 x0 = rng.randint(0, W_target - crop_w)
                 y0 = rng.randint(0, H_target - crop_h)
                 img = img.crop((x0, y0, x0+crop_w, y0+crop_h))
                 if depthmap is not None: depthmap = depthmap[y0:y0+crop_h, x0:x0+crop_w]
                 K[0,2] -= x0; K[1,2] -= y0
                 img = img.resize((W_target, H_target), PIL.Image.LANCZOS)
                 if depthmap is not None: depthmap = cv2.resize(depthmap, (W_target, H_target), interpolation=cv2.INTER_NEAREST)
                 K[0,:] *= zoom; K[1,:] *= zoom
                 
        return img, depthmap, K

    def _get_views(self, idx, resolution, rng, num_views):
        # BaseMultiViewDataset signature
        seq_name, frame_name = self.samples[idx]
        frame_dir = os.path.join(self.dataset_location, seq_name, frame_name)
        
        rgb_dir = os.path.join(frame_dir, 'rgbs')
        depth_dir = os.path.join(frame_dir, 'depth_rendered')
        cameras_path = os.path.join(frame_dir, 'cameras.json')
        
        all_views = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        num_available = len(all_views)
        
        if num_available == 0:
            # Fallback (recurse carefully)
            return self._get_views(rng.randint(len(self.samples)), resolution, rng, num_views)
            
        # Select Views
        if num_views is None: num_views = self.S
        
        if num_available <= num_views:
            selected_indices = np.arange(num_available)
        elif self.view_sample_mode == 'random':
             selected_indices = np.sort(rng.choice(num_available, num_views, replace=False))
        else:
             selected_indices = np.arange(min(num_available, num_views))
             
        # Load Cameras
        import json
        intrinsics_dict = {}
        extrinsics_dict = {}
        if os.path.exists(cameras_path):
            with open(cameras_path, 'r') as f: cams = json.load(f)
            cam_list = list(cams.values()) if isinstance(cams, dict) else cams
            for cam in cam_list:
                vid = int(cam.get('img_name', cam.get('id', 0)))
                K = np.array([[cam['fx'], 0, cam['width']/2], [0, cam['fy'], cam['height']/2], [0,0,1]], dtype=np.float32)
                intrinsics_dict[vid] = K
                if 'position' in cam and 'rotation' in cam:
                    c2w = np.eye(4, dtype=np.float32)
                    c2w[:3,:3] = np.array(cam['rotation'], dtype=np.float32)
                    c2w[:3,3] = np.array(cam['position'], dtype=np.float32)
                    extrinsics_dict[vid] = c2w
        
        views = []
        
        for i, view_idx in enumerate(selected_indices):
            rgb_path = all_views[view_idx]
            vid_str = os.path.splitext(os.path.basename(rgb_path))[0]
            vid = int(vid_str)
            
            # RGB
            img = imread_cv2(rgb_path) # Returns RGB numpy
            
            # Depth
            depthmap = None
            gt_path_png = os.path.join(depth_dir, f"{vid_str}.png")
            gt_path_npy = os.path.join(depth_dir, f"{vid_str}.npy")
            if os.path.exists(gt_path_png):
                 d = cv2.imread(gt_path_png, cv2.IMREAD_UNCHANGED)
                 if d is not None: depthmap = d.astype(np.float32) / 1000.0
            elif os.path.exists(gt_path_npy):
                 depthmap = np.load(gt_path_npy).astype(np.float32)
            
            if depthmap is None:
                 # Should probably invalidate or skip? For now zero.
                 depthmap = np.zeros(img.shape[:2], dtype=np.float32)
            
            # Intrinsics
            K = intrinsics_dict.get(vid, np.eye(3, dtype=np.float32))
            
            # Metadata
            c2w = extrinsics_dict.get(vid, np.eye(4, dtype=np.float32))
            
            # Transform
            img_pil, depthmap, K = self._crop_resize_dna_specific(img, depthmap, K.copy(), resolution, rng)
            img_np = np.array(img_pil) # RGB
            
            # Masks
            # For Dust3r, usually we set img_mask (valid pixels) and ray_mask (pixels to supervise?)
            # BaseMultiViewDataset has helper `get_img_and_ray_masks`
            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, i, rng)
            
            # Refine validity based on depth > 0 if metric
            valid_depth = depthmap > 0
            img_mask = img_mask & valid_depth
            ray_mask = ray_mask & valid_depth
            
            views.append(dict(
                img=img_np,
                depthmap=depthmap,
                camera_pose=c2w,
                camera_intrinsics=K,
                dataset=self.dataset_label,
                label=f"{seq_name}/{frame_name}",
                instance=f"{vid_str}",
                is_metric=self.is_metric,
                is_video=False,
                start_crop=np.array([0, 0], dtype=np.float32), # Placeholder if not tracking crop
                img_mask=img_mask,
                ray_mask=ray_mask
            ))
            
        return views

# Alias for config
DNA_Multi = DNAMultiSeqDataset

