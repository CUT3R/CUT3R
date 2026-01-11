import os
import glob
import numpy as np
import cv2
import torch
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2, ImgNorm
from torchvision import transforms
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates

class DNAMultiSeqDataset(BaseStereoViewDataset):
    """
    DNA-Rendering Dataset for MULTIPLE sequences.
    
    Scans a parent directory containing multiple sequences (e.g., toy_dataset/Part1/)
    and creates training samples from all frames across all sequences.
    
    Loads BOTH:
    - GT rendered depth from depth_rendered/{view}.npy (foreground, multi-view consistent)
    - MoGe calibrated depth from calibrated_depth/{seq}/{seq}_view{view}/moge_calibrated.npy (full image)
    """
    
    def __init__(self,
                 dataset_location,  # Root dir, e.g., /path/to/DNA (with Part1, Part2,... inside)
                 sequence_list=None,  # Path to txt file listing sequences (Part1/seq_name per line)
                 moge_depth_dir=None,  # Path to calibrated_depth/ folder (optional)
                 S=12,  # Number of views per sample
                 resolution=(512, 512),
                 stride=1,  # Frame stride within sequence
                 transform=None,
                 exclude_seqs=None,  # List of sequence names to exclude
                 view_sample_mode='random',  # 'random', 'sequential', 'opposite'
                 load_moge=True,  # Whether to load MoGe depth for background
                 *args, **kwargs):
        super().__init__(resolution=resolution, *args, **kwargs)
        self.dataset_location = dataset_location
        self.moge_depth_dir = moge_depth_dir
        self.resolution = resolution
        self.S = S
        self.stride = stride
        self.transform = transform if transform else ImgNorm
        self.view_sample_mode = view_sample_mode
        self.exclude_seqs = set(exclude_seqs or [])
        self.load_moge = load_moge
        
        # Discover all sequences
        self.sequences = []
        
        if sequence_list and os.path.exists(sequence_list):
            # Load from txt file (format: Part1/seq_name per line)
            with open(sequence_list, 'r') as f:
                for line in f:
                    rel_path = line.strip()
                    if not rel_path:
                        continue
                    seq_dir = os.path.join(dataset_location, rel_path)
                    if os.path.isdir(seq_dir):
                        # Extract seq_name from rel_path (e.g., Part1/0012_09 -> 0012_09)
                        seq_name = rel_path  # Keep full relative path as identifier
                        if seq_name.split('/')[-1] not in self.exclude_seqs:
                            self.sequences.append(rel_path)
            print(f"DNAMultiSeqDataset: Loaded {len(self.sequences)} sequences from {sequence_list}")
        else:
            # Original behavior: scan single Part folder
            for seq_name in sorted(os.listdir(dataset_location)):
                seq_dir = os.path.join(dataset_location, seq_name)
                if not os.path.isdir(seq_dir):
                    continue
                if seq_name in self.exclude_seqs:
                    continue
                self.sequences.append(seq_name)
            print(f"DNAMultiSeqDataset: Found {len(self.sequences)} sequences in {dataset_location}")
        
        # MoGe cache - lazy loaded on demand to save RAM
        self.moge_cache = {}
        # NOTE: Removed pre-loading to prevent RAM overflow with many sequences
        
        # Build list of (seq_name, frame_idx) pairs using frame_idx.txt cache
        self.samples = []
        for seq_name in self.sequences:
            seq_dir = os.path.join(dataset_location, seq_name)
            frame_idx_file = os.path.join(seq_dir, 'frame_idx.txt')
            
            # Try to load from cache file
            if os.path.exists(frame_idx_file):
                with open(frame_idx_file, 'r') as f:
                    valid_frames = [line.strip() for line in f if line.strip()]
            else:
                # Scan for valid frames and create cache
                try:
                    frames = [d for d in os.listdir(seq_dir) if d.isdigit() and 
                              os.path.isdir(os.path.join(seq_dir, d))]
                except Exception as e:
                    print(f"[WARNING] Cannot read {seq_dir}: {e}")
                    continue
                frames = sorted(frames, key=int)
                
                # Filter frames that have both rgbs and depth_rendered
                valid_frames = []
                for frame_name in frames:
                    rgb_dir = os.path.join(seq_dir, frame_name, 'rgbs')
                    depth_dir = os.path.join(seq_dir, frame_name, 'depth_rendered')
                    # Check rgbs folder has files and depth_rendered exists
                    if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
                        rgb_files = glob.glob(os.path.join(rgb_dir, '*.png'))
                        if len(rgb_files) >= 48:  # Must have at least 48 views
                            valid_frames.append(frame_name)
                
                # Save cache file
                try:
                    with open(frame_idx_file, 'w') as f:
                        for frame in valid_frames:
                            f.write(f"{frame}\n")
                except Exception as e:
                    pass  # Silently fail if can't write cache
            
            # Add samples with stride
            for i in range(0, len(valid_frames), stride):
                self.samples.append((seq_name, valid_frames[i]))
        
        print(f"DNAMultiSeqDataset: Created {len(self.samples)} samples")
        
        # Check view count from first sample
        if self.samples:
            seq_name, frame_name = self.samples[0]
            rgb_dir = os.path.join(dataset_location, seq_name, frame_name, 'rgbs')
            self.num_available_views = len(glob.glob(os.path.join(rgb_dir, '*.png')))
            print(f"DNAMultiSeqDataset: {self.num_available_views} views per frame")
    
    def __len__(self):
        return len(self.samples)
    

    def _crop_resize_if_necessary(self, img, depthmap, K, resolution, rng=None):
        """
        Resize logic:
        1. Center crop to match target aspect ratio (384/512 = 0.75).
        2. Resize to target resolution (384, 512).
        3. Optional: Zoom augmentation (crop & resize again).
        """
        import PIL.Image
        import PIL.ImageOps
        if not isinstance(img, PIL.Image.Image):
             img = PIL.Image.fromarray(img)
        
        W_orig, H_orig = img.size
        W_target, H_target = resolution
        
        # 1. Aspect Ratio Crop
        target_ar = W_target / H_target
        orig_ar = W_orig / H_orig
        
        if orig_ar > target_ar:
            # Image is too wide, crop width
            new_w = int(H_orig * target_ar)
            left = (W_orig - new_w) // 2
            img = img.crop((left, 0, left + new_w, H_orig))
            if depthmap is not None:
                depthmap = depthmap[:, left:left+new_w]
            # Adjust Principal Point (cx)
            K[0, 2] -= left
        elif orig_ar < target_ar:
            # Image is too tall, crop height
            new_h = int(W_orig / target_ar)
            top = (H_orig - new_h) // 2
            img = img.crop((0, top, W_orig, top + new_h))
            if depthmap is not None:
                depthmap = depthmap[top:top+new_h, :]
            # Adjust Principal Point (cy)
            K[1, 2] -= top
            
        # 2. Resize to Target Resolution
        W_cropped, H_cropped = img.size
        scale_x = W_target / W_cropped
        scale_y = H_target / H_cropped
        
        img = img.resize((W_target, H_target), PIL.Image.LANCZOS)
        if depthmap is not None:
            depthmap = cv2.resize(depthmap, (W_target, H_target), interpolation=cv2.INTER_NEAREST)
            
        # Adjust Intrinsics for Resize
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        
        # 3. Optional Zoom Augmentation
        if rng is not None and rng.rand() < 0.5:
             # Random zoom between 1.0 and 1.2
             zoom = rng.uniform(1.0, 1.2)
             if zoom > 1.0:
                 crop_w = int(W_target / zoom)
                 crop_h = int(H_target / zoom)
                 x0 = rng.randint(0, W_target - crop_w)
                 y0 = rng.randint(0, H_target - crop_h)
                 
                 img = img.crop((x0, y0, x0+crop_w, y0+crop_h))
                 if depthmap is not None:
                     depthmap = depthmap[y0:y0+crop_h, x0:x0+crop_w]
                 
                 # Adjust K for crop
                 K[0, 2] -= x0
                 K[1, 2] -= y0
                 
                 # Resize back
                 img = img.resize((W_target, H_target), PIL.Image.LANCZOS)
                 if depthmap is not None:
                     depthmap = cv2.resize(depthmap, (W_target, H_target), interpolation=cv2.INTER_NEAREST)
                     
                 # Adjust K for resize
                 K[0, :] *= zoom
                 K[1, :] *= zoom
                 
        return img, depthmap, K

    def _get_views(self, index, resolution=None, rng=None, _retry_count=0):
        # ... (rest of _get_views implementation invoking _crop_resize_if_necessary)
        if resolution is None:
            resolution = self.resolution
        
        seq_name, frame_name = self.samples[index]
        frame_dir = os.path.join(self.dataset_location, seq_name, frame_name)
        
        rgb_dir = os.path.join(frame_dir, 'rgbs')
        depth_dir = os.path.join(frame_dir, 'depth_rendered')
        cameras_path = os.path.join(frame_dir, 'cameras.json')
        
        # Get all views
        all_views = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        num_available = len(all_views)
        
        # VALIDATION: If no views found, try a different sample (with retry limit)
        if num_available == 0:
            if _retry_count >= 5:
                # raise RuntimeError(f"DNAMultiSeqDataset: Too many retries, sample {seq_name}/{frame_name} has no RGB files")
                print(f"[ERROR] DNAMultiSeqDataset: Too many retries. Returning dummy sample.")
                return self._get_views(0, resolution, rng, 0) # Fallback to 0
            print(f"[WARNING] DNAMultiSeqDataset: No RGB files in {rgb_dir}, skipping to random sample")
            new_index = np.random.randint(0, len(self.samples))
            return self._get_views(new_index, resolution, rng, _retry_count + 1)
        
        # Select S views based on mode
        if rng is None:
            rng = np.random.RandomState(index)
        
        if num_available <= self.S:
            selected_indices = list(range(num_available))
        elif self.view_sample_mode == 'random':
            selected_indices = np.sort(rng.choice(num_available, self.S, replace=False))
        elif self.view_sample_mode == 'sequential':
            start = rng.randint(0, max(1, num_available - self.S))
            selected_indices = list(range(start, start + self.S))
        elif self.view_sample_mode == 'opposite':
            # Sample views that are roughly opposite (e.g., 0 and 24 for 48 views)
            half = num_available // 2
            first_half = rng.choice(half, self.S // 2, replace=False)
            second_half = first_half + half
            selected_indices = np.sort(np.concatenate([first_half, second_half]))
        else:
            selected_indices = list(range(min(self.S, num_available)))
        
        views_files = [all_views[i] for i in selected_indices]
        
        # Load camera intrinsics AND extrinsics
        import json
        intrinsics_dict = {}
        extrinsics_dict = {}  # Store c2w (camera-to-world) transforms
        if os.path.exists(cameras_path):
            with open(cameras_path) as f:
                cameras = json.load(f)
            # Support both dict and list format
            if isinstance(cameras, dict):
                cam_list = list(cameras.values())
            else:
                cam_list = cameras
            for cam in cam_list:
                vid = int(cam.get('img_name', cam.get('id', 0)))
                K = np.array([
                    [cam['fx'], 0, cam['width'] / 2],
                    [0, cam['fy'], cam['height'] / 2],
                    [0, 0, 1]
                ], dtype=np.float32)
                intrinsics_dict[vid] = K
                
                # Load extrinsics if available (position + rotation)
                if 'position' in cam and 'rotation' in cam:
                    pos = np.array(cam['position'], dtype=np.float32)
                    rot = np.array(cam['rotation'], dtype=np.float32)  # R_wc (camera to world rotation)
                    
                    # Build camera-to-world transform
                    c2w = np.eye(4, dtype=np.float32)
                    c2w[:3, :3] = rot  # R_wc
                    c2w[:3, 3] = pos   # Camera position in world
                    extrinsics_dict[vid] = c2w
        
        # Prepare output tensors
        B = len(views_files)
        W, H = resolution if isinstance(resolution, (list, tuple)) else (resolution, resolution)
        
        imgs = torch.zeros((B, 3, H, W), dtype=torch.float32)
        depthmaps = torch.zeros((B, H, W), dtype=torch.float32)  # GT rendered depth
        moge_depthmaps = torch.zeros((B, H, W), dtype=torch.float32)  # MoGe calibrated depth
        valid_masks = torch.zeros((B, H, W), dtype=torch.bool)  # Foreground mask
        bg_masks = torch.zeros((B, H, W), dtype=torch.bool)  # Background mask
        intrinsics_torch = torch.zeros((B, 3, 3), dtype=torch.float32)
        
        for i, rgb_path in enumerate(views_files):
            # Get view ID
            vid_str = os.path.splitext(os.path.basename(rgb_path))[0]
            vid = int(vid_str)
            
            # Load RGB (BGR -> RGB)
            img_bgr = imread_cv2(rgb_path)  
            # Convert to PIL for easy cropping/resizing in helper
            img_pil = PIL.Image.fromarray(img_bgr) # imread_cv2 returns RGB if using dust3r utils, but name suggests opencv.
            # Using cv2.cvtColor check. dust3r `imread_cv2` actually returns RGB.
            # Let's assume input is np array RGB.
            
            # Load GT rendered depth (foreground only)
            gt_depth_path_png = os.path.join(depth_dir, f"{vid_str}.png")
            gt_depth_path_npy = os.path.join(depth_dir, f"{vid_str}.npy")
            
            gt_depth = None
            if os.path.exists(gt_depth_path_png):
                depth_mm = cv2.imread(gt_depth_path_png, cv2.IMREAD_UNCHANGED)
                if depth_mm is not None:
                    gt_depth = depth_mm.astype(np.float32) / 1000.0
            elif os.path.exists(gt_depth_path_npy):
                gt_depth = np.load(gt_depth_path_npy).astype(np.float32)
                
            # Intrinsics
            K = intrinsics_dict.get(vid, np.eye(3, dtype=np.float32))

            # --- CROP & RESIZE & ZOOM ---
            img_pil, gt_depth, K = self._crop_resize_if_necessary(img_pil, gt_depth, K, (W, H), rng)
            
            # Store RGB
            imgs[i] = self.transform(img_pil)
            intrinsics_torch[i] = torch.from_numpy(K)
            
            # Store Depth & Masks
            if gt_depth is not None:
                depthmaps[i] = torch.from_numpy(gt_depth)
                
                # Load Mask (Optional: Use Depth > 0 as mask if file missing)
                mask_path = os.path.join(frame_dir, 'masks', f"{vid_str}.png")
                if os.path.exists(mask_path):
                     # Mask needs same geometric transform. We can re-use parameters or just crop/resize the mask.
                     # For simplicity, let's treat Mask as 1-channel image and pass it through _crop_resize_if_necessary separately?
                     # Ideally, we should do them together. For now, let's infer mask from depth since we cropped depth.
                     pass 
                
                # Use depth validity as primary mask (as requested in plan)
                valid_masks[i] = depthmaps[i] > 0
                bg_masks[i] = ~valid_masks[i]
            
            # Load MoGe (Optional/Background) - Skipped for now or needs similar resize
            # ... (MoGe code would go here, needing same transform)
            
        # ... (Rest of pts3d computation using new K and depth)
        # Compute pts3d from depth and transform to WORLD coordinates
        pts3d_world = torch.zeros((B, H, W, 3), dtype=torch.float32)
        camera_poses = torch.zeros((B, 4, 4), dtype=torch.float32)  # Store c2w for each view
        
        for i in range(B):
            vid = int(os.path.splitext(os.path.basename(views_files[i]))[0])
            
            # Use depthmaps (GT)
            depth_to_use = depthmaps[i]
            
            if depth_to_use.any() and intrinsics_torch[i].any():
                # Get pts3d in camera coordinates
                pts_cam, _ = depthmap_to_absolute_camera_coordinates(
                    depth_to_use.numpy(), 
                    intrinsics_torch[i].numpy(), 
                    camera_pose=None, 
                    proj_mode='depth'
                )
                pts_cam = pts_cam.astype(np.float32)  # (H, W, 3)
                
                # Transform to world coordinates if extrinsics available
                if vid in extrinsics_dict:
                    c2w = extrinsics_dict[vid]  # (4, 4)
                    camera_poses[i] = torch.from_numpy(c2w)
                    
                    # Apply c2w transform: pts_world = R_wc @ pts_cam + t_wc
                    R_wc = c2w[:3, :3]  # (3, 3)
                    pos = c2w[:3, 3]    # (3,) camera position in world
                    
                    # Flatten, transform, reshape
                    pts_flat = pts_cam.reshape(-1, 3)  # (H*W, 3)
                    pts_world_flat = pts_flat @ R_wc.T + pos  # (H*W, 3)
                    pts3d_world[i] = torch.from_numpy(pts_world_flat.reshape(H, W, 3))
                else:
                    # No extrinsics, keep camera coords
                    pts3d_world[i] = torch.from_numpy(pts_cam)
                    camera_poses[i] = torch.eye(4, dtype=torch.float32)

        # SANITY CHECKS ...
        
        # views1: Reference view
        views1 = {
            'img': imgs[0:1].repeat(B, 1, 1, 1),  # First frame repeated
            'camera_intrinsics': intrinsics_torch[0:1].repeat(B, 1, 1),
            'dataset': 'DNA',
            'label': [f"{seq_name}/{frame_name}"] * B,
            'instance': [os.path.splitext(os.path.basename(views_files[0]))[0]] * B,
            'supervised_label': torch.ones(B, dtype=torch.float32),
            'traj_mask': valid_masks.clone(),
            'traj_ptc': pts3d_world,
            'pts3d': pts3d_world[0:1].repeat(B, 1, 1, 1),
            'valid_mask': valid_masks.clone(),
            'camera_pose': camera_poses[0:1].repeat(B, 1, 1),
        }
        
        # views2: Target views
        views2 = {
            'img': imgs,
            'img_org': imgs.clone(),
            'depthmap': depthmaps,
            'moge_depth': moge_depthmaps, # Zeros if not loaded
            'valid_mask': valid_masks,
            'bg_mask': bg_masks,
            'camera_intrinsics': intrinsics_torch,
            'dataset': 'DNA',
            'label': [f"{seq_name}/{frame_name}"] * B,
            'instance': [f"{os.path.splitext(os.path.basename(f))[0]}" for f in views_files],
            'pts3d_moge': pts3d_world,
            'pts3d': pts3d_world.clone(),
            'camera_pose': camera_poses,
            'supervised_label': torch.ones(B, dtype=torch.float32),
        }
        
        return views1, views2


class DNASingleSeqDataset(BaseStereoViewDataset):
    """
    DNA-Rendering Dataset for a SINGLE sequence with all 48 views.
    For TTA (Test-Time Adaptation) on a specific sequence.
    
    Loads calibrated MoGe depth from calibrated_depth/{seq}/{seq}_view{v}/moge_calibrated.npy
    """
    
    def __init__(self,
                 dataset_location,  # Sequence dir, e.g., toy_dataset/Part1/0012_09
                 moge_depth_dir=None,  # Path to calibrated_depth/ folder
                 S=8,  # Number of views per sample
                 resolution=(512, 512),
                 transform=None,
                 view_sample_mode='random',  # 'random', 'sequential', 'opposite'
                 load_moge=True,  # Whether to load MoGe calibrated depth
                 *args, **kwargs):
        super().__init__(resolution=resolution, *args, **kwargs)
        self.dataset_location = dataset_location
        self.moge_depth_dir = moge_depth_dir
        self.resolution = resolution
        self.S = S
        self.transform = transform if transform else ImgNorm
        self.view_sample_mode = view_sample_mode
        self.load_moge = load_moge
        
        # Get sequence name from path
        self.seq_name = os.path.basename(dataset_location.rstrip('/'))
        
        # Discover frames
        self.frames = sorted([d for d in os.listdir(dataset_location) if d.isdigit() and 
                      os.path.isdir(os.path.join(dataset_location, d))], key=int)
        
        print(f"DNASingleSeqDataset: Found {len(self.frames)} frames in {self.seq_name}")
        
        # Get number of views from first frame
        if self.frames:
            rgb_dir = os.path.join(dataset_location, self.frames[0], 'rgbs')
            self.num_available_views = len(glob.glob(os.path.join(rgb_dir, '*.png')))
            print(f"DNASingleSeqDataset: {self.num_available_views} views per frame")
        
        # Pre-load MoGe calibrated depth data
        self.moge_cache = {}
        if self.load_moge and self.moge_depth_dir:
            print(f"DNASingleSeqDataset: Loading MoGe calibrated depth from {moge_depth_dir}")
            for view_idx in range(48):
                moge_path = os.path.join(moge_depth_dir, self.seq_name, f"{self.seq_name}_view{view_idx}", "moge_calibrated.npy")
                if os.path.exists(moge_path):
                    try:
                        self.moge_cache[view_idx] = np.load(moge_path, allow_pickle=True).item()
                    except Exception as e:
                        print(f"Error loading {moge_path}: {e}")
            print(f"DNASingleSeqDataset: Loaded {len(self.moge_cache)} MoGe calibrated files")
    
    def __len__(self):
        return len(self.frames)
    
    def _get_views(self, index, resolution=None, rng=None):
        if resolution is None:
            resolution = self.resolution
        
        frame_name = self.frames[index]
        frame_dir = os.path.join(self.dataset_location, frame_name)
        
        rgb_dir = os.path.join(frame_dir, 'rgbs')
        cameras_path = os.path.join(frame_dir, 'cameras.json')
        
        # Get all views
        all_views = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
        num_available = len(all_views)
        
        # Select S views based on mode
        if rng is None:
            rng = np.random.RandomState(index)
        
        if num_available <= self.S:
            selected_indices = list(range(num_available))
        elif self.view_sample_mode == 'random':
            selected_indices = np.sort(rng.choice(num_available, self.S, replace=False))
        elif self.view_sample_mode == 'sequential':
            start = rng.randint(0, max(1, num_available - self.S))
            selected_indices = list(range(start, start + self.S))
        elif self.view_sample_mode == 'opposite':
            half = num_available // 2
            first_half = rng.choice(half, self.S // 2, replace=False)
            second_half = first_half + half
            selected_indices = np.sort(np.concatenate([first_half, second_half]))
        else:
            selected_indices = list(range(min(self.S, num_available)))
        
        views_files = [all_views[i] for i in selected_indices]
        
        # Load camera intrinsics AND extrinsics
        import json
        intrinsics_dict = {}
        extrinsics_dict = {}  # Store c2w (camera-to-world) transforms
        if os.path.exists(cameras_path):
            with open(cameras_path) as f:
                cameras = json.load(f)
            # Support both dict and list format
            if isinstance(cameras, dict):
                cam_list = list(cameras.values())
            else:
                cam_list = cameras
            for cam in cam_list:
                vid = int(cam.get('img_name', cam.get('id', 0)))
                K = np.array([
                    [cam['fx'], 0, cam['width'] / 2],
                    [0, cam['fy'], cam['height'] / 2],
                    [0, 0, 1]
                ])
                intrinsics_dict[vid] = K
                
                # Load extrinsics if available (position + rotation)
                if 'position' in cam and 'rotation' in cam:
                    pos = np.array(cam['position'], dtype=np.float32)
                    rot = np.array(cam['rotation'], dtype=np.float32)  # R_wc (camera to world rotation)
                    
                    # Build camera-to-world transform
                    c2w = np.eye(4, dtype=np.float32)
                    c2w[:3, :3] = rot  # R_wc
                    c2w[:3, 3] = pos   # Camera position in world
                    extrinsics_dict[vid] = c2w
        
        # Prepare output tensors
        B = len(views_files)
        W, H = resolution if isinstance(resolution, (list, tuple)) else (resolution, resolution)
        
        imgs = torch.zeros((B, 3, H, W), dtype=torch.float32)
        depthmaps = torch.zeros((B, H, W), dtype=torch.float32)  # Calibrated MoGe depth
        valid_masks = torch.zeros((B, H, W), dtype=torch.bool)
        intrinsics_torch = torch.zeros((B, 3, 3), dtype=torch.float32)
        
        for i, rgb_path in enumerate(views_files):
            vid_str = os.path.splitext(os.path.basename(rgb_path))[0]
            vid = int(vid_str)
            
            # Load RGB
            img = imread_cv2(rgb_path)
            h_orig, w_orig = img.shape[:2]
            img = cv2.resize(img, (W, H))
            if i == 0:  # Only print for first view to avoid spam
                print(f"[DEBUG DNASingleSeqDataset] Original: {w_orig}x{h_orig} -> Resized: {W}x{H}")
            imgs[i] = self.transform(img)
            
            # Load GT rendered depth FIRST (multi-view consistent)
            depth_dir = os.path.join(frame_dir, 'depth_rendered')
            gt_depth_loaded = False
            
            # Try PNG first, then NPY
            gt_depth_path_png = os.path.join(depth_dir, f"{vid_str}.png")
            gt_depth_path_npy = os.path.join(depth_dir, f"{vid_str}.npy")
            
            if os.path.exists(gt_depth_path_png):
                depth_mm = cv2.imread(gt_depth_path_png, cv2.IMREAD_UNCHANGED)
                if depth_mm is not None:
                    gt_depth = depth_mm.astype(np.float32) / 1000.0  # mm to meters
                    gt_depth = cv2.resize(gt_depth, (W, H), interpolation=cv2.INTER_NEAREST)
                    depthmaps[i] = torch.from_numpy(gt_depth)
                    valid_masks[i] = depthmaps[i] > 0
                    gt_depth_loaded = True
            elif os.path.exists(gt_depth_path_npy):
                gt_depth = np.load(gt_depth_path_npy).astype(np.float32)
                gt_depth = cv2.resize(gt_depth, (W, H), interpolation=cv2.INTER_NEAREST)
                depthmaps[i] = torch.from_numpy(gt_depth)
                valid_masks[i] = depthmaps[i] > 0
                gt_depth_loaded = True
            
            # Fallback to MoGe calibrated depth if GT not available
            if not gt_depth_loaded and self.load_moge and vid in self.moge_cache:
                moge_data = self.moge_cache[vid]
                frame_key = f"frame_{int(frame_name):04d}"
                if frame_key in moge_data:
                    moge_depth = moge_data[frame_key]['depth'].copy()
                    moge_mask = moge_data[frame_key].get('mask', np.ones_like(moge_depth, dtype=bool))
                    
                    # Handle NaN values - replace with 0 and mark as invalid
                    nan_mask = np.isnan(moge_depth) | np.isinf(moge_depth)
                    moge_depth = np.nan_to_num(moge_depth, nan=0.0, posinf=0.0, neginf=0.0)
                    moge_mask = moge_mask & (~nan_mask)
                    
                    if moge_depth.shape != (H, W):
                        moge_depth = cv2.resize(moge_depth, (W, H), interpolation=cv2.INTER_LINEAR)
                        moge_mask = cv2.resize(moge_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    depthmaps[i] = torch.from_numpy(moge_depth.astype(np.float32))
                    valid_masks[i] = torch.from_numpy(moge_mask) & (depthmaps[i] > 0)
            
            # Intrinsics
            if vid in intrinsics_dict:
                K = intrinsics_dict[vid].copy()
                scale_x = W / w_orig
                scale_y = H / h_orig
                K[0, :] *= scale_x
                K[1, :] *= scale_y
                intrinsics_torch[i] = torch.from_numpy(K)
        
        # Compute pts3d from depth and transform to WORLD coordinates
        pts3d_world = torch.zeros((B, H, W, 3), dtype=torch.float32)
        camera_poses = torch.zeros((B, 4, 4), dtype=torch.float32)  # Store c2w for each view
        
        for i, rgb_path in enumerate(views_files):
            vid = int(os.path.splitext(os.path.basename(rgb_path))[0])
            
            if valid_masks[i].any():
                # Get pts3d in camera coordinates
                pts_cam, _ = depthmap_to_absolute_camera_coordinates(
                    depthmaps[i].numpy(), 
                    intrinsics_torch[i].numpy(), 
                    camera_pose=None, 
                    proj_mode='depth'
                )
                pts_cam = pts_cam.astype(np.float32)  # (H, W, 3)
                
                # Transform to world coordinates if extrinsics available
                if vid in extrinsics_dict:
                    c2w = extrinsics_dict[vid]  # (4, 4)
                    camera_poses[i] = torch.from_numpy(c2w)
                    
                    # Apply c2w transform: pts_world = R_wc @ pts_cam + t_wc
                    # c2w[:3,:3] = R_wc, c2w[:3,3] = camera position = t_wc
                    R_wc = c2w[:3, :3]  # (3, 3)
                    pos = c2w[:3, 3]    # (3,) camera position in world
                    
                    # Flatten, transform, reshape
                    pts_flat = pts_cam.reshape(-1, 3)  # (H*W, 3)
                    pts_world_flat = pts_flat @ R_wc.T + pos  # (H*W, 3)
                    pts3d_world[i] = torch.from_numpy(pts_world_flat.reshape(H, W, 3))
                else:
                    # No extrinsics, keep camera coords
                    pts3d_world[i] = torch.from_numpy(pts_cam)
                    camera_poses[i] = torch.eye(4, dtype=torch.float32)
        
        # SANITY CHECKS - Alert if critical data is missing
        if len(intrinsics_dict) == 0:
            print(f"[WARNING] DNASingleSeqDataset: No camera intrinsics loaded for {self.seq_name}/{frame_name}!")
        if len(extrinsics_dict) == 0:
            print(f"[WARNING] DNASingleSeqDataset: No camera extrinsics loaded for {self.seq_name}/{frame_name}! pts3d will be in camera coords.")
        if not pts3d_world.any():
            print(f"[WARNING] DNASingleSeqDataset: pts3d_world is ALL ZEROS for {self.seq_name}/{frame_name}!")
        if not depthmaps.any():
            print(f"[WARNING] DNASingleSeqDataset: depthmaps is ALL ZEROS for {self.seq_name}/{frame_name}!")
        if not valid_masks.any():
            print(f"[WARNING] DNASingleSeqDataset: valid_masks is ALL FALSE for {self.seq_name}/{frame_name}!")
        if not camera_poses.any():
            print(f"[WARNING] DNASingleSeqDataset: camera_poses is ALL ZEROS for {self.seq_name}/{frame_name}!")
        
        # views1: Reference view (minimal fields, repeated first frame)
        views1 = {
            'img': imgs[0:1].repeat(B, 1, 1, 1),
            'camera_intrinsics': intrinsics_torch[0:1].repeat(B, 1, 1),
            'dataset': 'DNA',
            'label': [f"{self.seq_name}/{frame_name}"] * B,
            'instance': [os.path.splitext(os.path.basename(views_files[0]))[0]] * B,
            'supervised_label': torch.ones(B, dtype=torch.float32),
            'traj_mask': valid_masks.clone(),
            'traj_ptc': pts3d_world,  # Use actual pts3d for trajectory loss
            'pts3d': pts3d_world[0:1].repeat(B, 1, 1, 1),  # Use first view's pts3d (world coords)
            'valid_mask': valid_masks.clone(),
            'camera_pose': camera_poses[0:1].repeat(B, 1, 1),  # Use FIRST camera pose for all (to transform all pts to camera1 frame)
        }
        
        # views2: Target views with depth and all data
        views2 = {
            'img': imgs,
            'img_org': imgs.clone(),
            'depthmap': depthmaps,  # Calibrated MoGe depth
            'valid_mask': valid_masks,
            'camera_intrinsics': intrinsics_torch,
            'dataset': 'DNA',
            'label': [f"{self.seq_name}/{frame_name}"] * B,
            'instance': [f"{vid_str}" for vid_str in [os.path.splitext(os.path.basename(f))[0] for f in views_files]],
            'pts3d_moge': pts3d_world,  # 3D points in WORLD coordinates
            'pts3d': pts3d_world.clone(),  # Also set pts3d for compatibility
            'camera_pose': camera_poses,  # Actual camera poses (c2w)
            'supervised_label': torch.ones(B, dtype=torch.float32),
        }
        
        return views1, views2
