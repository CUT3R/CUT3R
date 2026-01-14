import argparse
import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from dust3r.datasets.dna import DNAMultiSeqDataset

def main():
    parser = argparse.ArgumentParser(description="Sample specific view from DNA dataset as MP4")
    parser.add_argument("--data_path", type=str, default="/mnt/raid/lin/nas-train/Concat_Dataset/DNA", help="Path to dataset")
    parser.add_argument("--view_id", type=int, default=22, help="Specific view ID to sample (e.g. 22)")
    parser.add_argument("--output_dir", type=str, default="output_sample", help="Directory to save output")
    parser.add_argument("--width", type=int, default=384, help="Width")
    parser.add_argument("--height", type=int, default=512, help="Height")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset from {args.data_path}...")
    print(f"Requesting view ID: {args.view_id} for a full sequence video (30fps).")

    # Initialize dataset
    # We pass S=1, stride=1 to get all frames available.
    dataset = DNAMultiSeqDataset(
        dataset_location=args.data_path,
        sequence_list=None,
        S=1, 
        resolution=(args.width, args.height),
        stride=1,
        specific_view_ids=[args.view_id],
        view_sample_mode='random', # Overridden
        transform=lambda x: np.array(x),
        use_zoom_aug=False # Disable jittery zoom for video 
    )

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    # Sample 10 sequences
    num_sequences_to_sample = 10
    sequences_processed = 0

    unique_seqs = sorted(list(set(s[0] for s in dataset.samples)))
    if not unique_seqs:
        print("No sequences found.")
        return

    print("unique_seqs:", unique_seqs)
    for seq_name in unique_seqs:
        if sequences_processed >= num_sequences_to_sample:
            break
            
        print(f"\n[{sequences_processed+1}/{num_sequences_to_sample}] Processing sequence: {seq_name}")
        
        # Filter indices for this sequence
        seq_indices = [i for i, s in enumerate(dataset.samples) if s[0] == seq_name]
        # Sort by frame name (assuming numeric frame names)
        seq_indices.sort(key=lambda i: int(dataset.samples[i][1]) if dataset.samples[i][1].isdigit() else dataset.samples[i][1])
        
        if not seq_indices:
            print(f"No frames found for {seq_name}, skipping.")
            continue

        # Inspect first frame to see available views
        if True:
            first_frame_idx = seq_indices[0]
            seq_n, frame_n = dataset.samples[first_frame_idx]
            frame_dir = os.path.join(dataset.dataset_location, seq_n, frame_n, 'rgbs')
            import glob
            img_files = glob.glob(os.path.join(frame_dir, "*.png"))
            available_vids = sorted([int(os.path.splitext(os.path.basename(f))[0]) for f in img_files])
            print(f"Sequence {seq_name} available views: {available_vids}")
            
            if args.view_id not in available_vids:
                print(f"ERROR: View ID {args.view_id} is NOT present in this sequence. Skipping.")
                continue

        print(f"Found {len(seq_indices)} frames for sequence {seq_name}")
        
        output_video_path = os.path.join(args.output_dir, f"seq_{seq_name.replace('/', '_')}_view_{args.view_id}.mp4")
        
        # Video Writer
        # Try mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_vid = cv2.VideoWriter(output_video_path, fourcc, 30.0, (args.width, args.height))
        
        count = 0
        for idx in seq_indices:
            # Get views for this frame
            try:
                views = dataset[idx]
            except Exception as e:
                print(f"Error loading frame index {idx}: {e}")
                continue

            # Find the specific view
            frame_img_np = None
            found_view_in_frame = False
            
            # Check if we got the correct frame (handle fallback case)
            expected_label = f"{seq_name}/{dataset.samples[idx][1]}"
            
            for view_dict in views:
                # Verify label matches expected frame
                # view_dict['label'] is "seq/frame"
                if view_dict['label'] != expected_label:
                    continue # This view belongs to a different frame (fallback triggered)
                
                if int(view_dict['instance']) == args.view_id:
                     frame_img_np = view_dict['img'] # Already numpy array due to transform
                     found_view_in_frame = True
                     break
            
            if found_view_in_frame and frame_img_np is not None:
                 # Check/Print resolution once
                 if count == 0:
                     print(f"Frame shape (H, W, C): {frame_img_np.shape}")
                     if frame_img_np.shape[1] != args.width or frame_img_np.shape[0] != args.height:
                         print(f"WARNING: Image shape {frame_img_np.shape} does not match requested ({args.height}, {args.width})")

                 # Convert RGB (PIL) to BGR (OpenCV)
                 frame_bgr = cv2.cvtColor(frame_img_np, cv2.COLOR_RGB2BGR)
                 out_vid.write(frame_bgr)
                 count += 1
                 if count % 10 == 0:
                     print(f"Processed {count}/{len(seq_indices)} frames...", end='\r')
            else:
                 # Logic for missing frame? 
                 # If fallback happened, we don't have this frame.
                 # Either repeat previous frame or skip (video will speed up).
                 # For diagnosis, printing warning.
                 # print(f"Warning: Frame {expected_label} view {args.view_id} missing (or fallback triggered).")
                 pass

        out_vid.release()
        
        if count > 0:
            print(f"\nVideo saved to {output_video_path} ({count} frames)")
            sequences_processed += 1
        else:
            print(f"\nNo frames written for {seq_name} (View {args.view_id} might be missing). Deleting empty file.")
            if os.path.exists(output_video_path):
                os.remove(output_video_path)

if __name__ == "__main__":
    main()
