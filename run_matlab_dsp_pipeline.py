import os
import glob
import math
import numpy as np
import cv2

input_base = 'raw_dataset'
output_base = 'dataset'
classes = ['normal', 'retinal_detachment']
split_ratio = 0.80  # 80% for training, 20% for validation

# Create target directory folders
for c in classes:
    os.makedirs(os.path.join(output_base, 'train', c), exist_ok=True)
    os.makedirs(os.path.join(output_base, 'val', c), exist_ok=True)

for c in classes:
    files = sorted(glob.glob(os.path.join(input_base, c, '*.png')))
    if not files:
        files = sorted(glob.glob(os.path.join(input_base, c, '*.jpg')))
    
    total = len(files)
    print(f"Running DSP Preprocessing on {total} scans for [{c}]...")
    
    train_count = math.floor(split_ratio * total)
    
    for i, file_path in enumerate(files, 1):
        # Read image
        I = cv2.imread(file_path)
        if I is None:
            continue
            
        # Convert to Grayscale & Double Precision [0.0, 1.0]
        if len(I.shape) == 3 and I.shape[2] == 3:
            I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        else:
            I_gray = I
            
        I_norm = I_gray.astype(np.float64) / 255.0
        
        # 1. Low-Pass Gaussian Filtering (Speckle Noise Removal)
        # MATLAB: fspecial('gaussian', [5 5], 1.5) with imfilter replicate
        I_denoised = cv2.GaussianBlur(I_norm, (5, 5), sigmaX=1.5, sigmaY=1.5, borderType=cv2.BORDER_REPLICATE)
        
        # 2. Unsharp Masking (Accentuating Anatomical Membrane Boundaries)
        # MATLAB: imsharpen(I_denoised, 'Radius', 2, 'Amount', 1.2)
        # Gaussian blur with radius 2 (sigma=2)
        blur_gaussian = cv2.GaussianBlur(I_denoised, (0, 0), sigmaX=2.0, sigmaY=2.0, borderType=cv2.BORDER_REPLICATE)
        I_enhanced = I_denoised + 1.2 * (I_denoised - blur_gaussian)
        I_enhanced = np.clip(I_enhanced, 0.0, 1.0)
        
        # Convert back to 8-bit unsigned int
        I_out = (I_enhanced * 255.0 + 0.5).astype(np.uint8)
        
        # Train / Val Split
        if i <= train_count:
            dest = os.path.join(output_base, 'train', c)
        else:
            dest = os.path.join(output_base, 'val', c)
            
        out_filename = f"dsp_{c}_{i:04d}.png"
        cv2.imwrite(os.path.join(dest, out_filename), I_out)
        
print("\nAll images cleaned and organized into /dataset/train and /dataset/val.")
