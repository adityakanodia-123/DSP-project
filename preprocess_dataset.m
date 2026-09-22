clc; clear; close all;

input_base = 'raw_dataset';
output_base = 'dataset';
classes = {'normal', 'retinal_detachment'};
split_ratio = 0.80; % 80% for training, 20% for validation

% 1. Create target directory folders for train and val
for c = 1:length(classes)
    train_dir = fullfile(output_base, 'train', classes{c});
    val_dir = fullfile(output_base, 'val', classes{c});
    if ~exist(train_dir, 'dir')
        mkdir(train_dir);
    end
    if ~exist(val_dir, 'dir')
        mkdir(val_dir);
    end
end

% 2. Pre-generate Gaussian Kernels for DSP Filtering
% Kernel 1: 5x5 Gaussian Kernel (sigma = 1.5) for Speckle Noise Removal
[x1, y1] = meshgrid(-2:2, -2:2);
h1 = exp(-(x1.^2 + y1.^2) / (2 * 1.5^2));
h1 = h1 / sum(h1(:));

% Kernel 2: 9x9 Gaussian Kernel (sigma = 2.0) for Unsharp Masking (Radius = 2)
[x2, y2] = meshgrid(-4:4, -4:4);
h2 = exp(-(x2.^2 + y2.^2) / (2 * 2.0^2));
h2 = h2 / sum(h2(:));

% 3. Process each class
for c = 1:length(classes)
    cls = classes{c};
    files = dir(fullfile(input_base, cls, '*.png'));
    if isempty(files)
        files = dir(fullfile(input_base, cls, '*.jpg'));
    end
    
    total = length(files);
    fprintf('Running DSP Preprocessing on %d scans for [%s]...\n', total, cls);
    
    for i = 1:total
        img_path = fullfile(files(i).folder, files(i).name);
        I = imread(img_path);
        
        % Convert to Grayscale & Double Precision [0.0, 1.0]
        if size(I, 3) == 3
            I_double = double(I);
            I_gray = 0.2989 * I_double(:,:,1) + 0.5870 * I_double(:,:,2) + 0.1140 * I_double(:,:,3);
            I_norm = I_gray / 255.0;
        else
            I_norm = double(I) / 255.0;
        end
        
        % --- DSP STEP 1: Low-Pass Gaussian Filtering (Speckle Noise Removal) ---
        top1 = repmat(I_norm(1, :), 2, 1);
        bot1 = repmat(I_norm(end, :), 2, 1);
        I_pad1 = [top1; I_norm; bot1];
        left1 = repmat(I_pad1(:, 1), 1, 2);
        right1 = repmat(I_pad1(:, end), 1, 2);
        I_pad1 = [left1, I_pad1, right1];
        
        I_denoised = conv2(I_pad1, h1, 'valid');
        
        % --- DSP STEP 2: Unsharp Masking (Accentuating Anatomical Boundaries) ---
        top2 = repmat(I_denoised(1, :), 4, 1);
        bot2 = repmat(I_denoised(end, :), 4, 1);
        I_pad2 = [top2; I_denoised; bot2];
        left2 = repmat(I_pad2(:, 1), 1, 4);
        right2 = repmat(I_pad2(:, end), 1, 4);
        I_pad2 = [left2, I_pad2, right2];
        
        I_blur = conv2(I_pad2, h2, 'valid');
        
        I_enhanced = I_denoised + 1.2 * (I_denoised - I_blur);
        I_enhanced = min(max(I_enhanced, 0.0), 1.0); % Clamp pixel values to [0, 1]
        
        % Convert to 8-bit unsigned int
        I_out = uint8(round(I_enhanced * 255.0));
        
        % Train / Val Split (80% Train, 20% Val)
        if i <= floor(split_ratio * total)
            dest = fullfile(output_base, 'train', cls);
        else
            dest = fullfile(output_base, 'val', cls);
        end
        
        imwrite(I_out, fullfile(dest, sprintf('dsp_%s_%04d.png', cls, i)));
    end
end

fprintf('\n==================================================\n');
fprintf('DSP PREPROCESSING & DATASET SPLIT COMPLETE\n');
fprintf('Cleaned images saved to: /dataset/train and /dataset/val\n');
fprintf('==================================================\n');
