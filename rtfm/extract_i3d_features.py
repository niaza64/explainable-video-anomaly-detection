"""
Extract I3D ResNet50 features (10-crop) from ShanghaiTech frame directories.

Produces .npy files with shape [10, T, 2048] where T = num_snippets (each snippet = 16 frames).
This matches the format expected by RTFM.

Usage:
    python extract_i3d_features.py --split test
    python extract_i3d_features.py --split train
    python extract_i3d_features.py --split both
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch-resnet3d'))
from models.resnet import I3Res50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'SHANGHAI')
PRETRAINED_PATH = os.path.join(BASE_DIR, 'pytorch-resnet3d', 'pretrained', 'i3d_r50_kinetics.pth')

CLIP_LEN = 16
CROP_SIZE = 224
RESIZE_SHORT = 256

MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]


def load_frames(frame_dir):
    frame_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    frames = []
    for ff in frame_files:
        img = Image.open(os.path.join(frame_dir, ff)).convert('RGB')
        frames.append(img)
    return frames


def ten_crop(frames_tensor, crop_size):
    """
    Apply 10-crop augmentation: 4 corners + center, each with horizontal flip.
    Input: (T, C, H, W)
    Output: list of 10 tensors, each (T, C, crop_size, crop_size)
    """
    _, _, h, w = frames_tensor.shape
    crops = []

    positions = [
        (0, 0),                             # top-left
        (0, w - crop_size),                  # top-right
        (h - crop_size, 0),                  # bottom-left
        (h - crop_size, w - crop_size),      # bottom-right
        ((h - crop_size) // 2, (w - crop_size) // 2),  # center
    ]

    for (top, left) in positions:
        crop = frames_tensor[:, :, top:top + crop_size, left:left + crop_size]
        crops.append(crop)
        crops.append(torch.flip(crop, dims=[3]))  # horizontal flip

    return crops


def preprocess_frames(frames):
    """Resize shortest side to RESIZE_SHORT, convert to tensor, normalize."""
    transform = transforms.Compose([
        transforms.Resize(RESIZE_SHORT),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return torch.stack([transform(f) for f in frames])


def build_model(device):
    model = I3Res50(num_classes=400, use_nl=False)
    state_dict = torch.load(PRETRAINED_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def extract_features_single_clip(model, clip_tensor, device):
    """
    Extract 2048-dim feature from a single clip.
    clip_tensor: (C, T, H, W) — already on device
    Returns: (2048,) numpy array
    """
    x = clip_tensor.unsqueeze(0)  # (1, C, T, H, W)

    with torch.no_grad():
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool1(x)
        x = model.layer1(x)
        x = model.maxpool2(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = x.view(x.shape[0], -1)  # (1, 2048)

    return x.squeeze(0).cpu().numpy()


def extract_video_features(model, frame_dir, device):
    """
    Extract 10-crop I3D features for a single video.
    Returns: numpy array of shape (10, T, 2048)
    """
    frames = load_frames(frame_dir)
    num_frames = len(frames)

    if num_frames == 0:
        print(f"  WARNING: No frames in {frame_dir}")
        return None

    frames_tensor = preprocess_frames(frames)  # (N, C, H, W)

    num_snippets = num_frames // CLIP_LEN
    if num_snippets == 0:
        num_snippets = 1

    all_crop_features = []

    crops = ten_crop(frames_tensor, CROP_SIZE)  # list of 10 tensors

    for crop_idx, crop_frames in enumerate(crops):
        snippet_features = []

        for snip_idx in range(num_snippets):
            start = snip_idx * CLIP_LEN
            end = start + CLIP_LEN

            if end > len(crop_frames):
                clip = crop_frames[-CLIP_LEN:]
            else:
                clip = crop_frames[start:end]

            if len(clip) < CLIP_LEN:
                pad_count = CLIP_LEN - len(clip)
                clip = torch.cat([clip, clip[-1:].repeat(pad_count, 1, 1, 1)], dim=0)

            # (T, C, H, W) -> (C, T, H, W) for I3D
            clip = clip.permute(1, 0, 2, 3).to(device)
            feat = extract_features_single_clip(model, clip, device)
            snippet_features.append(feat)

        all_crop_features.append(np.stack(snippet_features, axis=0))  # (T, 2048)

    features = np.stack(all_crop_features, axis=0)  # (10, T, 2048)
    return features


def get_video_dirs(split):
    if split == 'test':
        base = os.path.join(DATA_DIR, 'SHANGHAI_Test', 'frames')
    elif split == 'train':
        base = os.path.join(DATA_DIR, 'SHANGHAI_TRAIN', 'frames')
    else:
        raise ValueError(f"Unknown split: {split}")

    dirs = sorted([
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ])
    return base, dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='both', choices=['train', 'test', 'both'])
    parser.add_argument('--output-dir', type=str, default=os.path.join(BASE_DIR, 'data'))
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print("Loading I3D ResNet50 model...")
    model = build_model(device)
    print("Model loaded.\n")

    splits = ['train', 'test'] if args.split == 'both' else [args.split]

    for split in splits:
        if split == 'test':
            out_dir = os.path.join(args.output_dir, 'SH_Test_ten_crop_i3d')
        else:
            out_dir = os.path.join(args.output_dir, 'SH_Train_ten_crop_i3d')

        os.makedirs(out_dir, exist_ok=True)

        base, video_dirs = get_video_dirs(split)
        total = len(video_dirs)
        print(f"=== Extracting {split} features: {total} videos → {out_dir} ===\n")

        for idx, vid_name in enumerate(video_dirs):
            out_path = os.path.join(out_dir, f'{vid_name}_i3d.npy')

            if os.path.exists(out_path):
                print(f"[{idx+1}/{total}] {vid_name} — already exists, skipping")
                continue

            frame_dir = os.path.join(base, vid_name)
            num_frames = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg') or f.endswith('.png')])
            print(f"[{idx+1}/{total}] {vid_name} ({num_frames} frames)...", end=' ', flush=True)

            features = extract_video_features(model, frame_dir, device)
            if features is not None:
                np.save(out_path, features)
                print(f"saved {features.shape}")
            else:
                print("FAILED")

        print(f"\n=== Done with {split} ===\n")


if __name__ == '__main__':
    main()
