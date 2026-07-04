"""
Generate RTFM list files and gt-sh.npy for ShanghaiTech using local paths.

Reads the frame-level labels and SHANGHAI_train.txt / SHANGHAI_test.txt
to produce:
  - shanghai-i3d-train-10crop.list  (63 abnormal + 175 normal)
  - shanghai-i3d-test-10crop.list   (44 abnormal + 155 normal)
  - gt-sh.npy                       (frame-level ground truth for test set)
"""

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RTFM_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(RTFM_DIR, '..', 'data', 'SHANGHAI')

TRAIN_FEAT_DIR = os.path.join(RTFM_DIR, 'data', 'SH_Train_ten_crop_i3d')
TEST_FEAT_DIR = os.path.join(RTFM_DIR, 'data', 'SH_Test_ten_crop_i3d')
LABEL_DIR = os.path.join(DATA_DIR, 'SHANGHAI_Test', 'label')


def parse_train_txt():
    txt_path = os.path.join(DATA_DIR, 'SHANGHAI_TRAIN', 'SHANGHAI_train.txt')
    abnormal, normal = [], []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            vid_id = parts[0].split('/')[-1]
            is_anomaly = int(parts[-1])
            feat_path = os.path.join(TRAIN_FEAT_DIR, f'{vid_id}_i3d.npy')
            if is_anomaly == 1:
                abnormal.append(feat_path)
            else:
                normal.append(feat_path)
    return abnormal, normal


def parse_test_labels():
    """
    Read frame-level .npy labels for each test video.
    Determine which are abnormal (any label > 0) vs normal.
    Returns abnormal list, normal list, and ordered gt frames.
    """
    label_files = sorted(os.listdir(LABEL_DIR))
    abnormal, normal = [], []
    vid_labels = {}

    for lf in label_files:
        vid_id = lf.replace('.npy', '')
        labels = np.load(os.path.join(LABEL_DIR, lf))
        vid_labels[vid_id] = labels
        feat_path = os.path.join(TEST_FEAT_DIR, f'{vid_id}_i3d.npy')
        if labels.max() > 0:
            abnormal.append((feat_path, vid_id))
        else:
            normal.append((feat_path, vid_id))

    return abnormal, normal, vid_labels


def generate_gt(ordered_vid_ids, vid_labels):
    """
    Generate frame-level GT array matching RTFM's expected format.
    RTFM expands snippet-level predictions to frame-level by np.repeat(..., 16).
    So GT must have num_snippets * 16 frames per video.
    """
    gt = []
    for vid_id in ordered_vid_ids:
        labels = vid_labels[vid_id]
        num_frames = len(labels)
        num_snippets = num_frames // 16
        if num_snippets == 0:
            num_snippets = 1
        expanded_len = num_snippets * 16

        if expanded_len <= num_frames:
            gt_segment = labels[:expanded_len]
        else:
            gt_segment = np.concatenate([labels, np.zeros(expanded_len - num_frames)])

        gt.extend(gt_segment.tolist())

    return np.array(gt, dtype=np.int32)


def main():
    print("Generating train list...")
    train_abn, train_nor = parse_train_txt()
    train_list = train_abn + train_nor
    train_path = os.path.join(BASE_DIR, 'shanghai-i3d-train-10crop.list')
    with open(train_path, 'w') as f:
        for p in train_list:
            f.write(p + '\n')
    print(f"  Train: {len(train_abn)} abnormal + {len(train_nor)} normal = {len(train_list)}")

    print("Generating test list and GT...")
    test_abn, test_nor, vid_labels = parse_test_labels()
    test_list = test_abn + test_nor
    test_path = os.path.join(BASE_DIR, 'shanghai-i3d-test-10crop.list')
    with open(test_path, 'w') as f:
        for p, _ in test_list:
            f.write(p + '\n')
    print(f"  Test: {len(test_abn)} abnormal + {len(test_nor)} normal = {len(test_list)}")

    ordered_ids = [vid_id for _, vid_id in test_list]
    gt = generate_gt(ordered_ids, vid_labels)
    gt_path = os.path.join(BASE_DIR, 'gt-sh.npy')
    np.save(gt_path, gt)
    print(f"  GT saved: {gt.shape[0]} frames, {gt.sum()} anomalous")

    print("\nDone!")


if __name__ == '__main__':
    main()
