"""
Train RTFM on ShanghaiTech — standalone script without visdom dependency.

Usage:
    cd rtfm/
    python train_rtfm.py
    python train_rtfm.py --max-epoch 5000 --batch-size 16
"""

import os
import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from model import Model
from dataset import Dataset
from train import RTFM_loss, sparsity, smooth
from utils import process_feat
import option


class DummyViz:
    """No-op visualizer to replace visdom."""
    def plot_lines(self, *a, **kw): pass
    def lines(self, *a, **kw): pass
    def disp_image(self, *a, **kw): pass
    def scatter(self, *a, **kw): pass


def test_auc(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)

        for i, input in enumerate(dataloader):
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            _, _, _, _, _, _, logits, _, _, _ = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))

        gt = np.load('list/gt-sh.npy')
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, _ = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        return rec_auc


def train_step(nloader, aloader, model, batch_size, optimizer, device):
    model.train()

    ninput, nlabel = next(nloader)
    ainput, alabel = next(aloader)

    input = torch.cat((ninput, ainput), 0).to(device)

    score_abnormal, score_normal, feat_select_abn, feat_select_normal, \
        feat_abn_bottom, feat_normal_bottom, scores, scores_nor_bottom, \
        scores_nor_abn_bag, _ = model(input)

    scores = scores.view(batch_size * 32 * 2, -1).squeeze()
    abn_scores = scores[batch_size * 32:]

    nlabel = nlabel[0:batch_size]
    alabel = alabel[0:batch_size]

    loss_criterion = RTFM_loss(0.0001, 100)
    loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
    loss_smooth = smooth(abn_scores, 8e-4)
    cost = loss_criterion(score_normal, score_abnormal, nlabel, alabel,
                          feat_select_normal, feat_select_abn) + loss_smooth + loss_sparse

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    return cost.item()


def main():
    args = option.parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    train_nloader = DataLoader(
        Dataset(args, test_mode=False, is_normal=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)

    train_aloader = DataLoader(
        Dataset(args, test_mode=False, is_normal=False),
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)

    test_loader = DataLoader(
        Dataset(args, test_mode=True),
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)
    model = model.to(device)

    os.makedirs('./ckpt', exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)

    best_auc = -1
    print(f"\nTraining RTFM for {args.max_epoch} steps...\n")

    for step in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True):
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)
        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        loss = train_step(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device)

        if step % 5 == 0 and step > 200:
            current_auc = test_auc(test_loader, model, args, device)
            tqdm.write(f"  Step {step}: AUC = {current_auc:.4f} (best = {best_auc:.4f})")

            if current_auc > best_auc:
                best_auc = current_auc
                torch.save(model.state_dict(), './ckpt/rtfm_best.pkl')
                tqdm.write(f"  *** New best AUC: {best_auc:.4f} — saved ckpt/rtfm_best.pkl ***")

    torch.save(model.state_dict(), './ckpt/rtfm_final.pkl')
    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Checkpoints: ckpt/rtfm_best.pkl, ckpt/rtfm_final.pkl")


if __name__ == '__main__':
    main()
