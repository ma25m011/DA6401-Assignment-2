"""Training entrypoint
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_WANDB_SETTINGS = wandb.Settings(_disable_service=True)


def get_loaders(root, batch_size=32, num_workers=0):
    train_full = OxfordIIITPetDataset(root, split='trainval')
    test_ds = OxfordIIITPetDataset(root, split='test')
    n_val = int(len(train_full) * 0.1)
    n_train = len(train_full) - n_val
    g = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=g)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


# ── classifier ────────────────────────────────────────────────────────────────

def _train_epoch_clf(model, loader, opt, crit):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, labels, _, _, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = crit(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * len(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(imgs)
    return loss_sum / n, correct / n


@torch.no_grad()
def _val_clf(model, loader, crit):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for imgs, labels, _, _, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        loss = crit(logits, labels)
        loss_sum += loss.item() * len(imgs)
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(imgs)
    return loss_sum / n, correct / n


def _log_feature_maps(model, loader):
    fmaps = {}
    handles = []

    def hook(name):
        def h(m, inp, out):
            fmaps[name] = out.detach().cpu()
        return h

    handles.append(model.encoder.block1.register_forward_hook(hook('first')))
    handles.append(model.encoder.block5.register_forward_hook(hook('last')))

    model.eval()
    imgs, *_ = next(iter(loader))
    with torch.no_grad():
        model(imgs[:1].to(DEVICE))
    for h in handles:
        h.remove()

    log = {}
    for tag, fm in fmaps.items():
        fm = fm[0]
        panels = []
        for c in range(min(8, fm.shape[0])):
            ch = fm[c].numpy()
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
            panels.append(wandb.Image(ch, caption=f'ch{c}'))
        log[f'feature_maps/{tag}_conv'] = panels
    wandb.log(log)


def train_classifier(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, _ = get_loaders(args.data, args.batch_size, args.num_workers)

    # ablation_bn → sweep use_bn; ablation_dropout → sweep dp; else single run
    bn_vals = [True, False] if args.task == 'ablation_bn' else [args.use_bn]
    dp_vals = [0.0, 0.2, 0.5] if args.task == 'ablation_dropout' else [args.dropout_p]

    for use_bn in bn_vals:
        for dp in dp_vals:
            is_main = (use_bn is True) and (dp == args.dropout_p)
            run_name = f'cls_bn{use_bn}_dp{dp}'

            run = wandb.init(project=args.wandb_project, name=run_name, config={
                'task': 'classification', 'dropout_p': dp, 'use_bn': use_bn,
                'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size,
            }, reinit=True, settings=_WANDB_SETTINGS)

            model = VGG11Classifier(dropout_p=dp, use_bn=use_bn).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            crit = nn.CrossEntropyLoss()

            # hook for 3rd conv layer activations (block3 = first conv block with 2 convs)
            act_buf = []

            def act_hook(m, inp, out):
                act_buf.append(out.detach().cpu().flatten().numpy()[:2000].copy())

            handle = model.encoder.block3[0].register_forward_hook(act_hook)

            best_acc = 0.0
            for epoch in range(args.epochs):
                tr_loss, tr_acc = _train_epoch_clf(model, train_loader, opt, crit)
                val_loss, val_acc = _val_clf(model, val_loader, crit)
                sched.step()

                log = {
                    'train_loss': tr_loss, 'val_loss': val_loss,
                    'train_acc': tr_acc, 'val_acc': val_acc, 'epoch': epoch + 1,
                }
                if act_buf:
                    log['activations/block3'] = wandb.Histogram(np.concatenate(act_buf))
                    act_buf.clear()

                wandb.log(log)
                print(f'[{run_name}] {epoch+1}/{args.epochs}  '
                      f'tl={tr_loss:.4f} vl={val_loss:.4f} va={val_acc:.3f}')

                if is_main and val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(),
                               os.path.join(args.save_dir, 'classifier.pth'))

            handle.remove()

            if is_main:
                _log_feature_maps(model, val_loader)

            run.finish()


# ── localizer ─────────────────────────────────────────────────────────────────

def _train_epoch_loc(model, loader, opt, mse, iou):
    model.train()
    loss_sum, n = 0.0, 0
    for imgs, _, bboxes, _, has_bbox in loader:
        imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
        mask = has_bbox.bool()
        preds = model(imgs)
        if mask.sum() == 0:
            continue
        loss = mse(preds[mask], bboxes[mask]) + iou(preds[mask], bboxes[mask])
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * mask.sum().item()
        n += mask.sum().item()
    return loss_sum / max(n, 1)


@torch.no_grad()
def _val_loc(model, loader, mse, iou):
    model.eval()
    loss_sum, n = 0.0, 0
    for imgs, _, bboxes, _, has_bbox in loader:
        imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
        mask = has_bbox.bool()
        if mask.sum() == 0:
            continue
        preds = model(imgs)
        loss = mse(preds[mask], bboxes[mask]) + iou(preds[mask], bboxes[mask])
        loss_sum += loss.item() * mask.sum().item()
        n += mask.sum().item()
    return loss_sum / max(n, 1)


def train_localizer(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, _ = get_loaders(args.data, args.batch_size, args.num_workers)

    wandb.init(project=args.wandb_project, name=f'loc_{args.freeze}', config={
        'task': 'localization', 'freeze': args.freeze,
        'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size,
    }, settings=_WANDB_SETTINGS)

    model = VGG11Localizer().to(DEVICE)

    clf_ckpt = os.path.join(args.save_dir, 'classifier.pth')
    if os.path.exists(clf_ckpt):
        sd = torch.load(clf_ckpt, map_location='cpu')
        enc_sd = {k[8:]: v for k, v in sd.items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(enc_sd)
        print('Loaded encoder from classifier checkpoint')

    if args.freeze == 'frozen':
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif args.freeze == 'partial':
        for name, p in model.encoder.named_parameters():
            if any(f'block{i}' in name for i in [1, 2, 3]):
                p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    mse, iou = nn.MSELoss(), IoULoss()

    best_loss = float('inf')
    for epoch in range(args.epochs):
        tr_loss = _train_epoch_loc(model, train_loader, opt, mse, iou)
        val_loss = _val_loc(model, val_loader, mse, iou)
        sched.step()
        wandb.log({'train_loss': tr_loss, 'val_loss': val_loss, 'epoch': epoch + 1})
        print(f'[localizer] {epoch+1}/{args.epochs}  tl={tr_loss:.4f} vl={val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'localizer.pth'))

    wandb.finish()


# ── segmentation ──────────────────────────────────────────────────────────────

def _dice(preds, targets, num_classes=3):
    pred_cls = preds.argmax(1)
    score = 0.0
    for c in range(num_classes):
        p = (pred_cls == c).float()
        t = (targets == c).float()
        score += (2 * (p * t).sum()) / (p.sum() + t.sum() + 1e-8)
    return (score / num_classes).item()


def _train_epoch_seg(model, loader, opt, crit):
    model.train()
    loss_sum, n = 0.0, 0
    for imgs, _, _, trimaps, _ in loader:
        imgs, trimaps = imgs.to(DEVICE), trimaps.to(DEVICE)
        loss = crit(model(imgs), trimaps)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * len(imgs)
        n += len(imgs)
    return loss_sum / n


@torch.no_grad()
def _val_seg(model, loader, crit):
    model.eval()
    loss_sum, dice_sum, n = 0.0, 0.0, 0
    for imgs, _, _, trimaps, _ in loader:
        imgs, trimaps = imgs.to(DEVICE), trimaps.to(DEVICE)
        out = model(imgs)
        loss_sum += crit(out, trimaps).item() * len(imgs)
        dice_sum += _dice(out, trimaps) * len(imgs)
        n += len(imgs)
    return loss_sum / n, dice_sum / n


def train_segmentation(args):
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, _ = get_loaders(args.data, args.batch_size, args.num_workers)

    # ablation_seg runs all three freeze modes; segmentation runs just args.freeze
    freeze_modes = ['frozen', 'partial', 'full'] if args.task == 'ablation_seg' else [args.freeze]

    for freeze in freeze_modes:
        is_main = (freeze == args.freeze)
        run_name = f'seg_{freeze}'

        run = wandb.init(project=args.wandb_project, name=run_name, config={
            'task': 'segmentation', 'freeze': freeze,
            'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size,
        }, reinit=True, settings=_WANDB_SETTINGS)

        model = VGG11UNet().to(DEVICE)

        clf_ckpt = os.path.join(args.save_dir, 'classifier.pth')
        if os.path.exists(clf_ckpt):
            sd = torch.load(clf_ckpt, map_location='cpu')
            enc_sd = {k[8:]: v for k, v in sd.items() if k.startswith('encoder.')}
            model.encoder.load_state_dict(enc_sd)
            print(f'[{run_name}] Loaded encoder from classifier checkpoint')

        if freeze == 'frozen':
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif freeze == 'partial':
            for name, p in model.encoder.named_parameters():
                if any(f'block{i}' in name for i in [1, 2, 3]):
                    p.requires_grad = False

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        crit = nn.CrossEntropyLoss()

        best_dice = 0.0
        for epoch in range(args.epochs):
            tr_loss = _train_epoch_seg(model, train_loader, opt, crit)
            val_loss, val_dice = _val_seg(model, val_loader, crit)
            sched.step()
            wandb.log({'train_loss': tr_loss, 'val_loss': val_loss,
                       'val_dice': val_dice, 'epoch': epoch + 1})
            print(f'[{run_name}] {epoch+1}/{args.epochs}  '
                  f'tl={tr_loss:.4f} vl={val_loss:.4f} dice={val_dice:.4f}')
            if is_main and val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'unet.pth'))

        run.finish()


# ── entry ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='classifier',
                   choices=['classifier', 'localizer', 'segmentation',
                            'ablation_dropout', 'ablation_bn', 'ablation_seg'])
    p.add_argument('--data', default='../oxford-iiit-pet')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout_p', type=float, default=0.5)
    p.add_argument('--use_bn', type=lambda x: x.lower() != 'false', default=True)
    p.add_argument('--freeze', default='full', choices=['frozen', 'partial', 'full'])
    p.add_argument('--save_dir', default='checkpoints')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--wandb_project', default='da6401-assignment2')
    args = p.parse_args()

    if args.task in ('classifier', 'ablation_dropout', 'ablation_bn'):
        train_classifier(args)
    elif args.task == 'localizer':
        train_localizer(args)
    elif args.task in ('segmentation', 'ablation_seg'):
        train_segmentation(args)


if __name__ == '__main__':
    main()
