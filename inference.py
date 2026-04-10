"""Inference and evaluation
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import wandb
from sklearn.metrics import f1_score

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from multitask import MultiTaskPerceptionModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_BILINEAR = getattr(getattr(Image, 'Resampling', Image), 'BILINEAR')

SEG_PALETTE = [(60, 180, 60), (60, 60, 180), (200, 60, 60)]  # fg, bg, boundary
BREED_NAMES = [  # 37 Oxford-IIT Pet breeds, 0-indexed
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
    'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
    'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
    'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
    'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
    'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def get_test_loader(root, batch_size=32, num_workers=0):
    ds = OxfordIIITPetDataset(root, split='test')
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def denorm(t):
    """Normalized tensor [3,H,W] → uint8 ndarray [H,W,3]."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (t.cpu() * std + mean).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def draw_box(arr, box, color, lw=2):
    """Draw cx,cy,w,h box on uint8 ndarray. Returns PIL Image."""
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    cx, cy, w, h = box
    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    for i in range(lw):
        d.rectangle([x1+i, y1+i, x2-i, y2-i], outline=color)
    return img


def trimap_rgb(arr):
    """Class index ndarray [H,W] → RGB ndarray [H,W,3]."""
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for c, col in enumerate(SEG_PALETTE):
        out[arr == c] = col
    return out


def box_iou(pred, gt):
    """Compute IoU for cxcywh boxes. Both shape [4]."""
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt
    px1, py1, px2, py2 = px - pw/2, py - ph/2, px + pw/2, py + ph/2
    gx1, gy1, gx2, gy2 = gx - gw/2, gy - gh/2, gx + gw/2, gy + gh/2
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = pw*ph + gw*gh - inter + 1e-6
    return inter / union


# ── task evaluations ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_classifier(args):
    model = VGG11Classifier().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(args.save_dir, 'classifier.pth'), map_location='cpu'))
    model.eval()

    loader = get_test_loader(args.data, args.batch_size, args.num_workers)
    all_preds, all_labels = [], []
    for imgs, labels, _, _, _ in loader:
        logits = model(imgs.to(DEVICE))
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.tolist())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Classification  Macro F1: {macro_f1:.4f}')
    wandb.log({'test/macro_f1': macro_f1})
    return macro_f1


@torch.no_grad()
def eval_localizer(args):
    model = VGG11Localizer().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(args.save_dir, 'localizer.pth'), map_location='cpu'))
    model.eval()

    ds = OxfordIIITPetDataset(args.data, split='trainval')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    ious, imgs_logged = [], 0
    table = wandb.Table(columns=['image', 'iou', 'has_bbox'])

    for imgs, _, bboxes, _, has_bbox in loader:
        preds = model(imgs.to(DEVICE)).cpu()
        for i in range(len(imgs)):
            if not has_bbox[i]:
                continue
            iou = box_iou(preds[i].tolist(), bboxes[i].tolist())
            ious.append(iou)

            if imgs_logged < 15:
                arr = denorm(imgs[i])
                pil = draw_box(arr, bboxes[i].tolist(), (0, 220, 0))   # GT green
                pil = draw_box(np.array(pil), preds[i].tolist(), (220, 0, 0))  # pred red
                table.add_data(wandb.Image(pil), round(iou, 3), True)
                imgs_logged += 1

    mean_iou = float(np.mean(ious))
    ap50 = float(np.mean(np.array(ious) >= 0.5))
    ap75 = float(np.mean(np.array(ious) >= 0.75))
    print(f'Localization  mIoU={mean_iou:.4f}  AP@0.5={ap50:.4f}  AP@0.75={ap75:.4f}')
    wandb.log({'test/mean_iou': mean_iou, 'test/ap50': ap50,
               'test/ap75': ap75, 'detection_table': table})
    return mean_iou


@torch.no_grad()
def eval_segmentation(args):
    model = VGG11UNet().to(DEVICE)
    model.load_state_dict(torch.load(
        os.path.join(args.save_dir, 'unet.pth'), map_location='cpu'))
    model.eval()

    loader = get_test_loader(args.data, args.batch_size, args.num_workers)
    dice_sum, px_sum, n = 0.0, 0.0, 0
    samples_logged = 0

    for imgs, _, _, trimaps, _ in loader:
        imgs, trimaps = imgs.to(DEVICE), trimaps.to(DEVICE)
        out = model(imgs)
        pred_cls = out.argmax(1)

        for i in range(len(imgs)):
            # dice per image
            d = 0.0
            for c in range(3):
                p = (pred_cls[i] == c).float()
                t = (trimaps[i] == c).float()
                d += (2 * (p * t).sum() / (p.sum() + t.sum() + 1e-8)).item()
            dice_sum += d / 3

            # pixel accuracy per image
            px_sum += (pred_cls[i] == trimaps[i]).float().mean().item()
            n += 1

            if samples_logged < 5:
                orig = Image.fromarray(denorm(imgs[i].cpu()))
                gt_rgb = Image.fromarray(trimap_rgb(trimaps[i].cpu().numpy()))
                pred_rgb = Image.fromarray(trimap_rgb(pred_cls[i].cpu().numpy()))
                wandb.log({
                    f'seg_samples/{samples_logged}_orig': wandb.Image(orig),
                    f'seg_samples/{samples_logged}_gt': wandb.Image(gt_rgb),
                    f'seg_samples/{samples_logged}_pred': wandb.Image(pred_rgb),
                })
                samples_logged += 1

    mean_dice = dice_sum / n
    mean_px = px_sum / n
    print(f'Segmentation  Dice={mean_dice:.4f}  PixelAcc={mean_px:.4f}')
    wandb.log({'test/dice': mean_dice, 'test/pixel_acc': mean_px})
    return mean_dice, mean_px


# ── pipeline showcase ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_pipeline(args):
    if not args.images:
        print('Pass --images path1 path2 path3 for pipeline showcase')
        return

    model = MultiTaskPerceptionModel().to(DEVICE)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for idx, path in enumerate(args.images):
        img = Image.open(path).convert('RGB').resize((224, 224), _BILINEAR)
        t = torch.from_numpy(np.array(img)).float() / 255.0
        t = ((t.permute(2, 0, 1) - mean) / std).unsqueeze(0).to(DEVICE)

        out = model(t)
        cls_idx = out['classification'].argmax(1).item()
        breed = BREED_NAMES[cls_idx] if cls_idx < len(BREED_NAMES) else str(cls_idx)
        box = out['localization'][0].cpu().tolist()
        seg = out['segmentation'][0].argmax(0).cpu().numpy()

        arr = np.array(img)
        pil_box = draw_box(arr, box, (220, 0, 0))
        seg_rgb = Image.fromarray(trimap_rgb(seg))

        wandb.log({
            f'pipeline/{idx}_image': wandb.Image(pil_box, caption=breed),
            f'pipeline/{idx}_segmentation': wandb.Image(seg_rgb),
        })
        print(f'[{path}]  breed={breed}  box={[round(v,1) for v in box]}')


# ── entry ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='all',
                   choices=['all', 'classifier', 'localizer', 'segmentation', 'pipeline'])
    p.add_argument('--data', default='../oxford-iiit-pet')
    p.add_argument('--save_dir', default='checkpoints')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--wandb_project', default='da6401-assignment2')
    p.add_argument('--images', nargs='+', default=[],
                   help='Image paths for pipeline showcase (--task pipeline)')
    args = p.parse_args()

    wandb.init(project=args.wandb_project, name=f'eval_{args.task}', config=vars(args))

    if args.task in ('all', 'classifier'):
        eval_classifier(args)
    if args.task in ('all', 'localizer'):
        eval_localizer(args)
    if args.task in ('all', 'segmentation'):
        eval_segmentation(args)
    if args.task in ('all', 'pipeline') or args.images:
        run_pipeline(args)

    wandb.finish()


if __name__ == '__main__':
    main()
