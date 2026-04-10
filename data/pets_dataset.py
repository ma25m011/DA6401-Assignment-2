"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_BILINEAR = getattr(getattr(Image, 'Resampling', Image), 'BILINEAR')
_NEAREST = getattr(getattr(Image, 'Resampling', Image), 'NEAREST')


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader skeleton."""

    def __init__(self, root, split='trainval', transform=None, target_size=(224, 224)):
        self.img_dir = os.path.join(root, 'images')
        self.trimap_dir = os.path.join(root, 'annotations', 'trimaps')
        self.xml_dir = os.path.join(root, 'annotations', 'xmls')
        self.target_size = target_size
        self.transform = transform
        self.split = split

        split_file = os.path.join(root, 'annotations', f'{split}.txt')
        self.samples = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                name, class_id = parts[0], int(parts[1]) - 1  # 0-indexed
                self.samples.append((name, class_id))

    def __len__(self):
        return len(self.samples)

    def _parse_bbox(self, xml_path, orig_w, orig_h):
        bb = ET.parse(xml_path).getroot().find('.//bndbox')
        coords = {e.tag: float(e.text or 0) for e in (bb or [])}
        xmin, ymin = coords['xmin'], coords['ymin']
        xmax, ymax = coords['xmax'], coords['ymax']

        tw, th = self.target_size
        xmin, xmax = xmin * tw / orig_w, xmax * tw / orig_w
        ymin, ymax = ymin * th / orig_h, ymax * th / orig_h

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        return torch.tensor([cx, cy, xmax - xmin, ymax - ymin], dtype=torch.float32)

    def __getitem__(self, idx):
        name, label = self.samples[idx]

        img = Image.open(os.path.join(self.img_dir, name + '.jpg')).convert('RGB')
        orig_w, orig_h = img.size
        img = img.resize(self.target_size, _BILINEAR)

        do_flip = self.split == 'trainval' and np.random.rand() > 0.5
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        img_t = torch.from_numpy(np.array(img)).float() / 255.0
        img_t = img_t.permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_t = (img_t - mean) / std

        trimap = Image.open(os.path.join(self.trimap_dir, name + '.png')).resize(self.target_size, _NEAREST)
        if do_flip:
            trimap = trimap.transpose(Image.FLIP_LEFT_RIGHT)
        trimap_t = torch.from_numpy(np.array(trimap)).long() - 1  # 0,1,2

        xml_path = os.path.join(self.xml_dir, name + '.xml')
        if os.path.exists(xml_path):
            bbox = self._parse_bbox(xml_path, orig_w, orig_h)
            if do_flip:
                tw = self.target_size[0]
                bbox[0] = tw - bbox[0]  # flip cx
            has_bbox = True
        else:
            bbox = torch.zeros(4, dtype=torch.float32)
            has_bbox = False

        if self.transform:
            img_t = self.transform(img_t)

        return img_t, label, bbox, trimap_t, has_bbox
