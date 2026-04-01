import os
from PIL import Image, ImageFilter, ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import random
import cv2
from utils import custom_transforms as tr


def mask_to_onehot(mask, palette):

    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    semantic_map = np.argmax(semantic_map, axis=-1)
    return semantic_map


def onehot_to_mask(semantic_map, palette):

    colour_codes = np.array(palette)
    semantic_map = np.uint8(colour_codes[semantic_map.astype(np.uint8)])
    return semantic_map


def extract_edge_pil(mask_pil):

    gray = ImageOps.grayscale(mask_pil)
    edge = gray.filter(ImageFilter.FIND_EDGES)
    return edge


def create_partial_mask(full_mask, label_ratio, num_classes=6):

    H, W = full_mask.shape

    if label_ratio >= 1.0:
        return full_mask

    partial_mask = np.full((H, W), 255, dtype=np.uint8)

    total_pixels = H * W
    keep_pixels = int(total_pixels * label_ratio)

    unique_classes = np.unique(full_mask)
    pixels_per_class = max(keep_pixels // len(unique_classes), 1)

    for cls in unique_classes:
        if cls == 255:
            continue

        cls_indices = np.where(full_mask == cls)
        cls_count = len(cls_indices[0])

        if cls_count > 0:

            keep_this_class = min(pixels_per_class, cls_count)
            if keep_this_class > 0:

                chosen = np.random.choice(cls_count, keep_this_class, replace=False)
                rows = cls_indices[0][chosen]
                cols = cls_indices[1][chosen]
                partial_mask[rows, cols] = cls

    return partial_mask


class SemiSupervisedSegmentationDataset(data.Dataset):

    def __init__(self, img_root, gt_root, palette, trainsize, mode,
                 label_ratio=1.0, use_partial_labels=False):

        self.trainsize = trainsize
        self.image_root = img_root
        self.gt_root = gt_root
        self.palette = palette
        self.mode = mode
        self.label_ratio = label_ratio
        self.use_partial_labels = use_partial_labels

        self.image_files = []
        for ext in ['.tif', '.tiff', '.png', '.jpg']:
            if os.path.exists(self.image_root):
                for f in os.listdir(self.image_root):
                    if f.endswith(ext):
                        full_path = os.path.join(self.image_root, f)
                        if os.path.exists(full_path):
                            self.image_files.append(f)

        self.image_files = sorted(self.image_files)

        print(f"[{mode.upper()}] Found {len(self.image_files)} valid image files in {img_root}")

        if mode == 'unlabeled':
            self.images = [os.path.join(self.image_root, f) for f in self.image_files]
            self.gts = None
            print(f"[UNLABELED] Loaded {len(self.images)} unlabeled images")
        else:

            self.images = []
            self.gts = []

            for img_name in self.image_files:

                mask_name = img_name.replace('.tif', '.png').replace('.tiff', '.png').replace('.jpg', '.png')
                img_path = os.path.join(self.image_root, img_name)
                mask_path = os.path.join(self.gt_root, mask_name)

                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.images.append(img_path)
                    self.gts.append(mask_path)
                else:
                    if not os.path.exists(img_path):
                        print(f"Image file does not exist: {img_path}")
                    if not os.path.exists(mask_path):
                        print(f"Label file does not exist: {mask_path}")

            print(f"[LABELED] Valid images: {len(self.images)}")

            if len(self.images) == 0:
                raise FileNotFoundError(f"No valid image-label pairs found! Please check paths: {img_root}, {gt_root}")

            if self.use_partial_labels and self.label_ratio < 1.0:
                self.partial_masks = self._precompute_partial_masks()
                print(f"[LABELED] Loaded {len(self.images)} images, label ratio: {label_ratio * 100:.0f}%")
            else:
                print(f"[LABELED] Loaded {len(self.images)} images, full labels")

        self.size = len(self.images) if self.images else len(self.image_files)

    def _precompute_partial_masks(self):

        partial_masks = []
        for i, gt_path in enumerate(self.gts):
            try:
                full_mask = np.array(Image.open(gt_path))
                full_mask = mask_to_onehot(full_mask, self.palette)
                partial_mask = create_partial_mask(full_mask, self.label_ratio)
                partial_masks.append(partial_mask)
            except Exception as e:
                print(f"Error processing label {gt_path}: {e}")
                partial_masks.append(np.zeros((self.trainsize, self.trainsize), dtype=np.uint8))
        return partial_masks

    def __getitem__(self, index):

        if index >= self.size:
            index = 0

        try:

            if self.mode == 'unlabeled':
                if index >= len(self.images):
                    index = 0
                img_path = self.images[index]

                if not os.path.exists(img_path):
                    print(f"Image file does not exist: {img_path}, using first image")
                    img_path = self.images[0]

                image = Image.open(img_path).convert('RGB')

                sample = {'image': image}
                sample = self.transform_unlabeled(sample)

                result = {
                    'image': sample['image'],
                    'index': index,
                    'has_label': False
                }
                return result

            if index >= len(self.images):
                index = 0

            img_path = self.images[index]
            gt_path = self.gts[index]

            if not os.path.exists(img_path):
                print(f"Image file does not exist: {img_path}, using first image")
                img_path = self.images[0]

            if not os.path.exists(gt_path):
                print(f"Label file does not exist: {gt_path}, using first label")
                gt_path = self.gts[0]

            image = Image.open(img_path).convert('RGB')

            if self.use_partial_labels and self.label_ratio < 1.0:
                gt_onehot = self.partial_masks[index]
                gt = Image.fromarray(np.uint8(gt_onehot))
            else:
                try:
                    gt_np = np.array(Image.open(gt_path))
                    gt_onehot = mask_to_onehot(gt_np, self.palette)
                    gt = Image.fromarray(np.uint8(gt_onehot))
                except Exception as e:
                    print(f"Failed to load label {gt_path}: {e}")
                    gt = Image.fromarray(np.zeros((self.trainsize, self.trainsize), dtype=np.uint8))

            edge = extract_edge_pil(gt)

            sample = {'image': image, 'label': gt, 'edge': edge}

            if self.mode == 'train':
                result = self.transform_tr(sample)

                if isinstance(result, dict):
                    if 'label' not in result:
                        result['label'] = sample['label']
                    if 'edge' not in result:
                        result['edge'] = sample['edge']
                    if 'image' not in result:
                        result['image'] = sample['image']

                return result, index

            elif self.mode == 'val':
                result = self.transform_val(sample)
                if isinstance(result, dict):
                    if 'label' not in result:
                        result['label'] = sample['label']
                    if 'image' not in result:
                        result['image'] = sample['image']
                return result, index

            elif self.mode == 'test':
                file_name = os.path.basename(img_path).replace('.tif', '')
                result = self.transform_test(sample)
                return result, file_name

        except Exception as e:
            print(f"Error loading sample {index}: {e}")

            dummy_image = torch.zeros(3, self.trainsize, self.trainsize)
            dummy_label = torch.zeros(self.trainsize, self.trainsize, dtype=torch.long)

            if self.mode == 'unlabeled':
                return {'image': dummy_image, 'index': index, 'has_label': False}
            else:
                return (dummy_image, dummy_label), index

    def transform_tr(self, sample):

        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip_edge(),
            tr.RandomGaussianBlur_edge(),
            tr.FixScaleCrop_edge(crop_size=self.trainsize),
            tr.Normalize_edge(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor_edge()])
        return composed_transforms(sample)

    def transform_unlabeled(self, sample):

        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip_edge(),
            tr.RandomGaussianBlur_edge(),
            tr.FixScaleCrop_edge(crop_size=self.trainsize),
            tr.Normalize_edge(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor_edge()])
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.trainsize),
            tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __len__(self):
        return self.size


def get_loader(img_root, gt_root, dataset_name, batchsize, trainsize, mode,
               label_ratio=1.0, use_partial_labels=False, num_workers=4,
               shuffle=True, pin_memory=True):

    if dataset_name.lower() == 'potsdam':
        palette = [
            [255, 255, 255],
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
        ]
    elif dataset_name.lower() == 'vaihingen':
        palette = [
            [255, 255, 255],
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
        ]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported")

    if mode == 'unlabeled':
        dataset = SemiSupervisedSegmentationDataset(
            img_root=img_root,
            gt_root=None,
            palette=palette,
            trainsize=trainsize,
            mode=mode,
            label_ratio=1.0,
            use_partial_labels=False
        )
    else:
        dataset = SemiSupervisedSegmentationDataset(
            img_root=img_root,
            gt_root=gt_root,
            palette=palette,
            trainsize=trainsize,
            mode=mode,
            label_ratio=label_ratio,
            use_partial_labels=use_partial_labels
        )

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == 'train')
    )
    return loader


def create_semi_supervised_loaders(train_root, train_gt, val_root, val_gt,
                                   dataset_name, batchsize, trainsize,
                                   label_ratio=0.1, num_workers=4):

    labeled_loader = get_loader(
        img_root=train_root,
        gt_root=train_gt,
        dataset_name=dataset_name,
        batchsize=batchsize,
        trainsize=trainsize,
        mode='train',
        label_ratio=label_ratio,
        use_partial_labels=True,
        num_workers=num_workers,
        shuffle=True
    )

    unlabeled_loader = get_loader(
        img_root=train_root,
        gt_root=None,
        dataset_name=dataset_name,
        batchsize=batchsize,
        trainsize=trainsize,
        mode='unlabeled',
        label_ratio=1.0,
        use_partial_labels=False,
        num_workers=num_workers,
        shuffle=True
    )

    val_loader = get_loader(
        img_root=val_root,
        gt_root=val_gt,
        dataset_name=dataset_name,
        batchsize=batchsize,
        trainsize=trainsize,
        mode='val',
        label_ratio=1.0,
        use_partial_labels=False,
        num_workers=num_workers,
        shuffle=False
    )

    return labeled_loader, unlabeled_loader, val_loader