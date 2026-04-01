import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.dataloader import get_loader
from utils.metrics import Evaluator
from utils.utils import clip_gradient, adjust_lr

from model.duha_net import create_model
from model.udt import uncertainty

def train_one_epoch(train_loader, model, udt, optimizer, opt, epoch):
    model.train()
    Eva_train = Evaluator(opt.num_classes)
    epoch_loss = 0
    length = 0

    udt.set_epoch(epoch)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for i, sample in enumerate(pbar):
        if isinstance(sample, dict):
            X = sample['image'].cuda()
            Y = sample['label'].cuda()
        elif isinstance(sample, (tuple, list)):
            if len(sample) >= 2:
                if isinstance(sample[0], dict):
                    X = sample[0]['image'].cuda()
                    Y = sample[0]['label'].cuda()
                else:
                    X = sample[0].cuda()
                    Y = sample[1].cuda()
            else:
                continue
        else:
            continue

        Y = Y.long()

        optimizer.zero_grad()

        loss, uncertainty, confidence, outputs = udt.forward(model, X, Y)

        loss.backward()

        if opt.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)

        optimizer.step()

        epoch_loss += loss.item()

        if isinstance(outputs, dict):
            pred = outputs['seg'].data.cpu().numpy()
        else:
            pred = outputs.data.cpu().numpy()

        target = Y.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        Eva_train.add_batch(target, pred)
        length += 1

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Unc': f'{uncertainty.mean().item():.3f}' if uncertainty is not None else 'N/A'
        })

    F1 = Eva_train.F1Score()
    IOU = Eva_train.Intersection_over_Union()
    train_loss = epoch_loss / length

    print(f'\nEpoch [{epoch}/{opt.epoch}] Training Complete')
    print(f'   Loss: {train_loss:.4f}')
    print(f'   [Training] IOU: {IOU[1]:.4f}, F1: {F1:.4f}')

    stats = udt.get_stats()
    print(f'   UDT: Uncertainty={stats["uncertainty"]:.3f}, Consistency={stats["consistency"]:.3f}')

    return train_loss, IOU[1], Eva_train


def validate(val_loader, model, opt):
    model.eval()
    Eva_val = Evaluator(opt.num_classes)

    with torch.no_grad():
        for sample in tqdm(val_loader, desc="Validating"):
            if isinstance(sample, dict):
                X = sample['image'].cuda()
                Y = sample['label'].cuda()
            elif isinstance(sample, (tuple, list)):
                if len(sample) >= 2:
                    if isinstance(sample[0], dict):
                        X = sample[0]['image'].cuda()
                        Y = sample[0]['label'].cuda()
                    else:
                        X = sample[0].cuda()
                        Y = sample[1].cuda()
                else:
                    continue
            else:
                continue

            Y = Y.long()

            outputs = model(X)

            if isinstance(outputs, dict):
                pred = outputs['seg'].data.cpu().numpy()
            elif isinstance(outputs, (tuple, list)):
                pred = outputs[0].data.cpu().numpy()
            else:
                pred = outputs.data.cpu().numpy()

            target = Y.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            Eva_val.add_batch(target, pred)

    F1 = Eva_val.F1Score()
    IOU = Eva_val.Intersection_over_Union()

    print(f'\n[Validation] IOU: {IOU[1]:.4f}, F1: {F1:.4f}')

    return IOU[1], F1, Eva_val


def main():
    parser = argparse.ArgumentParser(description='Train DBMFNet with UDT-Full')

    parser.add_argument('--data_name', type=str, default='vaihingen', help='dataset name')
    parser.add_argument('--train_root', type=str, default=None, help='training image root')
    parser.add_argument('--train_gt', type=str, default=None, help='training gt root')
    parser.add_argument('--val_root', type=str, default=None, help='validation image root')
    parser.add_argument('--val_gt', type=str, default=None, help='validation gt root')
    parser.add_argument('--val', type=str, default='test', help='val or test')

    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=45, help='every n epochs decay learning rate')

    parser.add_argument('--num_classes', type=int, default=6, help='number of segmentation classes')
    parser.add_argument('--feat_dim', type=int, default=128, help='feature dimension')
    parser.add_argument('--use_hfem_stages', type=str, default='0,1', help='stages to use HFEM')

    parser.add_argument('--consistency_weight', type=float, default=0.4, help='consistency weight')
    parser.add_argument('--entropy_weight', type=float, default=0.4, help='entropy weight')
    parser.add_argument('--hfem_weight', type=float, default=0.2, help='HFEM weight')
    parser.add_argument('--uncertainty_loss_weight', type=float, default=0.2, help='uncertainty loss weight')
    parser.add_argument('--consistency_loss_weight', type=float, default=0.1, help='consistency loss weight')
    parser.add_argument('--base_threshold', type=float, default=0.7, help='base threshold')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
    parser.add_argument('--weight_type', type=str, default='exp', choices=['exp', 'square', 'linear'],
                        help='weight type')

    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--save_path', type=str, default='./output1/', help='save path')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    print(f'USE GPU {opt.gpu_id}')

    os.makedirs(opt.save_path, exist_ok=True)

    use_hfem_stages = [int(s) for s in opt.use_hfem_stages.split(',')]

    if opt.data_name == 'potsdam':
        opt.train_root = ''
        opt.train_gt = ''
        if opt.val == 'val':
            opt.val_root = ''
            opt.val_gt = ''
        elif opt.val == 'test':
            opt.val_root = ''
            opt.val_gt = ''

    print("\n" + "=" * 60)
    print("=" * 60)
    print(f"Dataset: {opt.data_name}")
    print(f"Image Size: {opt.trainsize}")
    print(f"Batch Size: {opt.batchsize}")
    print(f"Epochs: {opt.epoch}")
    print(f"Learning Rate: {opt.lr}")
    print(f"Feature Dimension: {opt.feat_dim}")
    print(f"HFEM Stages: {use_hfem_stages}")
    print("=" * 60)
    print("\nUDT-Full Configuration:")
    print(f"  Uncertainty Weights: Consistency={opt.consistency_weight}, Entropy={opt.entropy_weight}, HFEM={opt.hfem_weight}")
    print(f"  Loss Weights: Uncertainty={opt.uncertainty_loss_weight}, Consistency={opt.consistency_loss_weight}")
    print(f"  Dynamic Threshold: base={opt.base_threshold}, warmup={opt.warmup_epochs}")
    print(f"  Weight Type: {opt.weight_type}")
    print("=" * 60)

    print("\nLoading training data...")
    train_loader = get_loader(
        img_root=opt.train_root,
        gt_root=opt.train_gt,
        dataset_name=opt.data_name,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        mode='train',
        label_ratio=1.0,
        use_partial_labels=False,
        num_workers=4,
        shuffle=True
    )

    print("Loading validation data...")
    val_loader = get_loader(
        img_root=opt.val_root,
        gt_root=opt.val_gt,
        dataset_name=opt.data_name,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        mode='val',
        label_ratio=1.0,
        use_partial_labels=False,
        num_workers=4,
        shuffle=False
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")


    model = create_model(
        num_classes=opt.num_classes,
        feat_dim=opt.feat_dim,
        use_hfem_stages=use_hfem_stages,
        img_size=opt.trainsize
    ).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M (Trainable: {trainable_params / 1e6:.2f}M)")

    print("\nCreating UDT-Full training strategy...")
    udt = uncertainty(
        num_classes=opt.num_classes,
        consistency_weight=opt.consistency_weight,
        entropy_weight=opt.entropy_weight,
        haar_weight=opt.hfem_weight,
        uncertainty_loss_weight=opt.uncertainty_loss_weight,
        consistency_loss_weight=opt.consistency_loss_weight,
        base_threshold=opt.base_threshold,
        warmup_epochs=opt.warmup_epochs,
        weight_type=opt.weight_type,
        verbose=True,
        device='cuda'
    )

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    start_epoch = 0
    best_iou = 0
    best_epoch = 0

    if opt.resume is not None and os.path.exists(opt.resume):
        print(f"\nLoading checkpoint: {opt.resume}")
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0)
        best_epoch = checkpoint.get('best_epoch', 0)
        print(f"Resuming from epoch {start_epoch}")

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(start_epoch, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)

        train_loss, train_iou, Eva_train = train_one_epoch(
            train_loader, model, udt, optimizer, opt, epoch
        )

        val_iou, val_f1, Eva_val = validate(val_loader, model, opt)

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            save_path = os.path.join(opt.save_path, f'DBMFNet_{opt.data_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_iou': best_iou,
                'best_epoch': best_epoch,
                'udt_stats': udt.get_stats()
            }, save_path)
            print(f'Best model saved: IoU={val_iou:.4f} (epoch {epoch})')

        print(f'\nCurrent Best: IoU={best_iou:.4f} (epoch {best_epoch})')

    print("\n" + "=" * 60)
    print(f"Training Complete! Best IoU: {best_iou:.4f} (epoch {best_epoch})")
    print("=" * 60)


if __name__ == '__main__':
    main()