import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import prettytable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from utils import dataloader
from utils.metrics import Evaluator
from model.duha_net import create_model

CLASS_NAMES = ['Impervious', 'Building', 'Low Veg', 'Tree', 'Car', 'Background']

PALETTE = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0],
]


def onehot_to_mask(semantic_map, palette):
    colour_codes = np.array(palette)
    semantic_map = np.uint8(colour_codes[semantic_map.astype(np.uint8)])
    return semantic_map


def compute_metrics(all_labels, all_preds, num_classes):
    iou_per_class = []
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))

        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        iou_per_class.append(iou)
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    mIoU = np.mean(iou_per_class)
    mF1 = np.mean(f1_per_class)
    mPrecision = np.mean(precision_per_class)
    mRecall = np.mean(recall_per_class)
    oa = accuracy_score(all_labels, all_preds)

    return {
        'mIoU': mIoU,
        'mF1': mF1,
        'mPrecision': mPrecision,
        'mRecall': mRecall,
        'OA': oa,
        'IoU_per_class': iou_per_class,
        'F1_per_class': f1_per_class,
        'Precision_per_class': precision_per_class,
        'Recall_per_class': recall_per_class
    }


def plot_confusion_matrix(labels, preds, num_classes, save_path, model_name):
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=CLASS_NAMES[:num_classes],
                yticklabels=CLASS_NAMES[:num_classes])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def plot_per_class_metrics(metrics, save_path, model_name):
    num_classes = len(CLASS_NAMES)
    x = np.arange(num_classes)
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, metrics['IoU_per_class'], width, label='IoU', color='#2ecc71')
    plt.bar(x - 0.5 * width, metrics['Precision_per_class'], width, label='Precision', color='#3498db')
    plt.bar(x + 0.5 * width, metrics['Recall_per_class'], width, label='Recall', color='#e74c3c')
    plt.bar(x + 1.5 * width, metrics['F1_per_class'], width, label='F1', color='#f39c12')

    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title(f'Per-Class Metrics - {model_name}')
    plt.xticks(x, CLASS_NAMES[:num_classes], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics chart saved: {save_path}")


def test_dbmfnet(test_loader, model, device, save_vis=False, vis_save_path='./vis_results/', model_name='DBMFNet'):
    model.eval()
    model.to(device)

    Eva = Evaluator(6)
    all_preds = []
    all_labels = []

    if save_vis:
        os.makedirs(vis_save_path, exist_ok=True)

    with torch.no_grad():
        for i, sample in enumerate(tqdm(test_loader, desc="Testing")):
            if isinstance(sample, dict):
                X = sample['image'].cuda()
                Y = sample['label'].cuda()
                filename = str(i)
            elif isinstance(sample, (tuple, list)):
                if len(sample) >= 2:
                    if isinstance(sample[0], dict):
                        X = sample[0]['image'].cuda()
                        Y = sample[0]['label'].cuda()
                        filename = sample[1] if len(sample) > 1 else str(i)
                    else:
                        X = sample[0].cuda()
                        Y = sample[1].cuda()
                        filename = str(i)
                else:
                    continue
            else:
                continue

            Y = Y.long()

            output = model(X)

            if isinstance(output, dict):
                output = output['seg']
            elif isinstance(output, tuple):
                output = output[0]

            output = F.interpolate(output, size=Y.shape[1:], mode='bilinear', align_corners=True)

            pred = torch.argmax(output, dim=1)

            pred_np = pred.cpu().numpy()
            target_np = Y.cpu().numpy()

            for b in range(pred_np.shape[0]):
                Eva.add_batch(target_np[b], pred_np[b])
                all_preds.extend(pred_np[b].flatten())
                all_labels.extend(target_np[b].flatten())

            if save_vis and i < 10:
                for b in range(min(pred_np.shape[0], 10 - i)):
                    img = X[b].cpu().numpy().transpose(1, 2, 0)
                    img = np.clip(img, 0, 1)

                    gt_rgb = onehot_to_mask(target_np[b], PALETTE)
                    pred_rgb = onehot_to_mask(pred_np[b], PALETTE)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img)
                    axes[0].set_title('Input Image')
                    axes[0].axis('off')
                    axes[1].imshow(gt_rgb)
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    axes[2].imshow(pred_rgb)
                    axes[2].set_title(f'DBMFNet Prediction')
                    axes[2].axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_save_path, f'sample_{i}_{b}.png'), dpi=150)
                    plt.close()

    F1 = Eva.F1Score()
    IOU = Eva.Intersection_over_Union()
    mIoU = Eva.Mean_Intersection_over_Union()
    Precision = Eva.Precision()
    Recall = Eva.Recall()
    OA = Eva.OA()

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), 6)

    return {
        'mIoU': mIoU,
        'IoU_per_class': IOU,
        'F1': F1,
        'Precision': Precision,
        'Recall': Recall,
        'OA': OA,
        'metrics': metrics,
        'all_labels': np.array(all_labels),
        'all_preds': np.array(all_preds)
    }


def print_results(results, model_name, data_name):
    print("\n" + "=" * 70)
    print(f"Test Results - {model_name} on {data_name}")
    print("=" * 70)

    print(f"\nOverall Metrics:")
    print(f"   mIoU: {results['mIoU']:.4f}")
    print(f"   F1 Score: {results['F1']:.4f}")
    print(f"   Precision: {results['Precision']:.4f}")
    print(f"   Recall: {results['Recall']:.4f}")
    print(f"   OA: {results['OA']:.4f}")

    print(f"\nPer-Class IoU:")
    table = prettytable.PrettyTable()
    table.field_names = ['Class', 'IoU']
    for i, name in enumerate(CLASS_NAMES):
        table.add_row([name, f"{results['IoU_per_class'][i]:.4f}"])
    table.add_row(['mIoU', f"{results['mIoU']:.4f}"])
    print(table)


def save_results(results, save_path, model_name, data_name, generate_plots=True):
    os.makedirs(save_path, exist_ok=True)

    txt_path = os.path.join(save_path, f'{model_name}_{data_name}_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {data_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write("[Overall Metrics]\n")
        f.write("-" * 40 + "\n")
        f.write(f"mIoU: {results['mIoU']:.4f}\n")
        f.write(f"F1 Score: {results['F1']:.4f}\n")
        f.write(f"Precision: {results['Precision']:.4f}\n")
        f.write(f"Recall: {results['Recall']:.4f}\n")
        f.write(f"OA: {results['OA']:.4f}\n\n")

        f.write("[Per-Class IoU]\n")
        f.write("-" * 40 + "\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"{name:<18}: {results['IoU_per_class'][i]:.4f}\n")
        f.write(f"{'mIoU':<18}: {results['mIoU']:.4f}\n")

    print(f"Results saved: {txt_path}")

    if generate_plots:
        plot_confusion_matrix(results['all_labels'], results['all_preds'], 6,
                              os.path.join(save_path, f'{model_name}_{data_name}_confusion_matrix.png'),
                              model_name)
        plot_per_class_metrics(results['metrics'],
                               os.path.join(save_path, f'{model_name}_{data_name}_per_class_metrics.png'),
                               model_name)


def main():
    parser = argparse.ArgumentParser(description='Test DBMFNet Model')

    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_name', type=str, default='vaihingen', help='Dataset name')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimension')
    parser.add_argument('--use_hfem_stages', type=str, default='0,1', help='HFEM stages')

    parser.add_argument('--batchsize', type=int, default=4, help='Batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='Image size')
    parser.add_argument('--num_workers', type=int, default=2, help='Num workers')

    parser.add_argument('--save_path', type=str, default='./test_results/', help='Save path')
    parser.add_argument('--save_vis', action='store_true', help='Save visualization')
    parser.add_argument('--no_plots', action='store_true', help='Skip plots')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    use_hfem_stages = [int(s) for s in opt.use_hfem_stages.split(',')]

    if opt.data_name == 'vaihingen':
        test_img = ''
        test_gt = ''
    elif opt.data_name == 'potsdam':
        test_img = ''
        test_gt = ''
    else:
        raise ValueError(f"Unknown dataset: {opt.data_name}")

    print("\nLoading test data...")
    test_loader = dataloader.get_loader(
        img_root=test_img,
        gt_root=test_gt,
        dataset_name=opt.data_name,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize,
        mode='test',
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=True
    )
    print(f"Test set: {len(test_loader.dataset)} images")


    model = create_model(
        num_classes=opt.num_classes,
        feat_dim=opt.feat_dim,
        img_size=opt.trainsize,
        use_hfem_stages=use_hfem_stages
    ).to(device)

    print(f"Loading model weights: {opt.model_path}")
    checkpoint = torch.load(opt.model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    print("Model weights loaded successfully")

    print("\n" + "=" * 50)
    print("Starting Testing")
    print("=" * 50)

    vis_path = os.path.join(opt.save_path, 'visualization') if opt.save_vis else None

    results = test_dbmfnet(
        test_loader=test_loader,
        model=model,
        device=device,
        save_vis=opt.save_vis,
        vis_save_path=vis_path,
        model_name='DBMFNet'
    )

    print_results(results, 'DBMFNet', opt.data_name)

    save_results(
        results=results,
        save_path=opt.save_path,
        model_name='DBMFNet',
        data_name=opt.data_name,
        generate_plots=not opt.no_plots
    )

    print(f"\nTesting complete! Results saved to: {opt.save_path}")


if __name__ == '__main__':
    main()