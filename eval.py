import os
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# get root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, '..', '..'))

import matplotlib.pyplot as plt
import yaml

from tqdm import tqdm
from PIL import Image

import numpy as np
from types import SimpleNamespace
from dataset.mask_fungi import MaskFungiTastic
import torch

from torchmetrics.functional import jaccard_index


def evaluate_single_image(idx, dataset, masks_path, thresh, vis):
    image, gt_masks, class_id, file_path, label_data = dataset[idx]
    # 获取相对于数据根目录的路径
    rel_path = os.path.relpath(file_path, dataset.img_root)
    # 构建正确的掩码路径
    mask_path = os.path.join(masks_path, rel_path)
    pred_mask = Image.open(mask_path)

    # resize pred_mask to gt_mask size
    pred_mask = pred_mask.resize((gt_masks.shape[1], gt_masks.shape[0]), Image.NEAREST)

    # Handle different segmentation tasks
    if dataset.seg_task == 'binary':
        # Binary segmentation - single mask
        gt_mask = gt_masks
        iou = jaccard_index(
            preds=(torch.tensor(np.array(pred_mask)) / 255),
            target=torch.tensor(gt_mask),
            task='binary',
            threshold=thresh,
        )
    else:
        raise ValueError(f"Unknown seg_task or seg_task with evaluation not implemented: {dataset.seg_task}")

    if vis:
        # show image, gt_mask and pred_mask
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title(f"Class: {dataset.category_id2label[class_id]}")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask)
        plt.title(f"GT Mask ({dataset.seg_task})")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask)
        plt.title(f"Pred Mask")
        plt.axis('off')
        plt.suptitle(f"ID: {idx}; IoU: {iou}")
        plt.show()

    return iou.item()


def evaluate_saved_masks(
    dataset,
    masks_path,
    debug=False,
    vis=False,
    thresh=0.5,
    result_dir=None,
    chunk_size=10,
    show_mask=False,
    parallel=False,
    only_existing_masks=False,
    max_samples=0
):
    ious = []
    if only_existing_masks:
        existing = []
        for idx in range(len(dataset)):
            _, _, _, file_path, _ = dataset[idx]
            # 获取相对于数据根目录的路径
            rel_path = os.path.relpath(file_path, dataset.img_root)
            # 构建正确的掩码路径
            mask_path = os.path.join(masks_path, rel_path)
            if os.path.exists(mask_path):
                existing.append(idx)
        idxs = np.array(existing)
    else:
        idxs = np.arange(len(dataset))
    if debug:
        idxs = np.random.choice(idxs, min(10, len(idxs)), replace=False)
    if max_samples and max_samples > 0:
        idxs = idxs[:max_samples]

    if parallel:
        with ThreadPoolExecutor(max_workers=10) as executor:
            for i in tqdm(range(0, len(idxs), chunk_size)):
                chunk = idxs[i:i + chunk_size]
                futures = [executor.submit(evaluate_single_image, idx, dataset, masks_path, thresh, vis) for idx in chunk]
                for future in as_completed(futures):
                    ious.append(future.result())
    else:
        for idx in tqdm(idxs):
        # for idx in tqdm(idxs[:20]):
            iou = evaluate_single_image(idx, dataset, masks_path, thresh, vis)
            ious.append(iou)

    ious = np.array(ious)
    iou_all = ious.mean()
    print(f"IoU: {iou_all}")

    # exit()

    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        # save all as npy
        np.save(result_dir / f'ious_thresh{int(thresh * 100)}.npy', ious)
        # save the mean and std
        with open(result_dir / f'iou_thresh{int(thresh * 100)}.txt', 'w') as f:
            f.write(f"IoU: {iou_all:.2f} +/- {ious.std():.2f}")

    # plot the iou histogram and show the mean and stds
    plt.hist(ious, bins=20)
    plt.axvline(iou_all, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(iou_all + ious.std(), color='g', linestyle='dashed', linewidth=1)
    plt.axvline(iou_all - ious.std(), color='g', linestyle='dashed', linewidth=1)
    plt.legend(['mean', 'std'])
    if result_dir is not None:
        plt.savefig(result_dir / f'iou_hist_thresh{int(thresh * 100)}.png')
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate segmentation masks')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    args = parser.parse_args()
    
    split = 'val'
    with open(os.path.join(SCRIPT_DIR, 'config/seg.yaml'), "r") as f:
    # with open('config/seg.yaml', "r") as f:
        cfg = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg)

    result_dir = Path(cfg.path_out) / 'results' / 'seg' / split

    dataset = MaskFungiTastic(
        root=cfg.data_path,
        split=split,
        size='300',
        task='closed',
        data_subset='Mini',
        transform=None,
        seg_task='binary',  # Default to binary for evaluation
        debug=False,
    )

    # eval controls
    seg_cfg = getattr(cfg, 'seg', {}) if hasattr(cfg, 'seg') else {}
    eval_cfg = seg_cfg.get('eval', {}) if isinstance(seg_cfg, dict) else {}
    evaluate_saved_masks(
        dataset=dataset,
        # precomputed segmentation masks - TODO verify path, maybe further subpath needs to be added, ie 'FungiTastic'
        masks_path=os.path.join(cfg.mask_path),
        result_dir=result_dir,
        vis=args.vis,
        thresh=0.5,
        only_existing_masks=bool(eval_cfg.get('only_existing_masks', False)),
        max_samples=int(eval_cfg.get('max_samples', 0)),
        debug=bool(eval_cfg.get('debug', False)),
        parallel=bool(eval_cfg.get('parallel', False)),
    )


if __name__ == '__main__':
    main()