from typing import List, Tuple, Dict, Any
import os
import numpy as np
from PIL import Image
import torch

import groundingdino.datasets.transforms as T
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download

from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor

# Mapping of SAM model types to their corresponding checkpoint filenames
MODEL2FILENAME = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth"
}

# URLs for downloading SAM model checkpoints
SAM_MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

class GDINOSAM:
    """
    A wrapper class that combines Grounding-DINO and SAM (Segment Anything Model) for end-to-end object detection and segmentation.
    
    This class provides a unified interface for:
    1. Detecting objects using Grounding-DINO based on text prompts
    2. Generating segmentation masks using SAM for the detected objects
    
    Attributes:
        text_prompt (str): The text prompt used for object detection
        dataframes_dir (str): Directory for storing dataframes (if needed)
        name (str): Identifier for the model
        device (str): Device to run the models on ('cuda' or 'cpu')
        sam_predictor (SamPredictor): SAM model predictor instance
        groundingdino: Grounding-DINO model instance
        image_transform: Image transformation pipeline for Grounding-DINO
    """

    def __init__(
        self,
        ckpt_path: str,
        text_prompt: str,
        dataframes_dir: str = "",
        sam_type: str = "vit_h",
        dino_version: str = "B",
        return_prompts: bool = False,
        seg_cfg: Any = None,
    ):
        """
        Initialize the GDINOSAM model.

        Args:
            ckpt_path (str): Path to store/load model checkpoints
            text_prompt (str): Text prompt for object detection
            dataframes_dir (str, optional): Directory for dataframes. Defaults to "".
            sam_type (str, optional): Type of SAM model to use ('vit_h', 'vit_l', or 'vit_b'). Defaults to "vit_h".
            dino_version (str, optional): Version of Grounding-DINO to use ('B' or 'T'). Defaults to "B".
            return_prompts (bool, optional): Whether to return prompts in predictions. Defaults to False.
        """
        self.text_prompt = text_prompt
        self.dataframes_dir = dataframes_dir
        self.name = "gdino_sam"
        # device selection: prefer CUDA if available unless seg_cfg overrides
        cfg_device = None
        if seg_cfg is not None and isinstance(seg_cfg, dict):
            cfg_device = seg_cfg.get('device', None)
        self.device = (
            cfg_device if cfg_device in ["cuda", "cpu"] else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # store segmentation enhancement options with sensible defaults
        self.seg_opts = self._parse_seg_cfg(seg_cfg)

        self._build_sam(sam_type, ckpt_path)
        self._build_dino(dino_version, return_prompts)
        self.image_transform = get_image_transform()

    def predict(self, image_pil: Image.Image, box_thr: float = 0.3, txt_thr: float = 0.25):
        """
        Generate segmentation masks for objects in the image based on the text prompt.

        Args:
            image_pil (Image.Image): Input PIL image
            box_thr (float, optional): Confidence threshold for box detection. Defaults to 0.3.
            txt_thr (float, optional): Confidence threshold for text matching. Defaults to 0.25.

        Returns:
            tuple: (mask, extra)
                - mask (np.ndarray): Binary mask of detected objects
                - extra (dict): Additional information including:
                    - confs_gdino: Grounding-DINO confidence scores
                    - confs_seg: SAM segmentation confidence scores
                    - n_inst: Number of detected instances
                    - phrases: Detected phrases
                    - boxes: Bounding boxes of detected objects
        """
        # Optionally run TTA; otherwise single pass
        if self.seg_opts['tta']['enabled']:
            masks_prob_accum = []
            weights = []
            base_w, base_h = image_pil.size
            for scale in self.seg_opts['tta']['scales']:
                for flip in ([False, True] if self.seg_opts['tta']['hflip'] else [False]):
                    img_tta = image_pil
                    if abs(scale - 1.0) > 1e-6:
                        img_tta = image_pil.resize((int(base_w * scale), int(base_h * scale)), Image.BILINEAR)
                    if flip:
                        img_tta = img_tta.transpose(Image.FLIP_LEFT_RIGHT)
                    mask_bin_tta, extra_tta, mask_prob_tta = self._predict_single(img_tta, box_thr, txt_thr, return_prob=True)
                    # map back
                    mask_prob_tta = Image.fromarray((mask_prob_tta * 255).astype(np.uint8))
                    if flip:
                        mask_prob_tta = mask_prob_tta.transpose(Image.FLIP_LEFT_RIGHT)
                    if mask_prob_tta.size != (base_w, base_h):
                        mask_prob_tta = mask_prob_tta.resize((base_w, base_h), Image.BILINEAR)
                    masks_prob_accum.append(np.array(mask_prob_tta, dtype=np.float32) / 255.0)
                    weights.append(1.0)
            if len(masks_prob_accum) == 0:
                final_prob = np.zeros((base_h, base_w), dtype=np.float32)
            else:
                final_prob = np.average(np.stack(masks_prob_accum, axis=0), axis=0, weights=np.array(weights))
            final_mask = (final_prob >= self.seg_opts['postproc']['combine_vote_thr']).astype(np.uint8)
            # postproc on final mask
            final_mask = self._postprocess_mask(final_mask, image_pil) if self.seg_opts['postproc']['enabled'] else final_mask
            # minimal extra
            extra = {
                "tta": True,
            }
            return final_mask, extra
        else:
            mask_bin, extra, mask_prob = self._predict_single(image_pil, box_thr, txt_thr, return_prob=True)
            if self.seg_opts['postproc']['enabled']:
                mask_bin = self._postprocess_mask(mask_bin, image_pil)
            return mask_bin, extra

    def _predict_single(self, image_pil: Image.Image, box_thr: float, txt_thr: float, return_prob: bool = False):
        # 1) GroundingDINO with prompt ensemble and adaptive thresholds
        pe = self.seg_opts['prompt_ensemble']
        prompts = [self.text_prompt]
        if pe['enabled']:
            prompts = pe['prompts']

        all_boxes = []
        all_logits = []
        all_phrases = []

        for p in prompts:
            bthr, tthr = pe['box_thr_init'], pe['text_thr_init']
            boxes, gdino_logits, phrases = self._predict_gdino(image_pil, p, bthr, tthr)

            if pe['adaptive']:
                # simple adaptive search
                if len(boxes) < pe['adapt_min_boxes']:
                    # gradually lower thresholds within range
                    for bt in torch.linspace(pe['box_thr_range'][0], bthr, steps=3):
                        for tt in torch.linspace(pe['text_thr_range'][0], tthr, steps=3):
                            boxes, gdino_logits, phrases = self._predict_gdino(image_pil, p, float(bt), float(tt))
                            if len(boxes) >= pe['adapt_min_boxes']:
                                break
                        if len(boxes) >= pe['adapt_min_boxes']:
                            break
                elif len(boxes) > pe['adapt_max_boxes']:
                    # raise thresholds to prune
                    for bt in torch.linspace(bthr, pe['box_thr_range'][1], steps=3):
                        for tt in torch.linspace(tthr, pe['text_thr_range'][1], steps=3):
                            boxes, gdino_logits, phrases = self._predict_gdino(image_pil, p, float(bt), float(tt))
                            if len(boxes) <= pe['adapt_max_boxes']:
                                break
                        if len(boxes) <= pe['adapt_max_boxes']:
                            break

            all_boxes.append(boxes)
            all_logits.append(gdino_logits)
            all_phrases += phrases

        if len(all_boxes) > 0:
            boxes = torch.cat([b for b in all_boxes if len(b) > 0], dim=0) if any(len(b) > 0 for b in all_boxes) else torch.empty((0,4))
            logits = torch.cat([l for l in all_logits if len(l) > 0], dim=0) if any(len(l) > 0 for l in all_logits) else torch.empty((0,))
        else:
            boxes = torch.empty((0,4))
            logits = torch.empty((0,))

        # NMS to fuse prompts
        if len(boxes) > 0 and pe['enabled']:
            keep = self._nms(boxes, logits, iou_thr=pe['nms_iou'])
            boxes = boxes[keep]
            logits = logits[keep]

        # limit number of boxes
        if len(boxes) > pe['max_boxes']:
            topk = torch.topk(logits, k=pe['max_boxes']).indices
            boxes = boxes[topk]
            logits = logits[topk]

        # 2) SAM per box with multimask and jitter; combine into probability map
        mask_prob, seg_confs = self._predict_sam(image_pil, boxes)

        # 3) binarize
        mask_bin = self._binarize_prob_map(mask_prob)

        extra = {
            "confs_gdino": logits,
            "confs_seg": seg_confs,
            "n_inst": int((mask_bin > 0).sum() > 0),
            "phrases": all_phrases,
            "boxes": boxes,
        }
        if return_prob:
            return mask_bin, extra, mask_prob
        return mask_bin, extra

    def _build_sam(self, sam_type: str, ckpt_root: str):
        """
        Initialize and load the SAM model.

        Args:
            sam_type (str): Type of SAM model to use
            ckpt_root (str): Root directory for model checkpoints
        """
        ckpt_file = os.path.join(ckpt_root, MODEL2FILENAME[sam_type])
        if not os.path.exists(ckpt_file):
            os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
            torch.hub.download_url_to_file(SAM_MODEL_URLS[sam_type], ckpt_file)

        sam = sam_model_registry[sam_type](ckpt_file).to(self.device)
        self.sam_predictor = SamPredictor(sam)

    def _build_dino(self, version: str, return_prompts: bool):
        """
        Initialize and load the Grounding-DINO model.

        Args:
            version (str): Version of Grounding-DINO to use ('B' or 'T')
            return_prompts (bool): Whether to return prompts in predictions
        """
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = (
            "groundingdino_swinb_cogcoor.pth" if version == "B" else "groundingdino_swint_ogc.pth"
        )
        ckpt_cfg = (
            "GroundingDINO_SwinB.cfg.py" if version == "B" else "GroundingDINO_SwinT_OGC.py"
        )
        self.groundingdino = load_model_hf(
            repo_id=ckpt_repo_id, filename=ckpt_filename, ckpt_config_filename=ckpt_cfg, device=self.device
        )
        self.return_prompts = return_prompts

    def _predict_sam(self, image_pil: Image.Image, boxes: torch.Tensor):
        """
        Generate segmentation masks using SAM for the given bounding boxes.

        Args:
            image_pil (Image.Image): Input PIL image
            boxes (torch.Tensor): Bounding boxes from Grounding-DINO

        Returns:
            tuple: (masks, confidences)
                - masks: Segmentation masks for each box
                - confidences: Confidence scores for each mask
        """
        if len(boxes) == 0:
            h, w = image_pil.size[1], image_pil.size[0]
            return np.zeros((h, w), dtype=np.float32), []

        img_arr = np.asarray(image_pil)
        self.sam_predictor.set_image(img_arr)
        tb = self.sam_predictor.transform.apply_boxes_torch(boxes, img_arr.shape[:2]).to(self.device)

        sam_opts = self.seg_opts['sam']
        prob_accum = np.zeros(img_arr.shape[:2], dtype=np.float32)
        weight_accum = np.zeros(img_arr.shape[:2], dtype=np.float32)
        use_baseline_union = (not sam_opts['multimask']) and (not sam_opts['box_jitter']['enabled']) and (self.seg_opts.get('merge', {}).get('scheme', 'weighted') == 'union')
        seg_confs: List[float] = []

        for bi in range(tb.shape[0]):
            box = tb[bi].unsqueeze(0)

            # generate box variants (jitter)
            variants = [box]
            if sam_opts['box_jitter']['enabled']:
                scales = sam_opts['box_jitter']['scales']
                for s in scales:
                    if abs(s - 1.0) < 1e-6:
                        continue
                    cxcy = (box[:, :2] + box[:, 2:]) / 2.0
                    hw = (box[:, 2:] - box[:, :2]) * s
                    new_box = torch.cat([cxcy - hw / 2.0, cxcy + hw / 2.0], dim=1)
                    variants.append(new_box)
                # cap number of variants
                if len(variants) > sam_opts['box_jitter']['max_per_box']:
                    variants = variants[:sam_opts['box_jitter']['max_per_box']]

            # run SAM for each variant
            best_masks = []
            best_confs = []
            for vb in variants:
                masks, iou_preds, _ = self.sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=vb.to(self.device),
                    multimask_output=bool(sam_opts['multimask'])
                )
                # masks: [B, M, H, W] or [B, 1, H, W]
                masks = masks.squeeze(0).float().cpu()  # [M, H, W]
                iou_preds_np = iou_preds.squeeze(0).float().cpu().numpy().reshape(-1)

                # per-instance filtering
                if self.seg_opts.get('instance_filter', {}).get('enabled', True):
                    min_iou = float(self.seg_opts['instance_filter'].get('min_iou_pred', 0.0))
                    keep_idx = np.where(iou_preds_np >= min_iou)[0]
                    if keep_idx.size > 0:
                        masks = masks[keep_idx]
                        iou_preds_np = iou_preds_np[keep_idx]

                # select top-k per box
                k = max(1, int(sam_opts['topk_per_box'])) if sam_opts['multimask'] else 1
                if masks.ndim == 2:
                    masks = masks.unsqueeze(0)
                    iou_preds_np = np.array([iou_preds_np.item()]) if np.ndim(iou_preds_np) == 0 else iou_preds_np
                topk_idx = np.argsort(-iou_preds_np)[:k]
                for idx in topk_idx:
                    best_masks.append(masks[idx])
                    best_confs.append(float(iou_preds_np[idx]))

            # accumulate
            merge_scheme = self.seg_opts.get('merge', {}).get('scheme', 'weighted')
            if merge_scheme == 'topk_union':
                # pick top-k masks by confidence, union them
                topk = int(self.seg_opts['merge'].get('topk', 3))
                order = np.argsort(-np.array(best_confs))[:max(1, topk)]
                for oi in order:
                    m_np = best_masks[oi].numpy()
                    prob_accum = np.maximum(prob_accum, m_np)
                    weight_accum = np.maximum(weight_accum, (m_np > 0).astype(np.float32))
                    seg_confs.append(best_confs[oi])
            elif merge_scheme == 'union' or use_baseline_union:
                for m, c in zip(best_masks, best_confs):
                    m_np = m.numpy()
                    prob_accum = np.maximum(prob_accum, m_np)
                    weight_accum = np.maximum(weight_accum, (m_np > 0).astype(np.float32))
                    seg_confs.append(c)
            else:  # weighted (default)
                for m, c in zip(best_masks, best_confs):
                    m_np = m.numpy()
                    prob_accum += m_np * c
                    weight_accum += (np.ones_like(m_np) * c)
                    seg_confs.append(c)

        # avoid division by zero
        weight_accum = np.maximum(weight_accum, 1e-6)
        prob_map = prob_accum / weight_accum
        return prob_map.astype(np.float32), seg_confs

    def _binarize_prob_map(self, prob: np.ndarray) -> np.ndarray:
        strategy = self.seg_opts['postproc'].get('threshold_strategy', 'fixed')
        if strategy == 'otsu':
            try:
                import cv2
                prob_uint8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
                thr, _ = cv2.threshold(prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return (prob_uint8 >= thr).astype(np.uint8)
            except Exception:
                pass
        if strategy == 'percentile':
            p = float(self.seg_opts['postproc'].get('percentile', 0.6))
            thr = np.quantile(prob.astype(np.float32), p)
            return (prob >= thr).astype(np.uint8)
        # fixed
        thr = float(self.seg_opts['postproc'].get('combine_vote_thr', 0.5))
        return (prob >= thr).astype(np.uint8)
    
    def _predict_gdino(self, image_pil, text_prompt, box_threshold, text_threshold):
        """
        Detect objects using Grounding-DINO based on the text prompt.

        Args:
            image_pil (Image.Image): Input PIL image
            text_prompt (str): Text prompt for object detection
            box_threshold (float): Confidence threshold for box detection
            text_threshold (float): Confidence threshold for text matching

        Returns:
            tuple: (boxes, logits, phrases)
                - boxes: Detected bounding boxes
                - logits: Confidence scores for detections
                - phrases: Detected phrases
        """
        image_trans, _ = self.image_transform(image_pil, None)
        boxes, logits, phrases = predict(model=self.groundingdino,
                                         image=image_trans,
                                         caption=text_prompt,
                                         box_threshold=box_threshold,
                                         text_threshold=text_threshold,
                                         remove_combined=self.return_prompts,
                                         device=self.device)
        W, H = image_pil.size
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes, logits, phrases

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> List[int]:
        # simple greedy NMS using IoU
        if len(boxes) == 0:
            return []
        order = torch.argsort(scores, descending=True)
        keep: List[int] = []
        while order.numel() > 0:
            i = int(order[0])
            keep.append(i)
            if order.numel() == 1:
                break
            cur_box = boxes[i].unsqueeze(0)
            rest = boxes[order[1:]]
            ious = box_ops.box_iou(cur_box, rest)[0]
            inds = torch.where(ious <= iou_thr)[0]
            order = order[inds + 1]
        return keep

    def _parse_seg_cfg(self, seg_cfg: Any) -> Dict[str, Any]:
        # defaults
        defaults = {
            'device': None,
            'prompt_ensemble': {
                'enabled': False,
                'prompts': ['mushroom', 'fungus', 'toadstool', 'bracket fungus', 'mushroom cap', 'mushroom stem'],
                'max_boxes': 6,
                'nms_iou': 0.5,
                'box_thr_init': 0.30,
                'text_thr_init': 0.25,
                'adaptive': True,
                'adapt_min_boxes': 1,
                'adapt_max_boxes': 10,
                'box_thr_range': [0.20, 0.40],
                'text_thr_range': [0.20, 0.35],
            },
            'sam': {
                'multimask': False,
                'topk_per_box': 1,
                'box_jitter': {
                    'enabled': False,
                    'scales': [0.90, 1.00, 1.10],
                    'max_per_box': 2,
                },
                'center_point_prompt': False,
            },
            'tta': {
                'enabled': False,
                'scales': [1.0],
                'hflip': False,
            },
            'postproc': {
                'enabled': False,
                'min_area_ratio': 0.001,
                'fill_holes': True,
                'combine_vote_thr': 0.5,
                'crf': {
                    'enabled': False,
                    'sxy': 3,
                    'compat': 3,
                    'iters': 5,
                },
                'guided_filter': {
                    'enabled': False,
                    'radius': 3,
                    'eps': 1.0e-06,
                },
            },
        }
        if seg_cfg is None:
            return defaults
        if isinstance(seg_cfg, dict):
            # shallow merge
            def merge(a, b):
                for k, v in b.items():
                    if k in a and isinstance(a[k], dict) and isinstance(v, dict):
                        merge(a[k], v)
                    else:
                        a[k] = v
            cfg_copy = {k: (v.copy() if isinstance(v, dict) else v) for k, v in defaults.items()}
            merge(cfg_copy, seg_cfg)
            return cfg_copy
        return defaults

    def _postprocess_mask(self, mask_bin: np.ndarray, image_pil: Image.Image) -> np.ndarray:
        try:
            import cv2
        except Exception:
            return mask_bin
        h, w = mask_bin.shape
        min_area = max(1, int(h * w * float(self.seg_opts['postproc']['min_area_ratio'])))
        # remove small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), connectivity=8)
        out = np.zeros_like(mask_bin, dtype=np.uint8)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                out[labels == i] = 1
        if self.seg_opts['postproc']['fill_holes']:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        # optional morphology
        ok = int(self.seg_opts['postproc'].get('open_kernel', 0))
        ck = int(self.seg_opts['postproc'].get('close_kernel', 0))
        if ok > 0:
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((ok, ok), np.uint8))
        if ck > 0:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((ck, ck), np.uint8))
        if bool(self.seg_opts['postproc'].get('keep_largest_component', False)):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), connectivity=8)
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                li = int(1 + np.argmax(areas))
                out = (labels == li).astype(np.uint8)
        return out


def load_model_hf(repo_id, filename, ckpt_config_filename=None, model_config_path=None, device='cpu'):
    """
    Load a model from Hugging Face Hub.

    Args:
        repo_id (str): Hugging Face repository ID
        filename (str): Name of the model checkpoint file
        ckpt_config_filename (str, optional): Name of the config file in the repo. Defaults to None.
        model_config_path (str, optional): Local path to model config. Defaults to None.
        device (str, optional): Device to load the model on. Defaults to 'cpu'.

    Returns:
        model: Loaded model instance
    """
    assert model_config_path is not None or ckpt_config_filename is not None, "Please provide either model config path or checkpoint config filename"
    if model_config_path is None:
        config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    else:
        config_file = model_config_path

    args = SLConfig.fromfile(config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print(f"Model loaded from {cache_file} \n => {log}")
    try:
        model.to(device)
    except Exception:
        pass
    model.eval()
    return model


def get_image_transform():
    """
    Get the image transformation pipeline for Grounding-DINO.

    Returns:
        T.Compose: Image transformation pipeline
    """
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform