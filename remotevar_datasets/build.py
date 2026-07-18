import torch 
import numpy as np
from torch.utils.data import Dataset
from .change_dataset_simple import ChangeDataset
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
import cv2
import random
import os
from typing import List, Optional, Set, Tuple
import json
import time


class ChangeDetectionAugmentation:
    """
    Data augmentation for change detection tasks.
    Applies consistent transformations to pre-image, post-image, and mask.
    Gaussian blur is only applied to pre and post images, not to the mask.
    """
    def __init__(self, 
                 image_size=256,
                 enable_random_crop=True,
                 enable_random_flip=True,
                 enable_random_rotation=True,
                 enable_gaussian_blur=True,
                 enable_color_jitter: bool = False,
                 color_jitter_probability: float = 0.8,
                 color_jitter_brightness: float = 0.2,
                 color_jitter_contrast: float = 0.2,
                 color_jitter_saturation: float = 0.2,
                 color_jitter_hue: float = 0.05,
                 crop_scale_range=(0.8, 1.0),
                 min_crop_size: int = 64,
                 max_crop_size=None,
                 rotation_angle=10,
                 blur_probability=0.5,
                 blur_kernel_sizes=[3, 5, 7],
                 flip_probability=0.5):
        """
        Args:
            image_size: Target image size
            enable_random_crop: Enable random crop augmentation
            enable_random_flip: Enable random horizontal/vertical flip
            enable_random_rotation: Enable random rotation
            enable_gaussian_blur: Enable Gaussian blur (only for images, not mask)
            crop_scale_range: Scale range for random crop
            rotation_angle: Maximum rotation angle in degrees
            blur_probability: Probability of applying blur
            blur_kernel_sizes: List of possible blur kernel sizes
            flip_probability: Probability of flipping
        """
        self.image_size = image_size
        self.enable_random_crop = enable_random_crop
        self.enable_random_flip = enable_random_flip
        self.enable_random_rotation = enable_random_rotation
        self.enable_gaussian_blur = enable_gaussian_blur
        self.enable_color_jitter = bool(enable_color_jitter)
        self.color_jitter_probability = float(color_jitter_probability)
        self.color_jitter_brightness = float(color_jitter_brightness)
        self.color_jitter_contrast = float(color_jitter_contrast)
        self.color_jitter_saturation = float(color_jitter_saturation)
        self.color_jitter_hue = float(color_jitter_hue)
        self.crop_scale_range = crop_scale_range
        self.min_crop_size = int(min_crop_size)
        self.max_crop_size = int(max_crop_size) if max_crop_size is not None else int(image_size)
        self.rotation_angle = rotation_angle
        self.blur_probability = blur_probability
        self.blur_kernel_sizes = blur_kernel_sizes
        self.flip_probability = flip_probability

        # Import lazily to avoid pulling torchvision unless needed
        self._color_jitter = None
        if self.enable_color_jitter:
            try:
                import torchvision.transforms as _T
                self._color_jitter = _T.ColorJitter(
                    brightness=self.color_jitter_brightness,
                    contrast=self.color_jitter_contrast,
                    saturation=self.color_jitter_saturation,
                    hue=self.color_jitter_hue,
                )
            except Exception as e:
                print(f"[ChangeDetectionAugmentation] WARNING: failed to init ColorJitter ({e}); disabling.")
                self.enable_color_jitter = False
                self._color_jitter = None
    
    def __call__(self, A, B, gt):
        """
        Apply augmentations to pre-image (A), post-image (B), and ground truth mask (gt).
        
        Args:
            A: Pre-image (numpy array, HWC)
            B: Post-image (numpy array, HWC)
            gt: Ground truth mask (numpy array, HW)
            
        Returns:
            Augmented A, B, gt
        """
        h, w = A.shape[:2]
        
        # 1. Random crop (size in [min_crop_size, max_crop_size]) then resize to image_size
        if self.enable_random_crop:
            # optional: mild random scaling before cropping (kept for compatibility)
            scale = random.uniform(self.crop_scale_range[0], self.crop_scale_range[1])
            new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
            if (new_h, new_w) != (h, w):
                A = cv2.resize(A, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                B = cv2.resize(B, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            h2, w2 = A.shape[:2]
            min_cs = max(1, min(self.min_crop_size, self.image_size))
            max_cs = max(min_cs, min(self.max_crop_size, self.image_size))
            crop_size = random.randint(min_cs, max_cs)

            # If the image is smaller than the chosen crop, upscale first (no padding).
            if h2 < crop_size or w2 < crop_size:
                up_h = max(h2, crop_size)
                up_w = max(w2, crop_size)
                A = cv2.resize(A, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
                B = cv2.resize(B, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, (up_w, up_h), interpolation=cv2.INTER_NEAREST)
                h2, w2 = up_h, up_w

            top = 0 if h2 == crop_size else random.randint(0, h2 - crop_size)
            left = 0 if w2 == crop_size else random.randint(0, w2 - crop_size)

            A = A[top:top + crop_size, left:left + crop_size]
            B = B[top:top + crop_size, left:left + crop_size]
            gt = gt[top:top + crop_size, left:left + crop_size]

            # Always resize crop to target output resolution
            A = cv2.resize(A, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            B = cv2.resize(B, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        else:
            # Just resize to target size
            A = cv2.resize(A, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            B = cv2.resize(B, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            gt = cv2.resize(gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # 2. Random Flip (horizontal and/or vertical)
        if self.enable_random_flip:
            if random.random() < self.flip_probability:
                # Horizontal flip
                A = cv2.flip(A, 1)
                B = cv2.flip(B, 1)
                gt = cv2.flip(gt, 1)
            
            if random.random() < self.flip_probability:
                # Vertical flip
                A = cv2.flip(A, 0)
                B = cv2.flip(B, 0)
                gt = cv2.flip(gt, 0)
        
        # 3. Random Rotation
        if self.enable_random_rotation:
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)
            h, w = A.shape[:2]
            center = (w / 2, h / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            A = cv2.warpAffine(A, rotation_matrix, (w, h), 
                              flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_REFLECT)
            B = cv2.warpAffine(B, rotation_matrix, (w, h), 
                              flags=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_REFLECT)
            gt = cv2.warpAffine(gt, rotation_matrix, (w, h), 
                               flags=cv2.INTER_NEAREST, 
                               borderMode=cv2.BORDER_REFLECT)

        # 3.5 Color jitter (ONLY for images, NOT for mask) applied independently to pre/post
        if self.enable_color_jitter and self._color_jitter is not None:
            try:
                from PIL import Image as _Image
                if random.random() < self.color_jitter_probability:
                    A = np.array(self._color_jitter(_Image.fromarray(A)), dtype=np.uint8)
                if random.random() < self.color_jitter_probability:
                    B = np.array(self._color_jitter(_Image.fromarray(B)), dtype=np.uint8)
            except Exception:
                # Keep augmentation robust; skip jitter on failures.
                pass
        
        # 4. Gaussian Blur (only for images, NOT for mask)
        if self.enable_gaussian_blur and random.random() < self.blur_probability:
            kernel_size = random.choice(self.blur_kernel_sizes)
            A = cv2.GaussianBlur(A, (kernel_size, kernel_size), 0)
            B = cv2.GaussianBlur(B, (kernel_size, kernel_size), 0)
        
        return A, B, gt


class ChangeDatasetWrapper(Dataset):
    """Wrapper to convert ChangeDataset output to (pre_image, post_image, change_map) tuple format for change detection"""
    def __init__(self, change_dataset):
        self.change_dataset = change_dataset
    
    def __len__(self):
        return len(self.change_dataset)
    
    def __getitem__(self, index):
        item = self.change_dataset[index]
        # Return pre_image (A), post_image (B), and change_map (gt)
        pre_image = item['A']
        post_image = item['B']
        change_map = item['gt']
        
        # Ensure images are torch tensors
        if not isinstance(pre_image, torch.Tensor):
            pre_image = torch.from_numpy(np.ascontiguousarray(pre_image)).float()
        if not isinstance(post_image, torch.Tensor):
            post_image = torch.from_numpy(np.ascontiguousarray(post_image)).float()
        if not isinstance(change_map, torch.Tensor):
            change_map = torch.from_numpy(np.ascontiguousarray(change_map)).float()

        change_map = torch.stack([change_map] * 3, dim=-1)  # HWC format

        sample = {'image_pre': pre_image, 'image_post': post_image, 'mask': change_map}

        return sample



def create_dataset(dataset_name, args, split='train'):
    """
    Dataset factory.

    Publication / RemoteVAR scope: **change detection only**.

    Supported dataset_name values:
    - whu_cd (alias: change_dataset)
    - levircd
    - levircdplus
    - s2looking
    - levircd_union (levircd + levircdplus)
    - cd_union (union of multiple CD datasets)

    All datasets are resolved relative to `args.dataset_root` (or env `DATASET_ROOT`).
    Expected layout for each dataset root:
      <root>/{A,B,gt}/ and <root>/{train,val,test}.txt (or auto-generated from A/B/gt listing).
    """

    dataset_name = str(dataset_name)

    def _get_dataset_root() -> str:
        # Backward compatibility: allow `--data_dir` to stand in for dataset_root.
        root = getattr(args, "dataset_root", None) or os.environ.get("DATASET_ROOT")
        if not root:
            root = getattr(args, "data_dir", None)
        return str(root) if root else ""

    dataset_root = _get_dataset_root()
    if dataset_root == "":
        raise ValueError(
            "Missing dataset_root. Set `--dataset_root <PATH>` or export DATASET_ROOT, e.g.\n"
            "  export DATASET_ROOT=/path/to/datasets\n"
            "Expected per-dataset folders like:\n"
            "  $DATASET_ROOT/whu_cd, $DATASET_ROOT/levircd, $DATASET_ROOT/levircdplus, $DATASET_ROOT/s2looking"
        )

    def _ds_root(ds_name: str) -> str:
        return os.path.join(dataset_root, ds_name)

    if dataset_name in {"whu_cd", "change_dataset", "levircd", "levircdplus", "levircd_union", "cd_union", "s2looking"}:
        """
        Change-detection datasets using `datasets/change_dataset_simple.py:ChangeDataset`.

        Roots are resolved under:
          <dataset_root>/<dataset_name>
        e.g.:
          <dataset_root>/whu_cd
          <dataset_root>/levircd
          <dataset_root>/levircdplus
          <dataset_root>/s2looking
        """

        # Create augmentation preprocessing for split
        if split == 'train':
            preprocess = ChangeDetectionAugmentation(
                image_size=getattr(args, 'image_size', 256),
                enable_random_crop=getattr(args, 'enable_random_crop', True),
                enable_random_flip=getattr(args, 'enable_random_flip', True),
                enable_random_rotation=getattr(args, 'enable_random_rotation', True),
                enable_gaussian_blur=getattr(args, 'enable_gaussian_blur', True),
                enable_color_jitter=getattr(args, 'enable_color_jitter', False),
                color_jitter_probability=getattr(args, 'color_jitter_probability', 0.8),
                color_jitter_brightness=getattr(args, 'color_jitter_brightness', 0.2),
                color_jitter_contrast=getattr(args, 'color_jitter_contrast', 0.2),
                color_jitter_saturation=getattr(args, 'color_jitter_saturation', 0.2),
                color_jitter_hue=getattr(args, 'color_jitter_hue', 0.05),
                crop_scale_range=getattr(args, 'crop_scale_range', (0.8, 1.0)),
                min_crop_size=getattr(args, 'min_crop_size', 64),
                max_crop_size=getattr(args, 'max_crop_size', getattr(args, 'image_size', 256)),
                rotation_angle=getattr(args, 'rotation_angle', 10),
                blur_probability=getattr(args, 'blur_probability', 0.5),
                blur_kernel_sizes=getattr(args, 'blur_kernel_sizes', [3, 5, 7]),
                flip_probability=getattr(args, 'flip_probability', 0.5)
            )
        else:
            preprocess = ChangeDetectionAugmentation(
                image_size=getattr(args, 'image_size', 256),
                enable_random_crop=False,
                enable_random_flip=False,
                enable_random_rotation=False,
                enable_gaussian_blur=False,
                enable_color_jitter=False,
            )

        # Pass filtering parameters to dataset
        filter_empty_masks = getattr(args, 'filter_empty_masks', False)
        empty_mask_threshold = getattr(args, 'empty_mask_threshold', 0.001)

        def _resolve_root_for_split(root_dir: str, split_name: str) -> str:
            """
            Some datasets (e.g., S2Looking) store data under <root>/<split>/{A,B,gt}.
            If so, treat <root>/<split> as the effective root for ChangeDataset.
            Otherwise, use <root> as-is.
            """
            cand = os.path.join(root_dir, split_name)
            if os.path.isdir(os.path.join(cand, "A")) and os.path.isdir(os.path.join(cand, "B")) and os.path.isdir(os.path.join(cand, "gt")):
                return cand
            return root_dir

        def _rank_world():
            try:
                r = int(os.environ.get("RANK", "0"))
                w = int(os.environ.get("WORLD_SIZE", "1"))
            except Exception:
                r, w = 0, 1
            return r, w

        def _compute_rgb_mean_std(effective_root: str, split_name: str, *, image_size: int, max_samples: int, seed: int) -> Tuple[List[float], List[float], int]:
            """
            Compute RGB mean/std in [0,1] space for a change detection dataset root.
            Uses BOTH A and B images, resized to image_size, sampling up to max_samples filenames from <split>.txt.
            """
            split_path = os.path.join(effective_root, f"{split_name}.txt")
            with open(split_path, "r") as f:
                # Be robust to split files that include extensions (e.g., "train_246.png").
                # ChangeDataset._get_file_names strips extensions, so we do the same here.
                stems = []
                for ln in f.readlines():
                    s = ln.strip()
                    if not s:
                        continue
                    base, _ext = os.path.splitext(s)
                    stems.append(base if base else s)
            rng = random.Random(int(seed))
            rng.shuffle(stems)
            stems = stems[: max(1, int(max_samples))]

            sum_c = np.zeros(3, dtype=np.float64)
            sum_sq_c = np.zeros(3, dtype=np.float64)
            count = 0

            for s in stems:
                a_path = os.path.join(effective_root, "A", s + getattr(args, "A_format", ".png"))
                b_path = os.path.join(effective_root, "B", s + getattr(args, "B_format", ".png"))
                # Extra robustness: if split stem still includes an extension and the above doesn't exist, try raw.
                if not os.path.exists(a_path):
                    a_path_alt = os.path.join(effective_root, "A", s)
                    if os.path.exists(a_path_alt):
                        a_path = a_path_alt
                if not os.path.exists(b_path):
                    b_path_alt = os.path.join(effective_root, "B", s)
                    if os.path.exists(b_path_alt):
                        b_path = b_path_alt
                A = cv2.imread(a_path, cv2.IMREAD_COLOR)
                B = cv2.imread(b_path, cv2.IMREAD_COLOR)
                if A is None or B is None:
                    continue
                A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
                B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
                if int(image_size) > 0:
                    A = cv2.resize(A, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
                    B = cv2.resize(B, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
                # scale to [0,1]
                A = A.astype(np.float32) / 255.0
                B = B.astype(np.float32) / 255.0
                for img in (A, B):
                    flat = img.reshape(-1, 3)
                    sum_c += flat.sum(axis=0)
                    sum_sq_c += (flat * flat).sum(axis=0)
                    count += flat.shape[0]

            if count <= 0:
                # Signal failure explicitly; caller decides whether to cache/fallback.
                return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 0

            mean = (sum_c / count).tolist()
            var = (sum_sq_c / count - (sum_c / count) ** 2)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var).tolist()
            return mean, std, int(count)

        def _get_or_compute_rgb_stats(effective_root: str, split_name: str) -> Tuple[List[float], List[float]]:
            """
            Distributed-friendly cache of per-dataset RGB mean/std.
            """
            image_size = int(getattr(args, "image_size", 256))
            max_samples = int(getattr(args, "dataset_stats_max_samples", 500))
            seed = int(getattr(args, "dataset_stats_seed", 0))
            cache_path = os.path.join(effective_root, f"{split_name}_rgb_meanstd_sz{image_size}_n{max_samples}.json")

            rank, world = _rank_world()
            is_dist = world > 1

            def _is_invalid_cache(d: dict) -> bool:
                try:
                    mean = d.get("mean", None)
                    std = d.get("std", None)
                    count = d.get("count", None)
                    if count is not None and int(count) == 0:
                        return True
                    # Old caches might not have count; treat exact 0.5/0.5 as invalid sentinel.
                    if count is None and mean == [0.5, 0.5, 0.5] and std == [0.5, 0.5, 0.5]:
                        return True
                except Exception:
                    return True
                return False

            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    data = json.load(f)
                if not _is_invalid_cache(data):
                    return data["mean"], data["std"]
                # Invalid cache (likely created when image reads failed) -> recompute below.

            if is_dist and rank != 0:
                # wait for rank0 to write cache
                t0 = time.time()
                timeout_s = 60 * 30
                while not os.path.exists(cache_path) and (time.time() - t0) < timeout_s:
                    time.sleep(1.0)
                if os.path.exists(cache_path):
                    with open(cache_path, "r") as f:
                        data = json.load(f)
                    if not _is_invalid_cache(data):
                        return data["mean"], data["std"]
                    # else: fall through to local compute
                # fallback: compute locally

            mean, std, count = _compute_rgb_mean_std(
                effective_root,
                split_name,
                image_size=image_size,
                max_samples=max_samples,
                seed=seed,
            )
            if rank == 0:
                try:
                    if int(count) > 0:
                        tmp = cache_path + f".tmp_rank{rank}"
                        with open(tmp, "w") as f:
                            json.dump({"mean": mean, "std": std, "count": int(count)}, f)
                        os.replace(tmp, cache_path)
                        print(f"[datasets/build.py] Wrote dataset RGB stats cache: {cache_path} (mean={mean}, std={std}, count={count})")
                    else:
                        print(
                            f"[datasets/build.py] WARNING: could not compute dataset RGB stats for {effective_root} "
                            f"(count=0). Not writing cache; falling back to mean=std=0.5."
                        )
                except Exception as e:
                    print(f"[datasets/build.py] WARNING: failed to write stats cache to {cache_path}: {e}")
            return mean, std

        def _ensure_split_file_from_listing(root_dir: str, split_name: str) -> None:
            """
            Create <root>/<split>.txt by listing filenames under A/B/gt and taking the intersection.
            This is needed for datasets that don't ship split txts (e.g., S2Looking).
            """
            path = os.path.join(root_dir, f"{split_name}.txt")
            if os.path.exists(path):
                return
            a_dir = os.path.join(root_dir, "A")
            b_dir = os.path.join(root_dir, "B")
            g_dir = os.path.join(root_dir, "gt")
            if not (os.path.isdir(a_dir) and os.path.isdir(b_dir) and os.path.isdir(g_dir)):
                return

            def _stems(d: str) -> Set[str]:
                out: Set[str] = set()
                for fn in os.listdir(d):
                    # skip hidden / non-files
                    if fn.startswith("."):
                        continue
                    p = os.path.join(d, fn)
                    if not os.path.isfile(p):
                        continue
                    base, ext = os.path.splitext(fn)
                    if base:
                        out.add(base)
                return out

            stems = _stems(a_dir) & _stems(b_dir) & _stems(g_dir)
            if len(stems) == 0:
                raise FileNotFoundError(
                    f"Could not create {path}: no overlapping filenames across A/B/gt under root={root_dir}"
                )
            os.makedirs(root_dir, exist_ok=True)
            with open(path, "w") as f:
                for s in sorted(stems):
                    f.write(s + "\n")
            print(f"[datasets/build.py] Wrote split file: {path} ({len(stems)} items)")

        def _make_cd_dataset(root_dir: str, dataset_tag: Optional[str] = None):
            effective_root = _resolve_root_for_split(root_dir, split)
            _ensure_split_file_from_listing(effective_root, split)
            image_norm = str(getattr(args, "image_normalization", "m11"))
            mean = std = None
            if image_norm == "dataset":
                mean, std = _get_or_compute_rgb_stats(effective_root, "train" if split == "train" else "train")
            data_setting = {
                'root': effective_root,
                'A_format': getattr(args, 'A_format', '.png'),
                'B_format': getattr(args, 'B_format', '.png'),
                'gt_format': getattr(args, 'gt_format', '.png'),
                # align with user's requested label names
                'class_names': getattr(args, 'class_names', ['non-change', 'building']),
            }

            return ChangeDataset(
                data_setting,
                split,
                preprocess=preprocess,
                filter_empty_masks=filter_empty_masks,
                empty_mask_threshold=empty_mask_threshold,
                mask_rgb_by_location=getattr(args, 'mask_rgb_by_location', False),
                mask_rgb_grid_size=getattr(args, 'mask_rgb_grid_size', 11),
                mask_rgb_index_mode=getattr(args, 'mask_rgb_index_mode', "grid"),
                mask_rgb_levels=getattr(args, 'mask_rgb_levels', None),
                image_size=getattr(args, 'image_size', 256),
                image_normalization=image_norm,
                image_mean=mean,
                image_std=std,
                # If you ever want explicit multi-crop expansion, you can set these via args;
                # by default we rely on random crop->resize augmentation for all datasets (including LEVIR-CD+).
                expand_factor=int(getattr(args, "expand_factor", 1)),
                expand_mode=str(getattr(args, "expand_mode", "none")),
            )

        def _has_split_file(root_dir: str, split_name: str) -> bool:
            effective_root = _resolve_root_for_split(root_dir, split_name)
            return os.path.exists(os.path.join(effective_root, f"{split_name}.txt"))

        if dataset_name in {"whu_cd", "change_dataset"}:
            dataset = _make_cd_dataset(_ds_root("whu_cd"), dataset_tag="whu_cd")
        elif dataset_name == "levircd":
            dataset = _make_cd_dataset(_ds_root("levircd"), dataset_tag="levircd")
        elif dataset_name == "levircdplus":
            dataset = _make_cd_dataset(_ds_root("levircdplus"), dataset_tag="levircdplus")
        elif dataset_name == "s2looking":
            dataset = _make_cd_dataset(_ds_root("s2looking"), dataset_tag="s2looking")
        elif dataset_name == "levircd_union":
            # Allow overriding via args.data_dirs=[...], otherwise use dataset_root defaults.
            data_dirs = getattr(args, "data_dirs", None) or [_ds_root("levircd"), _ds_root("levircdplus")]
            if split != "train":
                # drop roots without this split (e.g., some datasets may have train-only)
                data_dirs = [d for d in data_dirs if _has_split_file(d, split)]
                if len(data_dirs) == 0:
                    raise FileNotFoundError(f"No '{split}.txt' found for levircd_union roots under dataset_root={dataset_root}")
            dataset = ConcatDataset([_make_cd_dataset(d) for d in data_dirs])
        elif dataset_name == "cd_union":
            data_dirs = getattr(args, "data_dirs", None)
            if not data_dirs:
                cd_datasets = getattr(args, "cd_union_datasets", ["whu_cd", "levircd", "levircdplus", "s2looking"])
                data_dirs = [_ds_root(str(ds)) for ds in cd_datasets]
            if split != "train":
                # drop roots without this split (e.g., LEVIR-CD+ has no val/test)
                before = list(data_dirs)
                data_dirs = [d for d in data_dirs if _has_split_file(d, split)]
                dropped = [d for d in before if d not in data_dirs]
                if dropped:
                    print(f"[datasets/build.py] cd_union: skipping roots without {split}.txt: {dropped}")
                if len(data_dirs) == 0:
                    raise FileNotFoundError(f"No '{split}.txt' found for any cd_union roots.")
            # keep per-dataset normalization by constructing separate ChangeDataset objects per root
            datasets_list = [_make_cd_dataset(d) for d in data_dirs]
            dataset = ConcatDataset(datasets_list)
        else:
            raise ValueError(f"Unsupported dataset_name='{dataset_name}'.")

    else:
        raise ValueError(
            f"Unsupported dataset_name='{dataset_name}' for RemoteVAR publication scope. "
            f"Supported: whu_cd, levircd, levircdplus, s2looking, levircd_union, cd_union."
        )

    return dataset
        
        