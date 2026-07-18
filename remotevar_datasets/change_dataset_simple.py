import os
from pickletools import uint8
import cv2
import torch
import numpy as np
from torch.utils import data
import random
import time

import warnings
import torch.nn as nn
import torch.nn.functional as F

from .utils import binary_mask_to_rgb_by_location, create_entity_like_color_map, auto_color_levels_for_required_colors
from typing import Union, Tuple

import numbers
import random
import collections



class ChangeDataset(data.Dataset):
    def __init__(
        self,
        setting,
        split_name,
        preprocess=None,
        filter_empty_masks=False,
        empty_mask_threshold=0.001,
        mask_rgb_by_location: bool = False,
        mask_rgb_grid_size: Union[int, Tuple[int, int]] = 11,
        mask_rgb_index_mode: str = "grid",
        mask_rgb_levels: Union[int, Tuple[int, ...], None] = None,
        image_size: int = 256,
        image_normalization: str = "m11",
        image_mean: Union[None, Tuple[float, float, float]] = None,
        image_std: Union[None, Tuple[float, float, float]] = None,
        expand_factor: int = 1,
        expand_mode: str = "none",
    ):
        super(ChangeDataset, self).__init__()
        self._split_name = split_name
        self._A_format = setting['A_format']
        self._B_format = setting['B_format']
        self._gt_format = setting['gt_format']
        self._root_path = setting['root']
        self.class_names = setting['class_names']
        self._file_names = self._get_file_names(split_name)
        self.preprocess = preprocess
        self.filter_empty_masks = filter_empty_masks
        self.empty_mask_threshold = empty_mask_threshold
        self.image_size = int(image_size)
        self.expand_factor = max(1, int(expand_factor))
        self.expand_mode = str(expand_mode)
        self._index_map = None  # optional list[(base_index, crop_index)] when expand_factor>1 and/or tile-level filtering

        # Optional: convert binary mask to RGB (location-coded) for richer supervision/conditioning.
        self.mask_rgb_by_location = bool(mask_rgb_by_location)
        self.mask_rgb_grid_size = mask_rgb_grid_size
        self.mask_rgb_index_mode = mask_rgb_index_mode
        self.mask_rgb_levels = mask_rgb_levels
        self._mask_rgb_colormap = None
        if self.mask_rgb_by_location:
            # Determine how many unique bins we need to represent.
            if isinstance(self.mask_rgb_grid_size, int):
                gy = gx = int(self.mask_rgb_grid_size)
            else:
                gy, gx = int(self.mask_rgb_grid_size[0]), int(self.mask_rgb_grid_size[1])
            gy = max(1, gy)
            gx = max(1, gx)
            required = gy * gx if str(self.mask_rgb_index_mode) == "grid" else gy * gx

            # Allow override: mask_rgb_levels can be:
            # - None or 0: auto (minimum L such that L^3-1 >= required)
            # - int >= 2: use exactly L levels
            # - tuple/list of ints: use explicit levels
            levels = None
            if self.mask_rgb_levels is None or self.mask_rgb_levels == 0:
                levels = auto_color_levels_for_required_colors(required, drop_black=True)
            elif isinstance(self.mask_rgb_levels, int):
                L = max(2, int(self.mask_rgb_levels))
                import numpy as _np
                lv = _np.round(_np.linspace(0, 255, L)).astype(_np.int32)
                lv[0] = 0
                lv[-1] = 255
                levels = tuple(int(x) for x in _np.unique(lv).tolist())
            else:
                levels = tuple(int(x) for x in self.mask_rgb_levels)

            self._mask_rgb_colormap = create_entity_like_color_map(levels=levels, drop_black=True)

        self.image_normalization = str(image_normalization)
        self.image_mean = tuple(image_mean) if image_mean is not None else None
        self.image_std = tuple(image_std) if image_std is not None else None
        if self.image_normalization not in {"m11", "dataset"}:
            raise ValueError(f"Unknown image_normalization={self.image_normalization}. Use 'm11' or 'dataset'.")
        if self.image_normalization == "dataset":
            if self.image_mean is None or self.image_std is None:
                raise ValueError("image_normalization='dataset' requires image_mean and image_std.")
        
        # Filter out empty masks if requested (only for training split)
        if self.filter_empty_masks and split_name == 'train':
            self._filter_empty_masks()
        # Build expanded index map (also supports tile-level filtering when enabled)
        self._rebuild_index_map()
    
    def _filter_empty_masks(self):
        """Filter out samples with empty (all-black) masks"""
        from tqdm import tqdm
        # Cache the filtered file list so we don't re-filter on every GPU/process.
        # In distributed training, only rank0 performs filtering; other ranks wait for the cache file.
        cache_path = os.path.join(
            self._root_path,
            f"{self._split_name}_filtered_thr{self.empty_mask_threshold}_sz{self.image_size}.txt",
        )

        def _rank_world():
            try:
                r = int(os.environ.get("RANK", "0"))
                w = int(os.environ.get("WORLD_SIZE", "1"))
            except Exception:
                r, w = 0, 1
            return r, w

        rank, world = _rank_world()
        is_dist = world > 1

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self._file_names = [ln.strip() for ln in f.readlines() if ln.strip()]
            if rank == 0:
                print(f"[ChangeDataset] Loaded filtered list from cache: {cache_path} ({len(self._file_names)} items)")
            return

        if is_dist and rank != 0:
            # Wait for rank0 to create cache.
            t0 = time.time()
            timeout_s = 60 * 30  # 30 minutes
            while not os.path.exists(cache_path):
                if time.time() - t0 > timeout_s:
                    break
                time.sleep(1.0)
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    self._file_names = [ln.strip() for ln in f.readlines() if ln.strip()]
                return
            # Fallback: if rank0 didn't produce it, continue to compute locally (should be rare).

        if rank == 0:
            print(f"Filtering empty masks from {self._split_name} split...")
        
        valid_file_names = []
        removed_count = 0
        
        it = self._file_names
        if rank == 0:
            it = tqdm(it, desc="Filtering empty masks")
        for file_name in it:
            gt_path = os.path.join(self._root_path, 'gt', file_name + self._gt_format)
            
            # Load mask
            gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)

            # IMPORTANT: Filtering must be deterministic and consistent across datasets in a union.
            # Do NOT run `self.preprocess` here because training preprocess may be random (crop/flip/rot/blur).
            # Instead, mimic the invariant part of __getitem__:
            #   - resize GT to target resolution (nearest)
            #   - binarize and compute foreground ratio
            if self.image_size is not None and gt is not None:
                if gt.shape[:2] != (self.image_size, self.image_size):
                    gt = cv2.resize(gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

            # Robust binarization for various GT encodings (0/1, 0/255, etc.)
            gt_binary = (gt > 0).astype(np.uint8)  # 0/1
            non_black_ratio = float(gt_binary.mean()) if gt_binary.size > 0 else 0.0
            
            # Keep masks with sufficient non-black pixels
            if non_black_ratio > self.empty_mask_threshold:
                valid_file_names.append(file_name)
            else:
                removed_count += 1
        
        original_size = len(self._file_names)
        self._file_names = valid_file_names
        filtered_size = len(self._file_names)
        
        # Persist cache (atomic write) so other ranks (and future runs) can reuse it.
        try:
            tmp = cache_path + f".tmp_rank{rank}"
            with open(tmp, "w") as f:
                for n in self._file_names:
                    f.write(str(n) + "\n")
            os.replace(tmp, cache_path)
        except Exception as e:
            if rank == 0:
                print(f"[ChangeDataset] Warning: failed to write filtered cache to {cache_path}: {e}")

        if rank == 0:
            print(f"Dataset filtering complete:")
            print(f"  Original size: {original_size}")
            print(f"  Filtered size: {filtered_size}")
            print(f"  Removed: {removed_count} ({removed_count/original_size*100:.1f}%)")

    def _rebuild_index_map(self):
        """
        Build an explicit index mapping.
        - If expand_factor == 1: no mapping needed.
        - If expand_factor > 1:
            - If filtering enabled (train split): filter at TILE level too (e.g., drop empty LEVIR-CD+ quadrants).
            - Else: include all tiles for each file.
        """
        if self.expand_factor <= 1:
            self._index_map = None
            return

        index_map = []
        do_tile_filter = bool(self.filter_empty_masks and self._split_name == "train")

        for base_i, file_name in enumerate(self._file_names):
            if not do_tile_filter:
                for crop_i in range(self.expand_factor):
                    index_map.append((base_i, crop_i))
                continue

            # Deterministic tile-level filtering: load GT once, crop tiles, resize to image_size, and test fg ratio.
            gt_path = os.path.join(self._root_path, 'gt', file_name + self._gt_format)
            gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
            if gt is None:
                continue

            h0, w0 = gt.shape[:2]
            for crop_i in range(self.expand_factor):
                gt_tile = gt
                if self.expand_mode == "tile2x2" and self.expand_factor == 4 and h0 >= 2 and w0 >= 2:
                    crop_h = h0 // 2
                    crop_w = w0 // 2
                    r = crop_i // 2
                    c = crop_i % 2
                    y0 = r * crop_h
                    x0 = c * crop_w
                    gt_tile = gt[y0:y0 + crop_h, x0:x0 + crop_w]
                else:
                    # Same deterministic fallback as __getitem__
                    import random as _random
                    rng = _random.Random(hash((file_name, crop_i)) & 0xFFFFFFFF)
                    crop_h = max(1, h0 // 2)
                    crop_w = max(1, w0 // 2)
                    y0 = 0 if h0 <= crop_h else rng.randint(0, h0 - crop_h)
                    x0 = 0 if w0 <= crop_w else rng.randint(0, w0 - crop_w)
                    gt_tile = gt[y0:y0 + crop_h, x0:x0 + crop_w]

                if self.image_size is not None and gt_tile.shape[:2] != (self.image_size, self.image_size):
                    gt_tile = cv2.resize(gt_tile, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

                gt_bin = (gt_tile > 0).astype(np.uint8)
                fg_ratio = float(gt_bin.mean()) if gt_bin.size > 0 else 0.0
                if fg_ratio > self.empty_mask_threshold:
                    index_map.append((base_i, crop_i))

        self._index_map = index_map

    def __len__(self):
        if self._index_map is not None:
            return len(self._index_map)
        return len(self._file_names) * self.expand_factor

    def __getitem__(self, index):
        if self._index_map is not None:
            base_index, crop_index = self._index_map[index]
        else:
            base_index = index // self.expand_factor
            crop_index = index % self.expand_factor
        item_name = self._file_names[base_index]
        A_path = os.path.join(self._root_path, 'A', item_name + self._A_format)
        B_path = os.path.join(self._root_path, 'B', item_name + self._B_format)
        gt_path = os.path.join(self._root_path, 'gt', item_name + self._gt_format)

        # Load images as RGB uint8
        A = self._open_image(A_path, cv2.IMREAD_COLOR, dtype=np.uint8)
        B = self._open_image(B_path, cv2.IMREAD_COLOR, dtype=np.uint8)
        if A is None or B is None:
            raise FileNotFoundError(f"Failed to read A or B: {A_path}, {B_path}")
        A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
        B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if gt is None:
            raise FileNotFoundError(f"Failed to read gt: {gt_path}")

        if self.preprocess is not None:
            A, B, gt = self.preprocess(A, B, gt)

        # Optional expansion for large images (e.g., LEVIR-CD+ 1024x1024):
        # turn each source image into multiple deterministic crops before final resize.
        if self.expand_factor > 1:
            h0, w0 = gt.shape[:2]
            if self.expand_mode == "tile2x2" and self.expand_factor == 4 and h0 >= 2 and w0 >= 2:
                crop_h = h0 // 2
                crop_w = w0 // 2
                r = crop_index // 2
                c = crop_index % 2
                y0 = r * crop_h
                x0 = c * crop_w
                A = A[y0:y0 + crop_h, x0:x0 + crop_w]
                B = B[y0:y0 + crop_h, x0:x0 + crop_w]
                gt = gt[y0:y0 + crop_h, x0:x0 + crop_w]
            else:
                # Fallback: random crop with deterministic seed per (item_name, crop_index)
                import random as _random
                rng = _random.Random(hash((item_name, crop_index)) & 0xFFFFFFFF)
                crop_h = max(1, h0 // 2)
                crop_w = max(1, w0 // 2)
                y0 = 0 if h0 <= crop_h else rng.randint(0, h0 - crop_h)
                x0 = 0 if w0 <= crop_w else rng.randint(0, w0 - crop_w)
                A = A[y0:y0 + crop_h, x0:x0 + crop_w]
                B = B[y0:y0 + crop_h, x0:x0 + crop_w]
                gt = gt[y0:y0 + crop_h, x0:x0 + crop_w]

        # Enforce fixed spatial resolution for union datasets (and general robustness)
        if self.image_size is not None:
            target = (self.image_size, self.image_size)
            if A.shape[:2] != target:
                A = cv2.resize(A, (target[1], target[0]), interpolation=cv2.INTER_LINEAR)
            if B.shape[:2] != target:
                B = cv2.resize(B, (target[1], target[0]), interpolation=cv2.INTER_LINEAR)
            if gt.shape[:2] != target:
                gt = cv2.resize(gt, (target[1], target[0]), interpolation=cv2.INTER_NEAREST)

        # Ensure dtypes are sane
        if A.dtype != np.uint8:
            A = np.clip(A, 0, 255).astype(np.uint8)
        if B.dtype != np.uint8:
            B = np.clip(B, 0, 255).astype(np.uint8)
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)

        # Convert images to CHW float in [0,1]
        A_tensor = torch.from_numpy(A).permute(2, 0, 1).contiguous().float().div_(255.0)
        B_tensor = torch.from_numpy(B).permute(2, 0, 1).contiguous().float().div_(255.0)

        # Normalize pre/post and clamp to [-1,1] before feeding to VQVAE/VAR.
        # - "m11": (x - 0.5) * 2  (legacy behavior)
        # - "dataset": (x - mean) / std, then clamp to [-1,1]
        if self.image_normalization == "m11":
            A_tensor = (A_tensor - 0.5) * 2.0
            B_tensor = (B_tensor - 0.5) * 2.0
        else:
            mean = torch.tensor(self.image_mean, dtype=A_tensor.dtype, device=A_tensor.device).view(3, 1, 1)
            std = torch.tensor(self.image_std, dtype=A_tensor.dtype, device=A_tensor.device).view(3, 1, 1)
            std = std.clamp_min(1e-6)
            A_tensor = (A_tensor - mean) / std
            B_tensor = (B_tensor - mean) / std

        A_tensor = A_tensor.clamp(-1.0, 1.0)
        B_tensor = B_tensor.clamp(-1.0, 1.0)

        # IMPORTANT: always binarize the raw GT first to prevent intermediate gray values.
        gt_bin = (gt > 127.5).astype(np.uint8)  # 0/1

        # Optionally: colorize foreground pixels by location (RGB mask)
        if self.mask_rgb_by_location:
            gt_rgb = binary_mask_to_rgb_by_location(
                gt_bin,
                colormap=self._mask_rgb_colormap,
                grid_size=self.mask_rgb_grid_size,
                index_mode=self.mask_rgb_index_mode,
            )  # (H,W,3) uint8
            gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).contiguous().float()
            gt_tensor = (gt_tensor / 255.0 - 0.5) * 2  # [-1, 1]
            gt_tensor = gt_tensor.clamp(-1.0, 1.0)
        else:
            gt_tensor_1 = torch.from_numpy(gt_bin.astype(np.float32) * 255.0).contiguous()
            gt_tensor_1 = (gt_tensor_1 / 255.0 - 0.5) * 2  # either -1 or 1
            gt_tensor = torch.stack([gt_tensor_1] * 3, dim=0)
            gt_tensor = gt_tensor.clamp(-1.0, 1.0)

        output_dict = dict(images_pre=A_tensor, images_post=B_tensor, mask=gt_tensor, cls=0, type=0, fn=item_name)

        return output_dict

    def foreground_ratio(self, index: int, *, pixel_thr: int = 0) -> float:
        """
        Fast foreground-area estimate for visualization/index selection.
        Reads ONLY the GT mask from disk (no A/B), resizes to image_size, binarizes, and returns fg ratio.

        Args:
            index: dataset index (supports expanded index_map if present)
            pixel_thr: pixel threshold on raw GT (0/255 masks supported). Foreground if gt > pixel_thr.

        Returns:
            Foreground ratio in [0,1].
        """
        if self._index_map is not None:
            base_index, _ = self._index_map[index]
        else:
            base_index = int(index) // self.expand_factor
        item_name = self._file_names[base_index]
        gt_path = os.path.join(self._root_path, 'gt', item_name + self._gt_format)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if gt is None:
            return 0.0
        if self.image_size is not None and gt.shape[:2] != (self.image_size, self.image_size):
            gt = cv2.resize(gt, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > int(pixel_thr)).astype(np.uint8)
        return float(gt_bin.mean()) if gt_bin.size > 0 else 0.0

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        source = os.path.join(self._root_path, split_name+'.txt')

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            file_name = item.strip()
            # Skip empty lines (can happen if a split file has trailing newlines)
            if not file_name:
                continue
            # Robustly drop extension if present (e.g., "123.png" -> "123")
            base, _ext = os.path.splitext(file_name)
            file_names.append(base if base else file_name)

        return file_names

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)
        return img

    @staticmethod
    def _gt_transform(gt):
        return gt - 1 

    @classmethod
    def get_class_colors(*args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 41
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors






if __name__ == "__main__" and os.environ.get("RUN_DATASET_DEMO") == "1":
    data_setting = {'root': os.path.join(os.environ.get("DATASET_ROOT", "data"), "whu_cd"),
                    'A_format': '.png',
                    'B_format': '.png',
                    'gt_format': '.png',
                    'class_names': ['Background', 'Change']}
    
    dataset = ChangeDataset(data_setting, 'val')
    item = dataset[15]

    print('A_img shape: ', item['A'].shape, ' B_img shape: ', item['B'].shape, 'gt shape: ', item['gt'].shape, 'filename: ', item['fn'], ' len dataset: ', item['n'])
    cv2.imwrite('tmpA.png', item['A'])
    cv2.imwrite('tmpB.png', item['B'])
    cv2.imwrite('tmp_gt.png', item['gt'])