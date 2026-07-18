import cv2
import numpy as np
from PIL import Image
import torch, io
try:
    import torchdata.datapipes as dps  # optional (webdataset datapipes)
except Exception:  # pragma: no cover
    dps = None
try:
    from braceexpand import braceexpand  # optional
except Exception:  # pragma: no cover
    braceexpand = None
from .mask_color import mask_colormap
try:
    from pycocotools import mask as mask_utils  # optional
except Exception:  # pragma: no cover
    mask_utils = None
from typing import Optional, Union, Tuple
import math

def create_entity_like_color_map(levels=(0, 64, 128, 192, 255), drop_black=True) -> np.ndarray:
    """
    Create a dense RGB colormap similar to `datasets/entityS.py:create_color_map()`.

    Args:
        levels: Channel levels to use for each of R/G/B.
        drop_black: If True, drops [0,0,0] (reserved for background).

    Returns:
        np.ndarray of shape (N, 3) with dtype uint8.
    """
    color_map = []
    for r in levels:
        for g in levels:
            for b in levels:
                color_map.append([r, g, b])
    cmap = np.asarray(color_map, dtype=np.uint8)
    if drop_black:
        # drop [0,0,0] to keep index 0 non-black (useful when background is black)
        cmap = cmap[1:]
    return cmap


def auto_color_levels_for_required_colors(required_colors: int, *, drop_black: bool = True) -> Tuple[int, ...]:
    """
    Choose the minimum number of channel levels L such that:
      available_colors = L^3 - (1 if drop_black else 0) >= required_colors

    Returns a tuple of length L with values in [0,255] (inclusive), approximately evenly spaced.
    """
    req = int(required_colors)
    if req <= 0:
        return (0, 255)
    need = req + (1 if drop_black else 0)
    L = int(math.ceil(need ** (1.0 / 3.0)))
    L = max(2, L)
    levels = np.linspace(0, 255, L)
    levels = np.round(levels).astype(np.int32)
    # Ensure monotonic unique levels and exact endpoints
    levels[0] = 0
    levels[-1] = 255
    levels = np.unique(levels)
    # If rounding collapsed values, increase L until unique count >= desired L.
    while levels.size < L:
        L += 1
        lv = np.linspace(0, 255, L)
        lv = np.round(lv).astype(np.int32)
        lv[0] = 0
        lv[-1] = 255
        levels = np.unique(lv)
    return tuple(int(x) for x in levels.tolist())


def binary_mask_to_rgb_by_location(
    binary_mask: np.ndarray,
    *,
    colormap: Optional[np.ndarray] = None,
    grid_size: Union[int, Tuple[int, int]] = 11,
    index_mode: str = "grid",
) -> np.ndarray:
    """
    Convert a 2D binary mask into an RGB image where each positive pixel gets a color
    based on its (quantized) location.

    - Background (mask==0) remains black.
    - Foreground (mask>0) is colored by cell index on a `grid_size` x `grid_size` grid.

    This is inspired by `datasets/entityS.py:process_anns()` but operates pixel-wise
    (no instance annotations required).

    Args:
        binary_mask: (H, W) array; treated as foreground where > 0.
        colormap: (N, 3) uint8 colormap. If None, uses `create_entity_like_color_map()`.
        grid_size: int or (gy, gx). Quantization grid.
        index_mode:
            - "grid": idx = y_bin * gx + x_bin  (default, better coverage)
            - "mul":  idx = x_bin * y_bin       (closer to entityS style, more collisions)

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"binary_mask must be 2D (H,W). Got shape={binary_mask.shape}")

    if colormap is None:
        colormap = create_entity_like_color_map()
    colormap = np.asarray(colormap, dtype=np.uint8)
    if colormap.ndim != 2 or colormap.shape[1] != 3:
        raise ValueError(f"colormap must have shape (N,3). Got shape={colormap.shape}")
    if colormap.shape[0] < 1:
        raise ValueError("colormap must have at least 1 color")

    if isinstance(grid_size, int):
        gy, gx = grid_size, grid_size
    else:
        gy, gx = int(grid_size[0]), int(grid_size[1])
    gy = max(1, gy)
    gx = max(1, gx)

    h, w = binary_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    ys, xs = np.nonzero(binary_mask > 0)
    if ys.size == 0:
        return rgb

    # Quantize pixel coordinates to grid bins.
    # Use integer math to avoid floating issues.
    x_bin = (xs.astype(np.int64) * gx) // max(1, w)
    y_bin = (ys.astype(np.int64) * gy) // max(1, h)
    x_bin = np.clip(x_bin, 0, gx - 1)
    y_bin = np.clip(y_bin, 0, gy - 1)

    if index_mode == "mul":
        idx = x_bin * y_bin
    elif index_mode == "grid":
        idx = y_bin * gx + x_bin
    else:
        raise ValueError(f"Unknown index_mode={index_mode}. Use 'grid' or 'mul'.")

    idx = (idx % colormap.shape[0]).astype(np.int64)
    rgb[ys, xs] = colormap[idx]
    return rgb


def rgb_mask_to_binary(
    rgb_mask: np.ndarray,
    *,
    non_black_threshold: float = 0.05,
    out_values: tuple[int, int] = (0, 255),
) -> np.ndarray:
    """
    Back-convert an RGB mask into a binary mask by treating any non-black pixel as foreground.

    This is the natural inverse of `binary_mask_to_rgb_by_location()` because that converter
    never uses pure black for foreground (colormap drops [0,0,0]).

    Notes:
        - Supports uint8 RGB in [0,255], float RGB in [0,1], or float RGB in [-1,1].
        - For float inputs, `non_black_threshold` is interpreted in [0,1] space.
        - For uint8 inputs, `non_black_threshold` is interpreted in [0,255] space.

    Args:
        rgb_mask: (H,W,3) uint8/float image.
        non_black_threshold: Foreground if max(channel) > threshold (after mapping to display space).
        out_values: Output values for (background, foreground).

    Returns:
        (H,W) uint8 image with values in {out_values[0], out_values[1]}.
    """
    if rgb_mask.ndim != 3 or rgb_mask.shape[2] != 3:
        raise ValueError(f"rgb_mask must have shape (H,W,3). Got {rgb_mask.shape}")
    m = np.asarray(rgb_mask)

    # Map to display space for thresholding:
    # - uint8: keep [0,255]
    # - float [0,1]: keep
    # - float [-1,1]: map to [0,1]
    if np.issubdtype(m.dtype, np.floating):
        mf = m.astype(np.float32)
        if mf.min() < -0.01:  # likely [-1,1]
            mf = (mf + 1.0) / 2.0
        mf = np.clip(mf, 0.0, 1.0)
        thr = float(non_black_threshold)
        non_black = np.max(mf, axis=-1) > thr
    else:
        mu = m.astype(np.uint8)
        thr_u8 = int(round(non_black_threshold))
        non_black = np.max(mu, axis=-1) > thr_u8

    out = np.empty(m.shape[:2], dtype=np.uint8)
    out[non_black] = np.uint8(out_values[1])
    out[~non_black] = np.uint8(out_values[0])
    return out


def rgb_mask_to_grayscale(rgb_mask: np.ndarray, *, non_black_threshold: int = 0) -> np.ndarray:
    """
    Alias for `rgb_mask_to_binary(..., out_values=(0,255))` for visualization convenience.
    """
    return rgb_mask_to_binary(rgb_mask, non_black_threshold=non_black_threshold, out_values=(0, 255))

def calculate_centroid_poly(polygons):
    """
    Calculate the centroid from multiple polygons.
    """
    total_x, total_y, total_points = 0, 0, 0
    for polygon in polygons:
        x_coordinates, y_coordinates = zip(*polygon)
        total_x += sum(x_coordinates)
        total_y += sum(y_coordinates)
        total_points += len(polygon)

    centroid_x = total_x / total_points
    centroid_y = total_y / total_points
    return centroid_x, centroid_y


def sort_annotations_by_centerness(annotations):
    """
    Sort annotations by the "centerness" of their masks considering all polygons.
    """
    centroids = []
    for ann in annotations:
        if 'segmentation' in ann:
            all_polygons = []
            for segment in ann['segmentation']:
                # Convert segmentation format to list of (x, y) tuples for each polygon
                polygon = [(segment[i], segment[i + 1]) for i in range(0, len(segment), 2)]
                all_polygons.append(polygon)
            if all_polygons:
                centroid = calculate_centroid_poly(all_polygons)
                centroids.append((centroid, ann))

    # Sort by y (descending) and then x (ascending)
    sorted_annotations = [ann for _, ann in sorted(centroids, key=lambda x: (-x[0][1], x[0][0]))]
    return sorted_annotations

def tensor_encoder(obj):
    """Custom encoder for PyTorch tensors."""
    if isinstance(obj, torch.Tensor):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        return buffer.getvalue()  # Return tensor serialized as bytes
    return obj  # Fallback for types this encoder doesn't handle


def decode_pkl(item):
    key, value = item
    value_file_obj = value.file_obj
    if key == '__key__':
        return key, value
    elif key.endswith('.code'):
        return key, torch.load(value_file_obj, map_location='cpu')
    elif key.endswith('.txt'):
        return key, {'text': value_file_obj.read().decode('utf-8')}
    else:
        return key, value_file_obj.read().decode('utf-8')


def unwarp_data(item):
    unwarpped = {}
    for key, value in item.items():
        if isinstance(value, dict):
            unwarpped.update(value)
        elif value is not None:
            unwarpped[key] = value
    if '__key__' in unwarpped and '/' in unwarpped['__key__']:
        unwarpped['__key__'] = unwarpped['__key__'].split('/')[-1]
    return unwarpped


def build_datapipe(data_dir,
                   masks='*.tar',
                   decode_fn=None,
                   max_length=1024,
                   reverse_ratio=0.5,
                   cycle_count=None,
                   batch_size=None,
                   shuffle=True,
                   recursive=True,
                   non_deterministic=False):
    if dps is None:
        raise ImportError("torchdata is required for build_datapipe(). Install `torchdata` to use this feature.")
    if braceexpand is None:
        raise ImportError("braceexpand is required for build_datapipe(). Install `braceexpand` to use this feature.")
    if isinstance(data_dir, str):
        data_dir = list(braceexpand(data_dir))

    datapipe = dps.iter.FileLister(data_dir, masks=masks, recursive=recursive, non_deterministic=non_deterministic)
    # cycle time of the dataset
    datapipe = datapipe.cycle(count=cycle_count)
    # shuffle
    if shuffle:
        datapipe = datapipe.shuffle()
    datapipe = datapipe.sharding_filter()
    datapipe = datapipe.open_files(mode='b')
    # TODO: use this according to decode_fn
    datapipe = datapipe.load_from_tar()

    # data processing and decoding map
    if decode_fn is not None:
        datapipe = datapipe.map(decode_fn)
    # streaming as dict
    datapipe = datapipe.webdataset()
    # unwrap the data
    datapipe = datapipe.map(unwarp_data)
    # filter data if necessary
    # datapipe = datapipe.filter(filter_data_for_llm)

    # shuffle with buffer size
    if shuffle:
        datapipe = datapipe.shuffle(buffer_size=4096)

    # batch data
    if batch_size is not None:
        datapipe = datapipe.batch(batch_size)
        datapipe = datapipe.collate()
    return datapipe


def calculate_centroid(label_image, label):
    """Calculate the centroid of a single labeled component."""
    rows, cols = np.where(label_image == label)
    if len(rows) == 0:
        return None
    centroid_x = np.mean(cols)
    centroid_y = np.mean(rows)
    return centroid_x, centroid_y


def semantic_to_instance_map(semantic_map_path):
    # Load the semantic map image
    semantic_map = np.array(Image.open(semantic_map_path))

    # Create the category mask: every non-black pixel is part of the category
    category_mask = np.any(semantic_map != [0, 0, 0], axis=-1).astype(np.uint8) * 255

    # Find connected components (individual instances) within the category mask
    num_labels, labels_im = cv2.connectedComponents(category_mask)

    # Collect centroids for sorting
    centroids = []
    for label in range(1, num_labels):  # Skip the background
        centroid = calculate_centroid(labels_im, label)
        if centroid:
            centroids.append((label, centroid))

    # Sort centroids based on their distance from the bottom-right corner
    # The image's bottom-right corner is at (max_x, max_y)
    max_x, max_y = labels_im.shape[1], labels_im.shape[0]
    centroids.sort(key=lambda x: -(x[1][0] + x[1][1]))  # Sort by the sum of x and y coordinates

    # Create a visual representation (optional)
    instance_map_visual = np.zeros(semantic_map.shape, dtype=np.uint8)
    for idx, (label, _) in enumerate(centroids, start=1):
        color = mask_colormap[idx]
        instance_map_visual[labels_im == label] = color

    # Convert the visual instance map to a PIL image and save

    # print(f"Found {num_labels - 1} instances, sorted")
    return Image.fromarray(instance_map_visual)