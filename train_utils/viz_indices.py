import json
import os
import random
from typing import List, Tuple

from .dataset_id import dataset_id_for_run


def select_viz_indices(
    dataset,
    *,
    k: int,
    pixel_thr_01: float = 0.2,
    area_thr: float = 0.2,
    target_ratio: float = 0.5,
    fallback_target_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[List[int], List[float]]:
    """
    Deterministically pick `k` indices with GT foreground ratios close to `target_ratio`,
    preferring those with ratio > area_thr.
    """
    n = len(dataset)
    if n == 0:
        return [], []

    ratios = []
    for i in range(n):
        r = None
        if hasattr(dataset, "foreground_ratio"):
            r = float(dataset.foreground_ratio(i, pixel_thr=0))
        else:
            sample = dataset[i]
            m = sample["mask"]
            m01 = (m + 1) / 2
            fg = (m01.max(dim=0).values > float(pixel_thr_01)).float().mean().item()
            r = float(fg)
        ratios.append(r)

    idxs = list(range(n))
    rnd = random.Random(int(seed))
    rnd.shuffle(idxs)

    above = [i for i in idxs if ratios[i] > float(area_thr)]
    above.sort(key=lambda i: abs(ratios[i] - float(target_ratio)))
    chosen = list(above[:k])

    if len(chosen) < k:
        remaining = [i for i in above if i not in set(chosen)]
        remaining.sort(key=lambda i: abs(ratios[i] - float(fallback_target_ratio)))
        need = k - len(chosen)
        chosen.extend(remaining[:need])

    if len(chosen) < k:
        remaining_all = [i for i in idxs if i not in set(chosen)]
        remaining_all.sort(key=lambda i: abs(ratios[i] - float(target_ratio)))
        need = k - len(chosen)
        chosen.extend(remaining_all[:need])

    chosen = chosen[:k]
    chosen_ratios = [ratios[i] for i in chosen]
    return chosen, chosen_ratios


def load_or_create_viz_indices(
    *,
    args,
    dataset,
    split_name: str,
    cache_dir: str,
    k: int = 4,
    pixel_thr_01: float = 0.2,
    area_thr: float = 0.2,
    seed: int = 0,
    target_ratio: float = 0.5,
    fallback_target_ratio: float = 0.2,
) -> List[int]:
    os.makedirs(cache_dir, exist_ok=True)
    dsid = dataset_id_for_run(args)
    path = os.path.join(
        cache_dir,
        f"{dsid}__split={split_name}__k={k}__pix={pixel_thr_01}__area={area_thr}"
        f"__t={target_ratio}__fb={fallback_target_ratio}.json",
    )

    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            idxs = [int(x) for x in data.get("indices", [])]
            idxs = [i for i in idxs if 0 <= i < len(dataset)]
            if len(idxs) > 0:
                print(f"[viz_indices] Loaded cached {split_name} indices from: {os.path.basename(path)}")
                return idxs[:k]
        except Exception:
            pass

    print(f"[viz_indices] Generating NEW {split_name} indices (dataset_len={len(dataset)})...")
    idxs, ratios = select_viz_indices(
        dataset,
        k=k,
        pixel_thr_01=pixel_thr_01,
        area_thr=area_thr,
        target_ratio=target_ratio,
        fallback_target_ratio=fallback_target_ratio,
        seed=seed,
    )
    payload = {
        "dataset_id": dsid,
        "dataset_name": getattr(args, "dataset_name", None),
        "split": split_name,
        "k": k,
        "pixel_thr_01": pixel_thr_01,
        "area_thr": area_thr,
        "target_ratio": target_ratio,
        "fallback_target_ratio": fallback_target_ratio,
        "seed": seed,
        "dataset_len": int(len(dataset)),
        "indices": idxs,
        "fg_ratios": ratios,
    }
    print(f"[viz_indices] Saved NEW {split_name} cache to: {os.path.basename(path)}")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return idxs

