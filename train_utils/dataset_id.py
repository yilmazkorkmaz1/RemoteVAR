import os
from typing import List, Optional


def infer_roots_for_run(args) -> Optional[List[str]]:
    """
    Build a deterministic list of dataset roots for stable cache keys.

    This MUST match how datasets are resolved in `datasets/build.py`:
    - if `args.data_dirs` is explicitly set, use it directly
    - otherwise resolve via `dataset_root/<dataset_name>`
    """
    dd = getattr(args, "data_dirs", None)
    if isinstance(dd, (list, tuple)) and len(dd) > 0:
        return list(dd)

    root = getattr(args, "dataset_root", None) or os.environ.get("DATASET_ROOT") or getattr(args, "data_dir", None)
    root = str(root) if root else ""
    if not root:
        return None

    ds = str(getattr(args, "dataset_name", ""))
    if ds in {"whu_cd", "change_dataset"}:
        return [os.path.join(root, "whu_cd")]
    if ds == "cd_union":
        cd_datasets = getattr(args, "cd_union_datasets", ["whu_cd", "levircd", "levircdplus", "s2looking"])
        return [os.path.join(root, str(x)) for x in cd_datasets]
    if ds == "levircd_union":
        return [os.path.join(root, "levircd"), os.path.join(root, "levircdplus")]
    if ds == "levircd":
        return [os.path.join(root, "levircd")]
    if ds == "levircdplus":
        return [os.path.join(root, "levircdplus")]
    if ds == "s2looking":
        return [os.path.join(root, "s2looking")]
    return None


def dataset_id_for_run(args) -> str:
    """
    Create a compact dataset_id string for caching that does NOT embed absolute paths.
    """
    import hashlib

    roots = infer_roots_for_run(args)
    if roots:
        roots_str = ",".join(sorted([str(r) for r in roots]))
        roots_hash = hashlib.md5(roots_str.encode()).hexdigest()[:12]
        roots_part = f"hash{roots_hash}"
    else:
        roots_part = "default"
    return (
        f"{getattr(args,'dataset_name',None)}__roots={roots_part}"
        f"__rgb={int(getattr(args,'mask_rgb_by_location', False))}"
        f"__grid={getattr(args,'mask_rgb_grid_size', None)}"
        f"__mode={getattr(args,'mask_rgb_index_mode', None)}"
    )

