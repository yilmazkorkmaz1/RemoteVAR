import os


def apply_resume_lr_override(optimizer, lr_scheduler, *, resume_lr=None, resume_lr_scale=1.0, logger=None):
    """
    When resuming with Accelerate, optimizer/scheduler state restores the LR.
    If we want to change LR at resume time, we must do it AFTER `accelerator.load_state()`.
    Also scale scheduler.base_lrs to avoid the next scheduler.step() snapping back.
    """
    if optimizer is None:
        return

    old_lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]

    if resume_lr is not None:
        new_lrs = []
        for pg in optimizer.param_groups:
            pg["lr"] = float(resume_lr)
            new_lrs.append(float(pg["lr"]))
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs = list(new_lrs)
        if lr_scheduler is not None and hasattr(lr_scheduler, "_last_lr"):
            lr_scheduler._last_lr = list(new_lrs)
    else:
        scale = float(resume_lr_scale) if resume_lr_scale is not None else 1.0
        if scale == 1.0:
            return
        new_lrs = []
        for pg in optimizer.param_groups:
            pg["lr"] = float(pg.get("lr", 0.0)) * scale
            new_lrs.append(float(pg["lr"]))
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs = [float(x) * scale for x in lr_scheduler.base_lrs]
        if lr_scheduler is not None and hasattr(lr_scheduler, "_last_lr"):
            lr_scheduler._last_lr = [float(x) * scale for x in lr_scheduler._last_lr]

    if logger is not None:
        logger.info(
            f"Applied resume LR override. "
            f"old_lrs={old_lrs} -> new_lrs={[float(pg.get('lr', 0.0)) for pg in optimizer.param_groups]}"
        )


def infer_resume_dir(path: str) -> str:
    # Allow passing either the directory or a file inside it (e.g., model.safetensors)
    if path is None:
        return None
    p = str(path)
    if os.path.isdir(p):
        return p
    return os.path.dirname(p)


def infer_starting_epoch_from_resume_dir(resume_dir: str):
    """
    If resume_dir is .../epoch_99, return 100 (next epoch index to run).
    If parsing fails, return None.
    """
    try:
        base = os.path.basename(os.path.normpath(resume_dir))
        if base.startswith("epoch_"):
            ep = int(base.split("_", 1)[1])
            return ep + 1
    except Exception:
        pass
    return None


def infer_run_name_from_resume_dir(resume_dir: str) -> str:
    """
    Given an Accelerate state dir like:
      experiments/<timestamp>-<run_name>/epoch_99
    infer and return <run_name>.

    Falls back to the parent directory name if parsing fails.
    """
    try:
        exp_dir_name = os.path.basename(os.path.dirname(os.path.normpath(resume_dir)))
        parts = exp_dir_name.split("-")
        # timestamp is YYYY-MM-DD-HH-MM-SS => 6 hyphen-separated parts
        if len(parts) > 6:
            return "-".join(parts[6:])
        return exp_dir_name
    except Exception:
        return os.path.basename(os.path.dirname(os.path.normpath(resume_dir)))


def strip_prefix_from_state_dict(state_dict, prefix: str):
    try:
        keys = list(state_dict.keys())
        if len(keys) == 0:
            return state_dict
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
        return state_dict
    except Exception:
        return state_dict


def load_safetensors_state(path: str):
    try:
        from safetensors.torch import load_file
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for loading .safetensors. Please ensure `safetensors` is installed."
        ) from e
    return load_file(path)


def fallback_load_models_from_accelerate_dir(
    resume_dir: str,
    *,
    accelerator: "Accelerator",
    var,
    cond_model,
    vqvae,
    strict: bool = False,
    logger=None,
):
    """
    Fallback for sequential fine-tuning when architecture changes:
    - Load model weights non-strictly from model*.safetensors in `resume_dir`.
    - Does NOT load optimizer state (param groups may differ).
    """
    model_paths = []
    for name in ["model.safetensors", "model_1.safetensors", "model_2.safetensors"]:
        p = os.path.join(resume_dir, name)
        if os.path.exists(p):
            model_paths.append(p)

    if len(model_paths) == 0:
        raise FileNotFoundError(f"No model*.safetensors found under resume_dir={resume_dir}")

    targets = [accelerator.unwrap_model(var)]
    if cond_model is not None:
        targets.append(accelerator.unwrap_model(cond_model))
    if vqvae is not None:
        targets.append(accelerator.unwrap_model(vqvae))

    n = min(len(model_paths), len(targets))
    if logger is not None:
        logger.warning(
            f"Falling back to weights-only load from {resume_dir} (strict={strict}). "
            f"Found {len(model_paths)} model files, have {len(targets)} model objects; loading {n}."
        )

    for i in range(n):
        sd = load_safetensors_state(model_paths[i])
        sd = strip_prefix_from_state_dict(sd, "module.")
        missing, unexpected = targets[i].load_state_dict(sd, strict=bool(strict))
        if logger is not None:
            logger.warning(
                f"Loaded {os.path.basename(model_paths[i])} into model[{i}] "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

