"""
Helper modules for `train_remote_var.py`.

We keep the top-level training script as a thin entrypoint while moving most
implementation details (arg parsing, resume helpers, viz index caching, and
train/val loops) into this package.
"""

