# RemoteVAR: Autoregressive Visual Modeling for Remote Sensing Change Detection

**Accepted to IGARSS 2026.** [arXiv:2601.11898](https://arxiv.org/abs/2601.11898)

## Main entry points

- `train_remote_var.py`: train RemoteVAR.
- `calculate_token_frequencies.py`: precompute optional token-loss weights.
- `generate_refiner_predictions.py`: cache autoregressive mask latents.
- `train_decoder_refiner.py`: fine-tune the skip-conditioned decoder.
- `inference.py`: evaluate RemoteVAR, optionally with a decoder refiner.

## Installation

The experiments used Python 3.9, PyTorch 2.5.1, and CUDA 12.1. From the
`RemoteVAR` directory, install the matching PyTorch build and the required
packages:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The model runs without optional attention kernels. To reproduce the accelerated
experiment environment, install:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install xformers==0.0.28.post3
```

## Pretrained weights

The released RemoteVAR and decoder-refiner weights are available from our
[Hugging Face repository](https://huggingface.co/yilmazkorkmaz/RemoteVAR).
Both checkpoints were trained on the `cd_union` dataset, which combines:

- WHU-CD
- LEVIR-CD
- LEVIR-CD+
- S2Looking

The decoder refiner used cached RemoteVAR predictions generated from the same
four-dataset training union.

Inference also requires the VQ-VAE checkpoint from the
[original VAR release](https://github.com/FoundationVision/VAR):

```bash
mkdir -p pretrained
wget -O pretrained/model.safetensors \
  https://huggingface.co/yilmazkorkmaz/RemoteVAR/resolve/main/model.safetensors
wget -O pretrained/best_decoder_refiner.pth \
  https://huggingface.co/yilmazkorkmaz/RemoteVAR/resolve/main/best_decoder_refiner.pth
wget -O pretrained/vae_ch160v4096z32.pth \
  https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
```

To train RemoteVAR from the ControlVAR initialization, also download the
depth-16 checkpoint from the
[original ControlVAR release](https://github.com/lxa9867/ControlVAR):

```bash
wget -O pretrained/d16.pth \
  https://huggingface.co/qiuk6/ControlVAR/resolve/main/d16.pth
```

The Hugging Face repository contains weights only. The matching model
hyperparameters are provided in `configs/change_detection.yaml` and
`configs/decoder_refiner.yaml`. Paths can be overridden with
`--vqvae_pretrained_path`, `--var_pretrained_path`, `--checkpoint`, and
`--decoder_refiner_checkpoint`.

## Dataset layout

All datasets are resolved beneath one root:

```text
<DATASET_ROOT>/
  whu_cd/
  levircd/
  levircdplus/
  s2looking/
```

Each dataset directory should contain:

```text
A/
B/
gt/
train.txt
val.txt
test.txt
```

Datasets without a validation or test split are automatically skipped from that
split when building `cd_union`.

## Train RemoteVAR

Single GPU:

```bash
python train_remote_var.py \
  --config configs/change_detection.yaml \
  --dataset_root /path/to/datasets
```

W&B is disabled in the published config. Add `--use_wandb --wandb_mode online`
to enable cloud logging, or use `--wandb_mode offline`.

Eight GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m accelerate.commands.launch --num_processes 8 \
  train_remote_var.py \
  --config configs/change_detection.yaml \
  --dataset_root /path/to/datasets
```

Optional precomputed token weighting can be generated before training:

```bash
python calculate_token_frequencies.py \
  --config configs/change_detection.yaml \
  --dataset_root /path/to/datasets
```

Then set `use_precomputed_weights: true` and point `token_freq_path` at the
generated JSON. If weighting is enabled and the JSON is absent, training runs
this calculation automatically.

## Generate decoder-refiner predictions

This generates index-aligned `mask_fhat` caches for the train and validation
splits. The checkpoint architecture is read from the same RemoteVAR config.

Single GPU:

```bash
python generate_refiner_predictions.py \
  --config configs/change_detection.yaml \
  --checkpoint pretrained/model.safetensors \
  --dataset_root /path/to/datasets \
  --out_dir predictions/remotevar_best \
  --splits train val \
  --batch_size 1 \
  --deterministic \
  --save_dtype fp16
```

Eight GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  generate_refiner_predictions.py \
  --config configs/change_detection.yaml \
  --checkpoint pretrained/model.safetensors \
  --dataset_root /path/to/datasets \
  --out_dir predictions/remotevar_best \
  --splits train val \
  --batch_size 1 \
  --num_workers 2 \
  --deterministic \
  --save_dtype fp16
```

The generated files are:

```text
predictions/remotevar_best/cd_union_train_mask_fhat.pt
predictions/remotevar_best/cd_union_val_mask_fhat.pt
```

## Train the decoder refiner

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m accelerate.commands.launch --num_processes 8 \
  train_decoder_refiner.py \
  --config configs/decoder_refiner.yaml \
  --dataset_root /path/to/datasets \
  --remotevar_checkpoint pretrained/model.safetensors \
  --predictions_dir predictions/remotevar_best
```

## Inference

RemoteVAR only:

```bash
python inference.py \
  --config configs/change_detection.yaml \
  --checkpoint pretrained/model.safetensors \
  --dataset_root /path/to/datasets \
  --test_dataset_name whu_cd
```

With a decoder refiner:

```bash
python inference.py \
  --config configs/change_detection.yaml \
  --checkpoint pretrained/model.safetensors \
  --decoder_refiner_checkpoint pretrained/best_decoder_refiner.pth \
  --dataset_root /path/to/datasets \
  --test_dataset_name whu_cd
```

## Acknowledgements

RemoteVAR uses code and pretrained models released by
[ControlVAR](https://github.com/lxa9867/ControlVAR) and
[VAR](https://github.com/FoundationVision/VAR).

## Citation

If you find RemoteVAR useful in your research, please cite:

```bibtex
@article{korkmaz2026remotevar,
  title={RemoteVAR: Autoregressive Visual Modeling for Remote Sensing Change Detection},
  author={Korkmaz, Yilmaz and Patel, Vishal M},
  journal={arXiv preprint arXiv:2601.11898},
  year={2026}
}
```

