# MixPS-Net

Official minimal PyTorch implementation of MixPS-Net for supervised Perimag /
Synomag harmonic-image separation on **MixPS-MPI-18K**.

## Architecture

The implementation follows the four stages in the paper figure:

1. **HFE:** `M [B,K,H,W] -> F [B,2C,H,W]` using two `1x1` harmonic
   projections and two reflection-padded residual blocks.
2. **PFD:** independent P- and S-specific two-head self-attention produces
   `FP, FS [B,2C,H,W]`.
3. **OC-Factorizer:** each tracer feature is decomposed into
   `Zc, Zh [B,C,H,W]`. A nonnegative shared concentration and `K` signed
   spatial operators are decoded from these factors.
4. **HCMM rendering:** every operator is convolved with the same tracer
   concentration field to reconstruct the `K` harmonic images. The mixed
   prediction is `M_hat = P_hat + S_hat`.

Default dimensions are `K=12`, `C=32`, two attention heads, `64x64` inputs,
and `21x21` harmonic operators.

## Installation

Python 3.9+ and PyTorch 2.1+ are required.

```bash
git clone https://github.com/BUAALGH/MixPS-MPI-18K.git
cd MixPS-MPI-18K/MixPS-Net
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Dataset

Download and extract MixPS-MPI-18K so that the HDF5 files follow this layout:

```text
/path/to/MixPS-MPI-18K/data/
├── MixPS-Simulation-7K/{train,val,test}/mix_*.h5
├── MixPS-Synthetic-10K/{train,val,test}/mix_*.h5
└── MixPS-Realworld-41/test/mix_*.h5
```

Each file must contain `mix`, `P`, and `S` groups with
`image/real` and `image/imag` arrays of shape `[6,64,64]`. Channels are loaded
as `[Re(2f)..Re(7f), Im(2f)..Im(7f)]`. One MaxAbs scale computed only from
the mixture is shared by M, P, and S.

## Training

The reference configuration trains for **100 epochs** with AdamW, cosine
learning-rate decay, batch size 8, and seed 42:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/mixps_net.yaml \
  --data-root /path/to/MixPS-MPI-18K/data \
  --output-dir runs/mixps_net
```

The best validation checkpoint is written to `runs/mixps_net/best.pt`.
Resume an interrupted run with:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/mixps_net.yaml \
  --data-root /path/to/MixPS-MPI-18K/data \
  --output-dir runs/mixps_net \
  --resume runs/mixps_net/last.pt
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint runs/mixps_net/best.pt \
  --data-root /path/to/MixPS-MPI-18K/data \
  --output runs/mixps_net/metrics.json
```

Evaluation reports mean and population standard deviation of per-sample,
per-harmonic PSNR and SSIM for Perimag and Synomag on Simulation-7K,
Synthetic-10K, and RealWorld-41. Each channel uses its ground-truth dynamic
range; the supplemental Simulation sample `mix_00701.h5` is excluded by
default.

## Reproducibility

The full protocol is stored in [`configs/mixps_net.yaml`](configs/mixps_net.yaml).
Training saves the copied configuration, complete history, optimizer,
scheduler, model state, epoch, and best validation loss. The default objective
is

```text
L = 1.0 * L1(P_hat,P) + 1.0 * L1(S_hat,S) + 0.25 * L1(M_hat,M).
```

Run the model test with `python -m pytest tests/test_model.py` after installing
`pytest`.

## License

Code is released under the MIT License. Dataset usage remains subject to the
license attached to the MixPS-MPI-18K release.
