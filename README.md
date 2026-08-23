# MixPS-MPI-18K

MixPS-MPI-18K is a paired harmonic-image dataset for dual-tracer magnetic
particle imaging (MPI) decomposition. Every HDF5 sample contains a mixed
measurement **M**, its Perimag component **P**, and its Synomag-70 component
**S**, together with acquisition and synthesis metadata.

The release name uses the conventional rounded size "18K". The current
release contains **18,242 HDF5 samples** in total.

<img width="4925" height="3116" alt="figure5" src="https://github.com/user-attachments/assets/a728468b-6211-4e78-8806-5859a0dfcb81" />

## Dataset partitions

| Subset                  |            Train |      Validation |            Test |            Total | Domain                   |
| ----------------------- | ---------------: | --------------: | --------------: | ---------------: | ------------------------ |
| `MixPS-Simulation-7K` |            6,300 |             700 |             701 |            7,701 | MNIST-derived simulation |
| `MixPS-Synthetic-10K` |            7,350 |           1,575 |           1,575 |           10,500 | paired phantom synthesis |
| `MixPS-Realworld-41`  |                0 |               0 |              41 |               41 | real MPI measurements    |
| **Total**         | **13,650** | **2,275** | **2,317** | **18,242** |                          |

`MixPS-Simulation-7K/test/mix_00701.h5` is an additional strong
P-dominant stress-test sample (`Alpha=50`, `Beta=1`). The standard simulation
test set consists of `mix_00001.h5` through `mix_00700.h5`.

## Repository layout

```text
MixPS-MPI-18K/
├── data/
│   ├── MixPS-Simulation-7K/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── MixPS-Synthetic-10K/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── MixPS-Realworld-41/
│       ├── train/                 # intentionally empty
│       ├── val/                   # intentionally empty
│       └── test/
├── docs/
│   ├── DATASET_INTRODUCTION.md
│   ├── DATA_LOADING_AND_USAGE.md
│   └── ZENODO_UPLOAD.md
├── examples/
│   └── load_mixps.py
├── manifests/
│   ├── dataset_manifest.csv
│   └── SHA256SUMS
├── tools/
│   ├── build_manifest.py
│   ├── package_zenodo.sh
│   └── validate_dataset.py
└── requirements.txt
```

## One HDF5 sample

```text
mix_XXXXX.h5
├── /mix/image/real    float32 or float64, [6, 64, 64]
├── /mix/image/imag    float32 or float64, [6, 64, 64]
├── /P/image/real      float32 or float64, [6, 64, 64]
├── /P/image/imag      float32 or float64, [6, 64, 64]
├── /S/image/real      float32 or float64, [6, 64, 64]
├── /S/image/imag      float32 or float64, [6, 64, 64]
└── root attributes    sample metadata
```

The public notation `M` maps to the HDF5 group `/mix`. The six planes are
harmonics `2f` through `7f`, with nominal frequencies 6, 9, 12, 15, 18, and
21 kHz. The spatial image size is 64 x 64 and the field of view is 20 mm x
20 mm. For the simulation and synthetic subsets, ground truth follows the
convention

```text
P = Alpha * raw_P
S = Beta  * raw_S
M = P + S
```

For `MixPS-Realworld-41`, M, P, and S are paired real measurements. M is an
independently acquired mixture rather than an array constructed by adding the
stored P and S scans, so exact element-wise `M=P+S` is neither expected nor
enforced. This measurement-domain residual is intentionally preserved.

The original HDF5 payloads are retained byte-for-byte so their provenance
attributes remain unchanged. In particular, the historical root attribute
`Project_Name` may still read `MixPS-MPI-50K`; the curated public release is
identified by this directory and its manifest as `MixPS-MPI-18K`.

## Quick start

```bash
python -m pip install -r requirements.txt
python examples/load_mixps.py \
  --root data \
  --dataset MixPS-Simulation-7K \
  --split train \
  --index 0
```

See [DATA_LOADING_AND_USAGE.md](docs/DATA_LOADING_AND_USAGE.md) for NumPy and
PyTorch usage, and [DATASET_INTRODUCTION.md](docs/DATASET_INTRODUCTION.md) for
the full dataset card.

## Integrity check

```bash
python tools/validate_dataset.py --root data --mode full
sha256sum --check manifests/SHA256SUMS
```

## Citation

Please cite the Zenodo record associated with the release. Replace the DOI
placeholder after reserving or publishing the Zenodo DOI:

```bibtex
@dataset{li_2026_mixps_mpi_18k,
  author    = {Li, Guanghui},
  title     = {MixPS-MPI-18K: A Paired Harmonic-Image Dataset for
               Dual-Tracer Magnetic Particle Imaging},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22069325},
  url       = {https://doi.org/10.5281/zenodo.22069325}
}
```

## License and contact

The dataset authors must select and add the final data license before the
Zenodo record is published. This is intentionally not inferred automatically,
because source-data and redistribution terms must be verified by the authors.

Contact: Guanghui Li, Beihang University (BUAA),
`liguanghui@buaa.edu.cn`.
