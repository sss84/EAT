# Exposure-Aware Training (EAT)

**Lightweight degradation-based training strategy for low-light object detection — without target-domain data, detector modifications, or inference overhead.**

> Su, Y., & Lu, M. (2026). Exposure-Aware Training for Low-Light Object Detection Without Target-Domain Data. *Journal of Imaging*, 12(6), 245. https://doi.org/10.3390/jimaging12060245

---

## Motivation

Low-light object detection suffers from an **illumination gap** between normal-light training data and low-light test data. Existing solutions typically require one or more of:

- Re-designing the detector architecture
- Adding an image-enhancement module at inference
- Using target-domain (low-light) data during training

EAT avoids all three. It operates **only at training time** by exposing the detector to controlled illumination degradation, learning more illumination-stable representations with **zero additional inference cost**.

## Method

### Degradation Model

During training, each normal-light image is transformed as:

```
I'(x, y) = α · I(x, y) + N(0, σ²)
```

| Parameter | Value | Source |
|-----------|-------|--------|
| α (illumination attenuation) | ≈ 0.745 | Estimated from paired low-light images (LSD dataset, 115 valid pairs after filtering) |
| σ (Gaussian noise) | 0.01 | Fixed; follows sensor noise modeling of Unprocessing |

Object annotations remain unchanged. The degradation is applied **online** during training and removed entirely at inference.

### Key Design Principles

1. **Task-oriented, not distribution-fitting** — only brightness attenuation and additive noise are retained; secondary effects (color shift, vignetting) are omitted.
2. **Data-driven parameters** — α is estimated from real paired low-light data rather than chosen heuristically.
3. **Training-only** — no enhancement module, no architecture change, no inference-stage computation.

### Training Pipeline

```
Normal-light images + annotations
        │
        ▼
EAT degradation: I' = α·I + N(0, σ²)
        │
        ▼
Train detector (YOLOv8 / Faster R-CNN) on degraded images
        │
        ▼
Inference: feed original low-light images directly — no preprocessing
```

## Installation

Environment requirements are consistent with the underlying diffusion-model framework:

```bash
git clone https://github.com/sss84/EAT.git
cd EAT
pip install -e .
```

### Dependencies

- Python ≥ 3.6
- PyTorch ≥ 2.0
- torchvision
- accelerate
- einops
- ema-pytorch ≥ 0.4.2
- numpy
- pillow
- pytorch-fid
- scipy
- tqdm

## Usage

### 1. Estimate degradation parameters (optional)

If you want to re-estimate α from your own paired low-light data:

```python
from eat.parameter_estimation import estimate_alpha
# alpha = estimate_alpha(normal_light_dir, low_light_dir)
```

Default α = 0.745, σ = 0.01 work stably across datasets.

### 2. Apply EAT degradation during training

Integrate the degradation transform into your detector's training dataloader:

```python
from eat.degradation import EATDegradation

transform = EATDegradation(alpha=0.745, sigma=0.01)
# Apply transform to training images only; keep annotations unchanged
```

Then train your detector (YOLOv8, Faster R-CNN, etc.) with standard detection loss — no architecture changes needed.

### 3. Inference

At inference, use the trained detector directly on low-light images. **No degradation or enhancement is applied.**

## Datasets

Experiments are conducted on:

- **ExDark** — Exclusively Dark dataset (low-light benchmark)
- **VOC** — PASCAL VOC (normal-light source domain)
- **TLD** — mixed-illumination detection dataset
- **LSD** — low-light paired dataset for parameter estimation

## Results

EAT achieves consistent improvements under both **cross-domain** (VOC → ExDark, zero target-domain data) and **mixed-training** settings:

- Stable gains on YOLOv8 and Faster R-CNN (~3.4–3.5% relative improvement)
- More noticeable gains for illumination-sensitive categories
- 68.6% mAP50 on ExDark, comparable to architecture-modification methods while maintaining a simpler pipeline

## Citation

```bibtex
@article{su2026exposure,
  title   = {Exposure-Aware Training for Low-Light Object Detection Without Target-Domain Data},
  author  = {Su, Yawen and Lu, Min},
  journal = {Journal of Imaging},
  volume  = {12},
  number  = {6},
  pages   = {245},
  year    = {2026},
  doi     = {10.3390/jimaging12060245}
}
```

## License

MIT
