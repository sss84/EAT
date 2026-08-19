# EAT

Exposure-Aware Training — a lightweight degradation-based training strategy for improving low-light object detection robustness. The degradation (brightness attenuation + Gaussian noise) is applied only during training; no modification to the detector architecture and no extra cost at inference.

## Requirements

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

## Installation

```bash
git clone https://github.com/sss84/EAT.git
cd EAT
pip install -e .
```

## Usage

### Degradation Model

The core training-time degradation follows:

```
I'(x, y) = α · I(x, y) + N(0, σ²)
```

where α ≈ 0.745 (estimated from paired low-light data) and σ = 0.01. Apply this transform to training images only; keep annotations unchanged.

### Training

Integrate the degradation transform into your detector's training dataloader (YOLOv8, Faster R-CNN, etc.) and train with standard detection loss. No architecture changes are needed.

### Inference

Feed low-light images directly into the trained detector — no degradation or enhancement preprocessing is applied.

## Project Structure

```
EAT/
├── eat/                  # core package (diffusion model backbone)
├── setup.py
├── LICENSE
└── README.md
```

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
