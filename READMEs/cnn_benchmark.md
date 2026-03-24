# DopplerNet CNN CQT Benchmark

CNN-based baseline models for multi-task Doppler audio analysis.  
Two architectures trained and evaluated across two dataset versions.

---

## Tasks

| Task | Output | Loss |
|---|---|---|
| Trajectory path classification | straight / parabola / bezier | Cross-entropy (label smoothing 0.05) |
| Source speed estimation | m/s | Huber (β = 0.5), normalised by 50 m/s |
| Source distance estimation | m | Huber (β = 0.5), log-normalised by MAX_DIST_M |

---

## Dataset Versions

| Version | Dataset folder | Distance range | `MAX_DIST_M` | Distance bins |
|---|---|---|---|---|
| `v1` | `Datasets/neurips_v1/audio_clips` | 5 – 120 m | 120.0 | 0-20, 20-40, 40-60, 60-100, 100-130 |
| `v2` | `Datasets/neurips_v2/audio_clips` | 5 – 1000 m | 1000.0 | 0-100, 100-250, 250-500, 500-750, 750-1000 |

Both versions share the same label format:
```
{vehicle}_{path}_{speed}mps_{dist}m_{id}.wav
```
7 vehicle types · 3 trajectory classes · speeds 10–50 m/s · 1 000 clips total.

Splits are class-balanced before capping:
- **Train:** 800 clips (80%)
- **Val:** 100 clips (10%)
- **Test:** 100 clips (10%)

---

## Switching Between v1 and v2

The notebook currently targets v2. To switch to v1, change the four
path/constant lines at the top of Section 1 and uncomment the v1 distance bins
in Sections 3A, 3B, and the comparative analysis section:

```python
# v2 (current)
DATA_ROOT  = os.path.join(ROOT, "Datasets", "neurips_v2", "audio_clips")
MODEL_ROOT = os.path.join(ROOT, "doppler_models", "v2_cqt_benchmark")
RESULTS_DIR = os.path.join(ROOT, "results",        "v2_cqt_benchmark")
MAX_DIST_M = 1000.0

# v1 (swap to these)
DATA_ROOT  = os.path.join(ROOT, "Datasets", "neurips_v1", "audio_clips")
MODEL_ROOT = os.path.join(ROOT, "doppler_models", "v1_cqt_benchmark")
RESULTS_DIR = os.path.join(ROOT, "results",        "v1_cqt_benchmark")
MAX_DIST_M = 120.0
```

v1 and v2 outputs are written to completely separate folders and never
overwrite each other.

---

## Models

### DopplerNet1D  —  `1D-CQT`

1D convolutional model operating on the 7-channel CQT ridge trajectory.

**Input:** `(B, 7, 432)` — 7 hand-crafted Doppler features × 432 time frames

| Channel | Name | Description |
|---|---|---|
| 0 | `dfdt_norm` | Normalised frequency derivative ∈ [−1, 1] |
| 1 | `dfdt2` | Second derivative of dominant frequency, normalised by std |
| 2 | `sign_dfdt` | Sign of dfdt: +1 approach, −1 recession |
| 3 | `freq_norm` | Dominant CQT bin frequency normalised to [0, 1] |
| 4 | `rms_norm` | Frame RMS energy normalised to [0, 1] |
| 5 | `topk_freq` | Normalised frequency of highest-magnitude top-k component |
| 6 | `t_rel` | Relative time ∈ [−1, +1], linearly spaced |

The dominant frequency is loaded from precomputed `frequency.npy` when available,
otherwise extracted as argmax over CQT bins. A median filter (kernel=5) suppresses
spurious frequency jumps before any derivative is computed.

Training augmentation: Gaussian noise (σ=0.004) on `freq_norm` + random temporal
roll of ±20 frames.

**Architecture:**

```
Input (B, 7, 432)
  └─ Conv1d(7→64,   k=9) + BN + ReLU + Dropout1d(0.10)
  └─ Conv1d(64→128, k=7) + BN + ReLU + Dropout1d(0.10)
  └─ Conv1d(128→256,k=5) + BN + ReLU + Dropout1d(0.15)
  └─ Conv1d(256→128,k=3) + BN + ReLU
     → feature map H ∈ (B, 128, 432)   [all convolutions: same padding]
  └─ Additive temporal attention
        w = softmax(Conv1d(tanh(Conv1d(H, 128→64)), 64→1))
        g = Σ wₜ hₜ  →  (B, 128)
  └─ Shared head: LayerNorm → Linear(128) → ReLU → Dropout(0.2)
  └─ Path head:   Linear(128→64) → ReLU → Linear(64→3)
  └─ Speed head:  Linear(128→64) → ReLU → Linear(64→1)
  └─ Dist head:   Linear(128→64) → ReLU → Linear(64→1)
```

---

### DopplerNet2D  —  `2D-CQT`

2D convolutional model operating on the full log-CQT spectrogram with a
learned frequency compression layer.

**Input:** `(B, 1, 84, 432)` — log-CQT spectrogram, per-bin z-score normalised

**Architecture:**

```
Input (B, 1, 84, 432)
  └─ Conv2d(1→32,   3×3) + BN + ReLU
  └─ Conv2d(32→64,  3×3) + BN + ReLU
  └─ MaxPool2d(2×2)               → freq: 84→42, time: 432→216
  └─ Conv2d(64→128, 3×3) + BN + ReLU
  └─ Conv2d(128→256,3×3) + BN + ReLU
  └─ MaxPool2d(2×2)               → freq: 42→21,  time: 216→108
  └─ Conv2d(256→128,3×3) + BN + ReLU
     → feature map (B, 128, 21, 108)   [all convolutions: same padding]
  └─ Learned frequency compression
        Conv2d(128→64, kernel 21×1)  →  (B, 64, 1, 108)
        Squeeze freq axis            →  (B, 64, 108)
        [kernel spans all 21 remaining freq bins — learns Doppler-relevant
         bin weighting rather than naive mean-pooling]
  └─ Pointwise projection
        Conv1d(64→128, k=1) + BN + ReLU  →  (B, 128, 108)
  └─ Permute → (B, 108, 128)
  └─ Additive temporal attention
        w = softmax(Linear(tanh(Linear(hₜ, 128→64)), 64→1))
        g = Σ wₜ hₜ  →  (B, 128)
  └─ Shared head: LayerNorm → Linear(256) → ReLU → Dropout(0.2) → Linear(128) → ReLU
  └─ Path head:   Linear(128→64) → ReLU → Linear(64→3)
  └─ Speed head:  Linear(128→64) → ReLU → Linear(64→1)
  └─ Dist head:   Linear(128→64) → ReLU → Linear(64→1)
```

> The attention mechanism in both models is **additive (content-based) attention**,
> not multi-head self-attention. It computes a scalar importance weight per time
> step and produces a single weighted sum. See the Self-Attention Benchmark notebook
> for full Transformer encoder variants of both models.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW, weight decay 1×10⁻⁴ |
| Learning rate | 3×10⁻⁴ |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10, η_min=10⁻⁶) |
| Batch size | 16 |
| Epochs | 500 |
| Gradient clipping | max norm 2.0 |
| Validation frequency | every 20 epochs + final epoch |

**Multi-task loss:**

$$\mathcal{L} = \mathcal{L}_{CE}(\hat{p}, p) + 2.5 \cdot \mathcal{L}_{Huber}\!\left(\hat{s},\ \frac{s}{s_{\max}}\right) + 1.5 \cdot \mathcal{L}_{Huber}\!\left(\hat{d},\ \frac{\log(1+d)}{\log(1+d_{\max})}\right)$$

Distance is regressed in log-space to compress the long tail of the distance range,
especially important for v2 (5–1000 m).

**Best checkpoint metric:**

$$\mathcal{M} = \frac{\text{Speed MAE}}{s_{\max}} + \frac{\text{Dist MAE}}{d_{\max}} + \left(1 - \frac{\text{Path Acc}}{100}\right)$$

Checkpoints are saved at three points per epoch: midpoint batch, every 20 batches
(rolling), and end-of-epoch. All saves overwrite a single file per model — no
storage bloat. Mid-epoch resume is fully supported.

---

## Output Structure

```
doppler_models/
  v2_cqt_benchmark/          (or v1_cqt_benchmark/)
    cnn_model_1d/  cqt_1d.pt
    cnn_model_2d/  cqt_2d.pt

results/
  v2_cqt_benchmark/          (or v1_cqt_benchmark/)
    1d_cqt/
      cnn_cqt_results_1d.csv
      cqt_summary_1d.csv
      figures/
        confusion_matrices.png
        speed_scatter.png
        distance_scatter.png
        mae_comparison.png
        dist_mae_by_range.png
        error_histograms.png
        per_path_mae.png
    2d_cqt/
      cnn_cqt_results_2d.csv
      cqt_summary_2d.csv
      figures/  (same set)
    1d_vs_2d/
      cqt_summary_comparison.csv
      figures/
        summary_bars.png
        speed_scatter_overlay.png
        dist_scatter_overlay.png
        speed_err_dist_overlay.png
        dist_err_dist_overlay.png
        per_path_accuracy.png
        dist_mae_bins_comparison.png
```

---

## Reported Metrics

| Metric | Description |
|---|---|
| Path accuracy (%) | Overall 3-class classification accuracy on test set |
| Speed MAE (m/s) | Mean absolute error of speed regression |
| Speed median error | Median absolute error — robust to outliers |
| Speed bias | Mean signed error — positive = overestimate |
| Dist MAE (m) | Mean absolute error of distance regression |
| Dist median error | Median absolute error of distance regression |
| Dist bias | Mean signed error — positive = overestimate |

---

## Notebook Structure

| Section | Content |
|---|---|
| **Setup** | Drive mount, constants, dataset paths, splits, feature extraction, dataset classes, shared training infrastructure (`train_model`, `run_inference`, `plot_results`) |
| **Section 2A** | `DopplerNet1D` definition + training |
| **Section 3A** | 1D model evaluation — inference, printed metrics, all figures |
| **Section 2B** | `DopplerNet2D` definition + training |
| **Section 3B** | 2D model evaluation — inference, printed metrics, all figures |
| **Comparative Analysis** | Loads both CSVs, summary table with winner-per-metric, 7 comparative figures |

The comparative analysis section can be run independently in a fresh session
as long as both result CSVs exist.

---

## Relationship to Self-Attention Benchmark

This notebook is the **CNN baseline** for the project. The Self-Attention Benchmark
(`v1/v2_attn_benchmark`) uses identical feature extraction, loss function, training
infrastructure, and evaluation protocol to allow direct model comparison.

| | CNN Benchmark (this notebook) | Self-Attention Benchmark |
|---|---|---|
| 1D temporal modelling | Conv1d encoder + additive attention | Full Transformer encoder (3 layers, nhead=4) |
| 2D temporal modelling | Conv2d backbone + additive attention | Conv2d backbone + Transformer encoder (2 layers) |
| Positional information | Implicit via convolutions | Explicit sinusoidal encoding |
| Extra metric | — | R² for speed and distance |
| Model names | `1D-CQT`, `2D-CQT` | `1D-Attn-v1/v2`, `2D-Attn-v1/v2` |
