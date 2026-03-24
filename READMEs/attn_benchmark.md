# DopplerNet Self-Attention Benchmark

Transformer-based models for multi-task Doppler audio analysis.  
Two architectures trained and evaluated on two dataset versions with a single toggle.

---

## Tasks

| Task | Output | Loss |
|---|---|---|
| Trajectory path classification | straight / parabola / bezier | Cross-entropy (label smoothing 0.05) |
| Source speed estimation | m/s | Huber (β = 0.5), normalised by 50 m/s |
| Source distance estimation | m | Huber (β = 0.5), log-normalised by MAX_DIST_M |

---

## Dataset Versions

| Version | Dataset folder | Distance range | `MAX_DIST_M` |
|---|---|---|---|
| `v1` | `Datasets/neurips_v1/audio_clips` | 5 – 120 m | 120.0 |
| `v2` | `Datasets/neurips_v2/audio_clips` | 5 – 1000 m | 1000.0 |

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

## Toggle

At the very top of the notebook, set one variable:

```python
TRAIN_ON = "v2"   # "v1"  →  neurips_v1, MAX_DIST_M = 120
                  # "v2"  →  neurips_v2, MAX_DIST_M = 1000
```

Everything downstream — data path, model save path, results path, distance bins,
model names — updates automatically.  
v1 and v2 outputs are written to completely separate folders and never overwrite each other.

---

## Models

### DopplerTransformer1D  —  `1D-Attn-{v1|v2}`

Pure self-attention model operating on the 7-channel CQT ridge trajectory.

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

**Architecture:**

```
Input (B, 7, 432)
  └─ Permute → (B, 432, 7)
  └─ Linear projection: 7 → 128          per time step
  └─ Sinusoidal positional encoding
  └─ Transformer Encoder × 3 layers
        nhead = 4 | dim_ff = 256 | dropout = 0.1 | norm_first = True
  └─ Mean pool over time → (B, 128)
  └─ Shared head: LayerNorm → Linear(128) → ReLU → Dropout(0.2)
  └─ Path head:   Linear(128→64) → ReLU → Linear(64→3)
  └─ Speed head:  Linear(128→64) → ReLU → Linear(64→1)
  └─ Dist head:   Linear(128→64) → ReLU → Linear(64→1)
```

Kept at 3 layers / d=128 to avoid overfitting on 800 training clips.

---

### DopplerCNNTransformer2D  —  `2D-Attn-{v1|v2}`

Hybrid CNN front-end + Transformer encoder operating on the full log-CQT spectrogram.  
The CNN extracts local time-frequency structure; the Transformer models long-range temporal dependencies across the Doppler sweep.

**Input:** `(B, 1, 84, 432)` — log-CQT spectrogram, per-bin z-score normalised

**Architecture:**

```
Input (B, 1, 84, 432)
  └─ CNN Backbone (5 × Conv2d + BN + ReLU, 2 × MaxPool2d(2×2))
        Channels: 1 → 32 → 64 → [pool] → 128 → 256 → [pool] → 128
        Shape after pooling: (B, 128, 21, 108)
  └─ Learned frequency compression
        Conv2d(128→64, kernel 21×1)  →  (B, 64, 1, 108)
        Squeeze freq axis            →  (B, 64, 108)
        [21×1 kernel spans all remaining freq bins — learns Doppler-relevant bin weighting]
  └─ Pointwise projection
        Conv1d(64→128, k=1)          →  (B, 128, 108)
  └─ Permute → (B, 108, 128)
  └─ Sinusoidal positional encoding
  └─ Transformer Encoder × 2 layers
        nhead = 4 | dim_ff = 256 | dropout = 0.1 | norm_first = True
  └─ Mean pool over time → (B, 128)
  └─ Shared head: LayerNorm → Linear(256) → ReLU → Dropout(0.2) → Linear(128) → ReLU
  └─ Path head:   Linear(128→64) → ReLU → Linear(64→3)
  └─ Speed head:  Linear(128→64) → ReLU → Linear(64→1)
  └─ Dist head:   Linear(128→64) → ReLU → Linear(64→1)
```

2 Transformer layers (vs 3 in 1D) because the CNN already handles local structure.

> **Note:** This is a hybrid model — CNN for local feature extraction, Transformer for
> global temporal modelling. It is not a pure self-attention architecture.
> `DopplerTransformer1D` is fully attention-based throughout.

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

**Best checkpoint metric:**

$$\mathcal{M} = \frac{\text{Speed MAE}}{s_{\max}} + \frac{\text{Dist MAE}}{d_{\max}} + \left(1 - \frac{\text{Path Acc}}{100}\right)$$

Checkpoints saved at three points per epoch: midpoint, every 20 batches, and end-of-epoch.  
All saves overwrite a single file per model — no storage bloat.

---

## Output Structure

```
doppler_models/
  v1_attn_benchmark/
    attn_model_1d/  attn_1d.pt
    attn_model_2d/  attn_2d.pt
  v2_attn_benchmark/
    attn_model_1d/  attn_1d.pt
    attn_model_2d/  attn_2d.pt

results/
  v1_attn_benchmark/
    1d_attn/
      attn_results_1d.csv
      attn_summary_1d.csv
      figures/
        confusion_matrices.png
        speed_scatter.png
        distance_scatter.png
        mae_comparison.png
        dist_mae_by_range.png
        error_histograms.png
        per_path_mae.png
        r2_scores.png          ← additional vs CNN benchmark
    2d_attn/
      attn_results_2d.csv
      attn_summary_2d.csv
      figures/  (same set)
    1d_vs_2d/
      attn_summary_comparison.csv
      figures/
        summary_bars.png
        speed_scatter_overlay.png
        dist_scatter_overlay.png
        per_path_accuracy.png
        dist_mae_bins_comparison.png
  v2_attn_benchmark/
    (same structure)
```

---

## Reported Metrics

| Metric | Description |
|---|---|
| Path accuracy (%) | Overall 3-class classification accuracy on test set |
| Speed MAE (m/s) | Mean absolute error of speed regression |
| Speed median error | Median absolute error of speed regression |
| Speed bias | Mean signed error — positive = overestimate |
| Speed R² | Coefficient of determination for speed regression |
| Dist MAE (m) | Mean absolute error of distance regression |
| Dist median error | Median absolute error of distance regression |
| Dist bias | Mean signed error — positive = overestimate |
| Dist R² | Coefficient of determination for distance regression |

R² is reported in addition to MAE to give a normalised sense of explained variance,  
which is especially useful when comparing v1 (120 m range) vs v2 (1000 m range) results directly.

---

## Notebook Structure

| Section | Content |
|---|---|
| **Section 1 — Setup** | Drive mount, constants, dataset toggle logic, splits, feature extraction, dataset classes, shared training infrastructure (`train_model`, `run_inference`, `plot_results`) |
| **Section 2A** | `DopplerTransformer1D` definition + training |
| **Section 2B** | `DopplerCNNTransformer2D` definition + training |
| **Section 3A** | 1D model evaluation — inference, metrics, all figures |
| **Section 3B** | 2D model evaluation — inference, metrics, all figures |
| **Section 3C** | Comparative analysis — summary table, winner-per-metric, comparative figures |

---

## Relationship to CNN Benchmark

This notebook is a direct successor to the CNN benchmark (`v1/v2_cqt_benchmark`).  
Feature extraction, loss function, training infrastructure, evaluation protocol,  
and output folder structure are intentionally identical to allow fair model comparison.

The only differences are:

| | CNN Benchmark | Attn Benchmark |
|---|---|---|
| 1D temporal modelling | Conv1d encoder + single additive attention | Full Transformer encoder (3 layers) |
| 2D temporal modelling | Conv2d backbone + single additive attention | Conv2d backbone + Transformer encoder (2 layers) |
| Positional information | Implicit via convolutions | Explicit sinusoidal encoding |
| Extra metric | — | R² for speed and distance |
