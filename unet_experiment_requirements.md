# U-Net Denoising Experiment Requirements

**Project:** Signal denoising for QPSK-like deep-space signals  
**Target model family:** U-Net only  
**Dataset used in current reports:** `deep_space_polygauss_qpsk_bs1024_n400000_c054e749`  
**Primary goal:** identify which U-Net changes materially improve output SNR and generalization across Gaussian and non-Gaussian noise.  
**Secondary goal:** make the U-Net training/evaluation pipeline reliable enough that later comparisons with Transformer, ResNet, Wavelet, and Hybrid models are fair.

---

## 1. Background and baseline context

The old comparison report showed Transformer as the strongest model, especially when trained on non-Gaussian noise:

| Report | Model | Train noise | Gaussian test | Non-Gaussian test |
|---|---|---:|---:|---:|
| `old.md` | Transformer | non-Gaussian | 9.77 dB | 10.61 dB |
| `old.md` | U-Net | non-Gaussian | -2.44 dB | -2.56 dB |
| `new.md` | U-Net | non-Gaussian | 7.43 dB | 7.68 dB |
| `new.md` | ResNet | non-Gaussian | 7.23 dB | 7.48 dB |

The new report shows that U-Net is no longer broken, but it is still likely limited by the current spectral magnitude-mask design. The current U-Net:

- uses STFT magnitude only as input;
- predicts a sigmoid mask in `[0, 1]`;
- multiplies the mask by noisy magnitude;
- reuses noisy phase for iSTFT reconstruction;
- optimizes magnitude-domain loss, while final score is time-domain SNR;
- selects best checkpoint by validation loss, not by validation SNR;
- early-stops almost immediately after the first LR reduction.

This document defines a development plan for agents to implement and run U-Net-only ablations in one controlled training session.

---

## 2. Non-goals and constraints

1. **Do not update Transformer, Wavelet, or Hybrid models.** They can be re-evaluated later only as fixed baselines.
2. **Do not modify dataset generation.** Use the same dataset and same train/validation/test split logic unless explicitly testing split sensitivity.
3. **Do not use stale inference scripts for conclusions.** `inference/inference_unet.py` and `inference/inference_all_models.py` must be fixed or ignored.
4. **Do not compare runs trained with different seeds, splits, STFT settings, or checkpoint-selection logic unless the difference is the intended experiment.**
5. **Do not let `compare_report.py` silently load the wrong architecture.** If a variant changes architecture, save and use `model_config.json` and a variant-aware loader.
6. **Default U-Net behavior must remain backward compatible.** Existing saved baseline weights should still load when using default arguments.

---

## 3. Global requirements for all experiments

### 3.1 Reproducibility

Every run must save:

```text
run_dir/
  experiment_config.json
  model_config.json
  training_args.json
  epoch_metrics.csv
  result.json
  model_best_snr.pth
  model_best_loss.pth
  model_last.pth
  figures/
    training_curves.png
    snr_curve.png
```

`experiment_config.json` must include:

- experiment id;
- git commit hash if available;
- dataset path and dataset uid;
- model variant;
- noise type;
- seed;
- STFT parameters;
- loss profile;
- optimizer/scheduler settings;
- checkpoint metric;
- early-stopping settings;
- code version timestamp.

### 3.2 Fixed default training protocol

Unless an experiment explicitly changes these values, use:

```text
noise_type: non_gaussian for the first screening suite
signal_len: from dataset_config.json, expected 1024
sample_rate: from dataset_config.json
nperseg: 128
noverlap: 96
hop_length: 32
batch_size: 512, or highest stable value on GPU
optimizer: AdamW
learning_rate: 1e-3
weight_decay: 1e-4
epochs: 50
min_epochs: 25
early_stop_patience: 15
scheduler: ReduceLROnPlateau
scheduler_metric: val_snr
scheduler_mode: max
scheduler_patience: 5
scheduler_factor: 0.5
scheduler_cooldown: 2
scheduler_threshold: 0.02 dB
min_lr: 1e-5
checkpoint_metric: val_snr
gradient_clip_norm: 1.0
seed: 42
```

Rationale: the current scheduler uses validation loss, has patience 3, and early stopping has patience 5. That means the LR can be reduced and then the run can stop before the lower LR has time to improve validation SNR.

### 3.3 Metrics

Each experiment must report:

- best validation SNR;
- validation loss at best SNR;
- test SNR on the same noise type as training;
- cross-test SNR on the opposite noise type;
- MSE, MAE, RMSE;
- per-input-SNR curve from `evaluate_per_snr`;
- training time;
- peak GPU memory;
- epoch of best SNR;
- epoch of best loss;
- final learning rate;
- whether early stopping triggered.

Primary ranking metric for screening:

```text
NG train -> NG test SNR
```

Secondary ranking metrics:

```text
NG train -> G test SNR
mean of NG->NG and NG->G
minimum of NG->NG and NG->G
```

A change should be considered promising if it improves the current fixed-trainer baseline by at least **0.3 dB** on NG->NG without hurting NG->G by more than **0.2 dB**. A change above **1.0 dB** is a major improvement and should be repeated with additional seeds.

---

## 4. Required code changes before running the suite

### 4.1 Update `models/autoencoder_unet.py`

Keep the current default model compatible, but parameterize it.

Required new arguments:

```python
class UnetAutoencoder(nn.Module):
    def __init__(
        self,
        input_shape,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        pooling_mode: str = "isotropic",       # isotropic | freq_only_first2 | time_preserve
        output_mode: str = "mask_sigmoid",     # mask_sigmoid | mask_scaled_sigmoid | mask_softplus | direct_mag | residual_mag | complex_crm | complex_stft
        mask_max: float = 1.0,
        softplus_max: float = 3.0,
        use_se: bool = True,
        use_dropout: bool = False,
        dropout_p: float = 0.0,
    ):
        ...
```

Output mode behavior:

```text
mask_sigmoid:
  mask = sigmoid(raw)
  out_mag = mask * noisy_mag

mask_scaled_sigmoid:
  mask = mask_max * sigmoid(raw)
  out_mag = mask * noisy_mag

mask_softplus:
  mask = clamp(softplus(raw), max=softplus_max)
  out_mag = mask * noisy_mag

direct_mag:
  out_mag = softplus(raw) or relu(raw)

residual_mag:
  delta = raw
  out_mag = clamp(noisy_mag + delta, min=0)

complex_crm:
  input: real/imag or real/imag/mag channels
  output: 2-channel complex ratio mask
  out_spec = crm * noisy_spec

complex_stft:
  input: real/imag or real/imag/mag channels
  output: 2-channel clean real/imag STFT
```

Important: for all non-complex modes, `forward()` can return either a mask or final magnitude, but this must be explicit. Do not keep ambiguous behavior. Recommended:

```python
def forward(self, x, noisy_mag=None, noisy_spec=None):
    """Return a dictionary: {'out_mag': ..., 'mask': ..., 'out_spec': ...}."""
```

If this is too large a refactor, add a wrapper function in the trainer that interprets the model output consistently.

### 4.2 Fix pooling options

Current U-Net downsamples both frequency and time at every level. For a 1024-sample signal with `nperseg=128` and hop 32, the spectrogram is roughly `65 x 33`; after three `2x2` pools the bottleneck has only about `8 x 4` spatial positions.

Add pooling modes:

```python
isotropic:
  pool = MaxPool2d((2, 2)) at every level

freq_only_first2:
  pool1 = MaxPool2d((2, 1))
  pool2 = MaxPool2d((2, 1))
  pool3 = MaxPool2d((2, 2))

time_preserve:
  pool = MaxPool2d((2, 1)) at every level
```

Decoder upsampling must mirror the chosen pooling mode.

### 4.3 Update `train/training_uae.py`

Required new CLI/trainer arguments:

```text
--unet-variant
--input-domain                  mag | log1p_mag | real_imag | real_imag_mag
--output-mode                   mask_sigmoid | mask_scaled_sigmoid | mask_softplus | direct_mag | residual_mag | complex_crm | complex_stft
--mask-max
--softplus-max
--pooling-mode                  isotropic | freq_only_first2 | time_preserve
--loss-profile                  mag | mag_time | mag_time_mrstft | time_mrstft | complex_time | snr_aux
--loss-name                     mse | smoothl1 | huber | charbonnier | l1
--robust-beta
--time-loss-weight
--mrstft-loss-weight
--snr-loss-weight
--checkpoint-metric             val_snr | val_loss
--scheduler-metric              val_snr | val_loss
--min-epochs
--early-stop-patience
--disable-early-stop
--weight-decay
--grad-clip-norm
--hop-length
--noverlap
--save-every-epoch
```

Fix the current ignored `noverlap` behavior. The constructor currently accepts `noverlap` but overwrites it with `nperseg * 3 // 4`. It must become:

```python
self.noverlap = noverlap
```

or the argument must be removed everywhere. The experiment suite needs `hop_length`, so prefer:

```python
if hop_length is not None:
    self.noverlap = nperseg - hop_length
else:
    self.noverlap = noverlap
```

### 4.4 Checkpoint by SNR and keep loss checkpoint too

Replace single best-loss checkpointing with two checkpoints:

```python
if val_snr > best_val_snr + min_delta_snr:
    save model_best_snr.pth

if val_loss < best_val_loss - min_delta_loss:
    save model_best_loss.pth
```

For final reporting, load `model_best_snr.pth` by default.

### 4.5 Scheduler must support SNR mode

Required behavior:

```python
if scheduler_metric == "val_snr":
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=scheduler_patience,
        factor=scheduler_factor,
        threshold=scheduler_threshold,
        threshold_mode="abs",
        cooldown=scheduler_cooldown,
        min_lr=min_lr,
    )
    scheduler.step(val_snr)
else:
    scheduler = ReduceLROnPlateau(... mode="min" ...)
    scheduler.step(val_loss)
```

### 4.6 Add waveform-aware losses

Use the existing `multi_res_stft_loss()` function, but make it callable from the trainer.

Required loss profiles:

```text
mag:
  loss = spectral_mag_loss(out_mag, clean_mag)

mag_time:
  loss = spectral_mag_loss(out_mag, clean_mag)
       + time_loss_weight * mse(out_wave, clean_wave)

mag_time_mrstft:
  loss = spectral_mag_loss(out_mag, clean_mag)
       + time_loss_weight * mse(out_wave, clean_wave)
       + mrstft_loss_weight * multi_res_stft_loss(out_wave, clean_wave)

time_mrstft:
  loss = mse(out_wave, clean_wave)
       + mrstft_loss_weight * multi_res_stft_loss(out_wave, clean_wave)

complex_time:
  loss = complex_stft_loss(out_spec, clean_spec)
       + time_loss_weight * mse(out_wave, clean_wave)
       + mrstft_loss_weight * multi_res_stft_loss(out_wave, clean_wave)

snr_aux:
  loss = base_loss + snr_loss_weight * negative_snr_loss(out_wave, clean_wave)
```

Implement negative SNR loss as:

```python
def negative_snr_loss(pred, target, eps=1e-8):
    noise = pred - target
    signal_power = torch.mean(target ** 2, dim=1) + eps
    noise_power = torch.mean(noise ** 2, dim=1) + eps
    return -10.0 * torch.log10(signal_power / noise_power).mean()
```

### 4.7 Fix inference scripts or mark deprecated

`inference/inference_unet.py` currently treats the model output as denoised magnitude, but the current model returns a mask. Fix:

```python
mask = model(mag_tensor)
out_mag = (mask * mag_tensor).squeeze(0).squeeze(0).numpy()
```

`inference/inference_all_models.py` imports `models.autoencoder_unet_v2`, which is not in the provided archive. Either fix it to use the new variant-aware loader or mark the script deprecated at the top.

### 4.8 Do not break `compare_report.py`

The existing `compare_report.py` can evaluate the default U-Net. For variants, implement either:

1. `train/compare_unet_experiments.py`, independent from `compare_report.py`; or
2. a variant-aware U-Net loader used only when `model_config.json` exists.

Recommended: implement the separate U-Net experiment summarizer first. Do not risk breaking the general report.

---

## 5. New scripts to create

### 5.1 `train/unet_oracle_analysis.py`

Purpose: measure theoretical ceilings before training more models.

Command:

```bash
python train/unet_oracle_analysis.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --noise-types gaussian,non_gaussian \
  --nperseg 128 \
  --hop-length 32 \
  --batch-size 1024 \
  --output-dir data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749/runs/unet_oracle
```

Required outputs:

```text
oracle_summary.csv
oracle_summary.json
oracle_ratio_stats.csv
oracle_report.md
figures/oracle_snr_bar.png
```

Required oracle rows:

```text
input_noisy
oracle_clean_mag_noisy_phase
oracle_mask_0_1_noisy_phase
oracle_mask_0_2_noisy_phase
oracle_mask_0_3_noisy_phase
oracle_direct_clean_complex_stft
```

Also compute ratio statistics:

```text
fraction clean_mag / noisy_mag > 1
fraction clean_mag / noisy_mag > 2
fraction clean_mag / noisy_mag > 3
percentiles: 50, 75, 90, 95, 99, 99.9
```

Interpretation rules:

- If `oracle_clean_mag_noisy_phase` is near current U-Net SNR, magnitude-only U-Net has a phase ceiling; prioritize complex or 1D time-domain U-Net.
- If `oracle_mask_0_2_noisy_phase` is much better than `oracle_mask_0_1_noisy_phase`, mask range is a likely bottleneck.
- If `oracle_clean_mag_noisy_phase` is much higher than current U-Net, training/loss/architecture can still improve spectral U-Net.

### 5.2 `train/run_unet_experiment_suite.py`

Purpose: run all U-Net experiments in one controlled session.

Command:

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/core_v1.json \
  --noise-types non_gaussian \
  --epochs 50 \
  --seed 42 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

Required behavior:

- Create one parent run directory:

```text
runs/unet_ablation_<YYYYMMDD_HHMMSS>_<shortid>/
```

- Create one subdirectory per experiment:

```text
<exp_id>__<noise_type>__seed<seed>/
```

- Run each experiment as a separate Python subprocess to avoid GPU memory fragmentation.
- Continue after a failed experiment, but record the failure in `failures.jsonl`.
- After all experiments finish, automatically call `train/summarize_unet_experiments.py`.
- Save exact command line for every subprocess to `commands.jsonl`.

### 5.3 `train/summarize_unet_experiments.py`

Purpose: rank experiments and generate a compact report.

Command:

```bash
python train/summarize_unet_experiments.py \
  --run-dir data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749/runs/unet_ablation_<id> \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749
```

Required outputs:

```text
unet_ablation_summary.csv
unet_ablation_summary.json
unet_ablation_report.md
figures/summary_best_snr_bar.png
figures/summary_val_snr_curves.png
figures/summary_per_snr_curves.png
figures/summary_train_time_bar.png
```

The summary must include:

- ranking by NG->NG SNR;
- ranking by average cross-noise SNR;
- ranking by worst-case cross-noise SNR;
- delta vs fixed baseline;
- whether improvement is above the 0.3 dB threshold;
- training time and peak memory;
- notes on failed runs.

### 5.4 `train/unet_experiment_configs/core_v1.json`

Create a JSON experiment matrix file. See Section 7 for the first version.

### 5.5 Optional: `models/wave_unet_1d.py`

Purpose: test a time-domain U-Net while still keeping the model family as U-Net.

This is optional for the first suite if time is limited, but should be implemented if oracle analysis shows a strong phase ceiling.

---

## 6. Required trainer API design

The trainer should support a generic experiment config like this:

```json
{
  "id": "B01_mask2",
  "description": "Scaled sigmoid mask [0,2] with fixed trainer",
  "architecture": "spectral_unet",
  "input_domain": "mag",
  "output_mode": "mask_scaled_sigmoid",
  "mask_max": 2.0,
  "pooling_mode": "isotropic",
  "loss_profile": "mag",
  "loss_name": null,
  "nperseg": 128,
  "hop_length": 32,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "checkpoint_metric": "val_snr",
  "scheduler_metric": "val_snr",
  "min_epochs": 25,
  "early_stop_patience": 15
}
```

Do not hardcode experiment logic inside the trainer. The suite runner should pass arguments to the trainer.

---

## 7. Core experiment matrix

Run these experiments first on **non-Gaussian training** only. After ranking, rerun the top 3-5 on both Gaussian and non-Gaussian training for cross-evaluation.

### 7.1 Screening suite: `core_v1`

| ID | Name | What changes | Purpose | Expected signal |
|---|---|---|---|---|
| A00 | current_repro | Current model and current early-stop/loss-checkpoint behavior | Reproduce current behavior and verify early stop | May stop around 10-15 epochs |
| A01 | fixed_trainer | Current `[0,1]` mask, but 50 epochs, SNR checkpoint, SNR scheduler, longer patience | Isolate training-loop bug | If this improves, early stopping/checkpointing mattered |
| A02 | fixed_no_early_stop | Same as A01 but early stopping disabled | Test whether validation SNR rises late | Improvement after epoch 25 supports full training |
| B01 | mask2 | `mask = 2 * sigmoid(raw)` | Test if `[0,1]` mask is too restrictive | Better than A01 if clean/noisy mag ratio often > 1 |
| B02 | mask3 | `mask = 3 * sigmoid(raw)` | Test if more amplification helps or overfits | May help, but can create artifacts |
| B03 | softplus_mask3 | `mask = clamp(softplus(raw), max=3)` | Positive unbounded-style mask with cap | Often smoother than scaled sigmoid |
| B04 | residual_mag | `out_mag = clamp(noisy_mag + delta, min=0)` | Let model add/subtract magnitude directly | Good if ratio mask is awkward |
| C01 | mag_time | magnitude loss + waveform MSE | Align training with SNR | Should improve time-domain SNR |
| C02 | mag_time_mrstft | magnitude loss + waveform MSE + MR-STFT | Improve multi-scale spectral fidelity | Strong candidate for SNR/visual quality |
| D01 | hop16 | `nperseg=128`, `hop_length=16` | More time frames | Helps if bottleneck time resolution is limiting |
| D02 | aniso_pool | preserve time dimension longer with `(2,1)` pools | Reduce temporal bottleneck | Helps if time information was compressed too hard |
| E01 | log1p_mag | train on `log1p(mag)` or normalized log magnitude | Stabilize dynamic range | Helps if raw magnitude scale dominates loss |
| F01 | complex_crm | real/imag input, complex ratio mask output | Phase-aware spectral U-Net | Best spectral candidate if phase matters |
| G01 | waveunet1d_residual | 1D time-domain U-Net predicts clean or noise residual | Fairer U-Net vs Transformer | Best candidate if spectral phase ceiling is low |

Minimum viable suite if compute is limited:

```text
A01 fixed_trainer
B01 mask2
B04 residual_mag
C02 mag_time_mrstft
D02 aniso_pool
F01 complex_crm
G01 waveunet1d_residual
```

### 7.2 Proposed `core_v1.json`

```json
{
  "suite_name": "core_v1",
  "global_defaults": {
    "architecture": "spectral_unet",
    "input_domain": "mag",
    "output_mode": "mask_sigmoid",
    "mask_max": 1.0,
    "pooling_mode": "isotropic",
    "loss_profile": "mag",
    "loss_name": null,
    "nperseg": 128,
    "hop_length": 32,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "checkpoint_metric": "val_snr",
    "scheduler_metric": "val_snr",
    "scheduler_patience": 5,
    "scheduler_cooldown": 2,
    "scheduler_factor": 0.5,
    "scheduler_threshold": 0.02,
    "min_epochs": 25,
    "early_stop_patience": 15,
    "grad_clip_norm": 1.0,
    "time_loss_weight": 0.1,
    "mrstft_loss_weight": 0.05,
    "snr_loss_weight": 0.01
  },
  "experiments": [
    {
      "id": "A00_current_repro",
      "description": "Current behavior: loss checkpoint, loss scheduler, short early stop",
      "checkpoint_metric": "val_loss",
      "scheduler_metric": "val_loss",
      "scheduler_patience": 3,
      "min_epochs": 0,
      "early_stop_patience": 5,
      "weight_decay": 0.0,
      "optimizer": "adam"
    },
    {
      "id": "A01_fixed_trainer",
      "description": "Current U-Net architecture with fair SNR-based trainer"
    },
    {
      "id": "A02_fixed_no_early_stop",
      "description": "Current U-Net architecture, no early stopping for 50 epochs",
      "disable_early_stop": true
    },
    {
      "id": "B01_mask2",
      "description": "Scaled sigmoid mask in [0,2]",
      "output_mode": "mask_scaled_sigmoid",
      "mask_max": 2.0
    },
    {
      "id": "B02_mask3",
      "description": "Scaled sigmoid mask in [0,3]",
      "output_mode": "mask_scaled_sigmoid",
      "mask_max": 3.0
    },
    {
      "id": "B03_softplus_mask3",
      "description": "Softplus positive mask capped at 3",
      "output_mode": "mask_softplus",
      "softplus_max": 3.0
    },
    {
      "id": "B04_residual_mag",
      "description": "Predict magnitude residual instead of ratio mask",
      "output_mode": "residual_mag"
    },
    {
      "id": "C01_mag_time",
      "description": "Magnitude loss plus waveform MSE",
      "loss_profile": "mag_time",
      "time_loss_weight": 0.1
    },
    {
      "id": "C02_mag_time_mrstft",
      "description": "Magnitude loss plus waveform MSE plus multi-resolution STFT loss",
      "loss_profile": "mag_time_mrstft",
      "time_loss_weight": 0.1,
      "mrstft_loss_weight": 0.05
    },
    {
      "id": "D01_hop16",
      "description": "More STFT time frames with hop length 16",
      "hop_length": 16
    },
    {
      "id": "D02_aniso_pool",
      "description": "Preserve time resolution with anisotropic pooling",
      "pooling_mode": "freq_only_first2"
    },
    {
      "id": "E01_log1p_mag",
      "description": "Use log1p magnitude input/target normalization",
      "input_domain": "log1p_mag"
    },
    {
      "id": "F01_complex_crm",
      "description": "Phase-aware complex ratio mask U-Net",
      "architecture": "spectral_unet",
      "input_domain": "real_imag_mag",
      "output_mode": "complex_crm",
      "loss_profile": "complex_time",
      "time_loss_weight": 0.1,
      "mrstft_loss_weight": 0.05
    },
    {
      "id": "G01_waveunet1d_residual",
      "description": "1D time-domain U-Net residual denoiser",
      "architecture": "waveunet1d",
      "input_domain": "waveform",
      "output_mode": "waveform_residual",
      "loss_profile": "time_mrstft",
      "time_loss_weight": 1.0,
      "mrstft_loss_weight": 0.05
    }
  ]
}
```

---

## 8. Recommended execution plan

### Phase 0: quick validation

Run oracle analysis:

```bash
python train/unet_oracle_analysis.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --noise-types gaussian,non_gaussian \
  --nperseg 128 \
  --hop-length 32 \
  --output-dir data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749/runs/unet_oracle
```

Run shape tests on a tiny subset:

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/core_v1.json \
  --noise-types non_gaussian \
  --epochs 2 \
  --partial-train 0.005 \
  --seed 42 \
  --device cuda
```

Do not continue until every experiment produces the expected output shape `[B, signal_len]` and no NaNs.

### Phase 1: fast screen

Use a shorter run to remove broken or very slow variants:

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/core_v1.json \
  --noise-types non_gaussian \
  --epochs 15 \
  --partial-train 0.1 \
  --seed 42 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

Drop variants that are unstable, produce NaNs, are slower by more than 3x without improvement, or are clearly below baseline.

### Phase 2: full non-Gaussian suite

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/core_v1.json \
  --noise-types non_gaussian \
  --epochs 50 \
  --seed 42 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

Rank by NG->NG and NG->G.

### Phase 3: cross-evaluation of top variants

Create `top_v1.json` with top 3-5 variants and run both training noise types:

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/top_v1.json \
  --noise-types gaussian,non_gaussian \
  --epochs 50 \
  --seed 42 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

### Phase 4: repeatability check

For the best 1-2 variants, run seeds 42, 43, and 44:

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/best_repeatability_v1.json \
  --noise-types non_gaussian \
  --epochs 50 \
  --seeds 42,43,44 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

Report mean and standard deviation. A single lucky seed is not enough.

---

## 9. Detailed experiment requirements

### A00: current reproduction

Purpose: verify that the current behavior really early-stops before the LR reduction can help.

Settings:

```text
output_mode: mask_sigmoid
mask_max: 1.0
loss_profile: mag
optimizer: Adam
weight_decay: 0
scheduler_metric: val_loss
scheduler_patience: 3
early_stop_patience: 5
checkpoint_metric: val_loss
```

Required report fields:

```text
epochs completed
first epoch LR changed
number of epochs after LR change
best val loss epoch
best val SNR epoch
```

### A01: fixed trainer baseline

Purpose: establish the new baseline for all further comparisons.

Settings:

```text
same model as current U-Net
optimizer: AdamW
scheduler_metric: val_snr
checkpoint_metric: val_snr
min_epochs: 25
early_stop_patience: 15
epochs: 50
```

This experiment is the denominator for all deltas.

### A02: no early stopping

Purpose: directly test whether the model improves late.

Settings:

```text
same as A01
disable_early_stop: true
epochs: 50
```

If A02 beats A01, increase patience further or disable early stopping for future full runs.

### B01-B03: mask range

Purpose: test whether the suppressive `[0,1]` mask is too restrictive.

Required diagnostics:

- mean predicted mask;
- fraction predicted mask > 1 for B01/B02/B03;
- ratio statistics from oracle analysis;
- SNR by input-SNR bin.

Expected outcome:

- If B01 improves, use mask range > 1 as the new default.
- If B02/B03 improve but create unstable outputs, cap at 2 or add regularization.

### B04: residual magnitude prediction

Purpose: avoid ratio-mask limitations.

Implementation:

```python
raw = model_core(x)
out_mag = torch.clamp(noisy_mag + raw, min=0.0)
```

Add optional residual scaling:

```python
out_mag = torch.clamp(noisy_mag + residual_scale * raw, min=0.0)
```

Test `residual_scale = 1.0` first.

### C01-C02: waveform-aware losses

Purpose: align training objective with final SNR.

For non-complex spectral variants:

```python
phase = noisy_spec / (noisy_spec.abs() + 1e-8)
out_spec = out_mag * phase
out_wave = istft(out_spec)
```

Then compute waveform and MR-STFT losses.

Important: do not detach `out_wave`; gradients must flow through `torch.istft`.

### D01: hop length 16

Purpose: increase time frames and reduce temporal compression.

Settings:

```text
nperseg: 128
hop_length: 16
noverlap: 112
```

The trainer must not silently overwrite `noverlap`.

### D02: anisotropic pooling

Purpose: preserve time dimension at bottleneck.

Settings:

```text
pooling_mode: freq_only_first2
nperseg: 128
hop_length: 32
```

The upsampling path must mirror pooling. Validate output shape exactly matches input spectrogram shape.

### E01: log magnitude

Purpose: stabilize magnitude dynamic range.

Options:

```text
input: log1p(noisy_mag)
target: log1p(clean_mag)
model output: log magnitude or mask applied in linear magnitude
```

Preferred implementation for clarity:

```python
x = log1p(noisy_mag)
target = log1p(clean_mag)
pred_logmag = model(x)
out_mag = expm1(pred_logmag).clamp_min(0)
```

Do not combine log-domain training with mask output in the first version; keep interpretation simple.

### F01: complex ratio mask U-Net

Purpose: remove noisy-phase ceiling while staying spectral U-Net.

Input channels:

```text
channel 0: real(noisy_spec)
channel 1: imag(noisy_spec)
channel 2: log1p(abs(noisy_spec)) optional, recommended
```

Output channels:

```text
channel 0: real mask
channel 1: imag mask
```

Apply complex ratio mask:

```python
mr = raw[:, 0]
mi = raw[:, 1]
Yr = noisy_spec.real
Yi = noisy_spec.imag
out_real = mr * Yr - mi * Yi
out_imag = mr * Yi + mi * Yr
out_spec = torch.complex(out_real, out_imag)
out_wave = istft(out_spec)
```

Loss:

```text
complex L1 or MSE on real/imag STFT
+ waveform MSE
+ optional MR-STFT
```

Use gradient clipping because complex mask can amplify noise strongly.

### G01: 1D Wave-U-Net-style residual model

Purpose: compare U-Net architecture in the same time-domain information setting as Transformer.

Input/output:

```text
input: noisy waveform [B, 1, T]
output: predicted noise residual or clean waveform [B, 1, T]
recommended: predicted residual noise, clean_hat = noisy - predicted_noise
```

Architecture:

```text
Conv1d encoder blocks
strided Conv1d or pooling downsampling
skip connections
bottleneck with dilated Conv1d or small attention block
ConvTranspose1d or interpolate+Conv1d decoder
final Conv1d to 1 channel
```

Loss:

```text
waveform MSE + MR-STFT + optional negative SNR loss
```

This variant is still a U-Net, but it avoids STFT magnitude/noisy-phase limitations.

---

## 10. Evaluation and comparison rules

### 10.1 Primary table

The final summary report must include this table:

| Rank | Exp ID | Train noise | Test G SNR | Test NG SNR | Mean | Worst | Delta vs A01 NG | Epoch best | Time | Peak VRAM | Notes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

### 10.2 Per-SNR curves

Plot at least:

- A01 fixed trainer baseline;
- best mask-range variant;
- best loss variant;
- best pooling/STFT variant;
- complex CRM if implemented;
- 1D Wave-U-Net if implemented.

### 10.3 Decision logic

Use the following rules:

1. If A01 beats A00 materially, keep the new trainer protocol permanently.
2. If A02 beats A01, disable early stopping or increase patience/min_epochs.
3. If B01/B02/B03 beat A01, change default mask from `[0,1]` to the best wider mask.
4. If B04 beats B01/B02, prefer residual magnitude over ratio mask.
5. If C02 beats all B variants, make waveform-aware loss the new default.
6. If D01/D02 beat A01 but not C02, combine the best D variant with C02 in a second-stage experiment.
7. If F01 beats all magnitude-only variants, prioritize complex spectral U-Net.
8. If G01 beats F01, prioritize time-domain U-Net for the U-Net family.
9. If oracle clean-mag/noisy-phase is below old Transformer score, do not spend many iterations on magnitude-only U-Net.

---

## 11. Second-stage combination experiments

After `core_v1`, build `combo_v1.json` using only winning ingredients.

Likely combinations:

| ID | Combination |
|---|---|
| X01 | best mask output + waveform-aware loss |
| X02 | best mask output + anisotropic pooling |
| X03 | best mask output + hop16 |
| X04 | best mask output + waveform-aware loss + anisotropic pooling |
| X05 | residual magnitude + waveform-aware loss |
| X06 | complex CRM + anisotropic pooling |
| X07 | complex CRM + MR-STFT |
| X08 | 1D Wave-U-Net + negative SNR auxiliary loss |

Do not run all possible combinations blindly. Use only changes that showed positive individual signal.

---

## 12. Code quality and test requirements

### 12.1 Unit tests / smoke tests

Add or run smoke tests for:

```text
model forward shape for every output_mode
STFT -> iSTFT length preservation
no NaNs in loss for one batch
checkpoint save/load for every architecture
variant loader reconstructs model from model_config.json
summary script reads all result files
```

### 12.2 NaN handling

If NaNs occur:

- stop that experiment;
- save the failing batch index if possible;
- save last finite metrics;
- record failure in `failures.jsonl`;
- continue the suite.

### 12.3 W&B logging

Log these keys consistently:

```text
train/loss
val/loss
val/snr_db
train/lr
train/epoch_time_sec
train/peak_vram_gb
test/snr_db
test/mse
test/mae
test/rmse
```

Do not log `train/mse_loss` when the actual loss is SmoothL1, Charbonnier, or a composite loss.

---

## 13. Suggested implementation order for agents

1. Fix trainer reliability: SNR checkpointing, SNR scheduler, early-stop settings, `noverlap` handling, metric CSV.
2. Add `unet_oracle_analysis.py`.
3. Add suite runner and summarizer.
4. Implement output modes: `mask_scaled_sigmoid`, `mask_softplus`, `residual_mag`.
5. Implement waveform-aware losses.
6. Implement anisotropic pooling and hop-length support.
7. Implement log-magnitude experiment.
8. Implement complex CRM.
9. Implement 1D Wave-U-Net if oracle analysis or core suite indicates phase/time-domain limitation.
10. Fix or deprecate stale inference scripts.
11. Run Phase 0 smoke tests.
12. Run Phase 1 fast screen.
13. Run Phase 2 full non-Gaussian suite.
14. Run Phase 3 cross-evaluation for top variants.
15. Run Phase 4 repeatability for best variant.

---

## 14. Acceptance criteria

The development task is complete when:

1. `train/unet_oracle_analysis.py` runs and produces oracle report.
2. `train/run_unet_experiment_suite.py` can execute the core suite without manual intervention.
3. `train/summarize_unet_experiments.py` produces CSV, JSON, Markdown report, and figures.
4. At least the minimum viable suite runs successfully for non-Gaussian training.
5. The report identifies the best U-Net variant and the delta versus A01 fixed baseline.
6. The best variant is cross-evaluated on both Gaussian and non-Gaussian test sets.
7. All runs save enough config to reproduce them exactly.
8. Default `UnetAutoencoder` remains compatible with existing baseline use.
9. No Transformer, Wavelet, or Hybrid code is changed except optional fixed-baseline evaluation scripts.

---

## 15. Expected outcomes and interpretation

Potential outcomes:

### Outcome 1: fixed trainer alone gives most of the gain

Conclusion: the current U-Net was undertrained. Keep current architecture but change scheduler/checkpoint/early-stop permanently.

### Outcome 2: wider mask gives clear gain

Conclusion: the `[0,1]` suppressive mask is too restrictive. Use `[0,2]`, `[0,3]`, softplus mask, or residual magnitude as default.

### Outcome 3: waveform/MR-STFT loss gives clear gain

Conclusion: magnitude loss was misaligned with time-domain SNR. Keep spectral U-Net but train with waveform-aware loss.

### Outcome 4: complex CRM dominates magnitude variants

Conclusion: noisy phase was a major ceiling. Continue with phase-aware spectral U-Net.

### Outcome 5: 1D Wave-U-Net dominates spectral U-Net

Conclusion: U-Net can compete better when it has time-domain phase/timing information like Transformer.

### Outcome 6: all variants remain far below Transformer

Conclusion: the Transformer advantage is likely due to global time-domain modeling. Next U-Net-only direction should be 1D U-Net with dilated/attention bottleneck, not more magnitude-mask tuning.

---

## 16. Final deliverables for the experiment run

After running the suite, produce:

```text
unet_ablation_report.md
unet_ablation_summary.csv
unet_ablation_summary.json
oracle_report.md
best_variant_config.json
best_variant_model_best_snr.pth
figures/*.png
```

The final report should answer:

1. Did longer training fix the problem?
2. Did SNR-based checkpointing change the selected model?
3. Is `[0,1]` mask range a bottleneck?
4. Is noisy phase a bottleneck?
5. Does waveform-aware loss improve SNR?
6. Does preserving time resolution improve SNR?
7. Which U-Net variant should become the new default?
8. How close is the best U-Net to the old Transformer baseline?
9. Which experiment should be repeated with more seeds?
