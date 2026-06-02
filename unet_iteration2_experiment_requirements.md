# U-Net Denoising Experiments — Iteration 2 Requirements

**Project:** signal denoising for QPSK-like deep-space signals  
**Target model family:** U-Net only  
**Dataset:** `deep_space_polygauss_qpsk_bs1024_n400000_c054e749`  
**Purpose:** combine the strongest single-change U-Net results from iteration 1 and test a cleaner U-Net infrastructure that can support spectral, complex spectral, and 1D time-domain U-Net variants without breaking evaluation.

---

## 1. Current evidence from iteration 1

### 1.1 Main result

The strongest result in the latest cross-evaluation report is:

| Candidate | Train noise | Test Gaussian | Test Non-Gaussian | Mean | Worst |
|---|---|---:|---:|---:|---:|
| `A01 fixed_trainer` | Non-Gaussian | 6.77 dB | 7.61 dB | 7.19 dB | 6.77 dB |
| `B01 mask2` | Non-Gaussian | 7.29 dB | 8.16 dB | 7.725 dB | 7.29 dB |
| `B02 mask3` | Non-Gaussian | **7.30 dB** | **8.18 dB** | **7.74 dB** | **7.30 dB** |
| `B03 softplus_mask3` | Non-Gaussian | 7.21 dB | 8.08 dB | 7.645 dB | 7.21 dB |

Decision: **`B02 mask3` becomes the iteration-2 spectral U-Net baseline.**

### 1.2 What the results imply

1. **Wider masks are the only clearly positive individual change.**  
   `B01`, `B02`, and `B03` all beat `A01` by roughly `+0.47` to `+0.57 dB` on NG→NG. This is above the previous `+0.3 dB` promising-change threshold.

2. **`B02 mask3` is best for Non-Gaussian-trained ranking.**  
   It gives the best NG→NG score and the best NG→G score in the latest comparison table.

3. **`B03 softplus_mask3` is still important.**  
   It is slightly weaker than `B02` for Non-Gaussian-trained testing, but it is strongest among Gaussian-trained variants: `G→G = 7.18 dB`, `G→NG = 7.84 dB`. It may generalize better when the train distribution is easier.

4. **Early stopping is no longer the main bottleneck.**  
   `A01 fixed_trainer` and `A02 fixed_no_early_stop` have identical cross-evaluation numbers. Keep the fixed trainer, but do not spend iteration 2 on early-stop-only tests.

5. **Standalone waveform-aware losses did not help with the `[0,1]` mask.**  
   `C01` and `C02` are approximately equal to or slightly worse than `A01`. However, this does not prove that waveform-aware losses are bad; they should be retested only when combined with the winning wider mask.

6. **Temporal-resolution changes were weak alone but not useless.**  
   `D01 hop16` and `D02 aniso_pool` are only marginally better than `A01` when used alone. They are worth testing only in combination with `B02` or the new residual/complex/time-domain infrastructures.

7. **`B04 residual_mag` should not be part of the first combo set.**  
   It ranked high by validation SNR in the ablation report, but cross-evaluation shows worse test results than `A01` in several cells. Treat it as unstable/misaligned until diagnostics explain the gap.

8. **`F01 complex_crm` failed and must be treated as an infrastructure/debug result, not as evidence that complex U-Net is bad.**  
   The ablation report labels `F01_complex_crm` with loss `mag`, which is suspicious for a complex-output model. A correct complex U-Net must use complex STFT and waveform losses, identity-safe initialization, and bounded/regularized complex masks.

---

## 2. Iteration 2 goals

### 2.1 Primary goals

1. Test whether the best single-change result, `B02 mask3`, improves further when combined with:
   - waveform-aware loss;
   - multi-resolution STFT loss;
   - hop length 16;
   - anisotropic pooling;
   - SNR auxiliary loss;
   - stronger U-Net blocks.

2. Build a clean U-Net infrastructure that supports:
   - spectral magnitude-mask U-Net;
   - phase-aware complex spectral U-Net;
   - 1D time-domain Wave-U-Net-style residual denoiser.

3. Keep all comparisons reproducible and directly comparable to iteration 1.

### 2.2 Non-goals

1. Do not modify Transformer, Wavelet, Hybrid, or dataset generation code.
2. Do not re-run every possible combination. Iteration 2 should be targeted.
3. Do not use `F01 complex_crm` result as a final architecture conclusion until the complex pipeline passes identity/oracle smoke tests.
4. Do not make a new model the default unless it beats `B02 mask3` by at least `+0.2 dB` on NG→NG and does not degrade NG→G by more than `0.1 dB`.

---

## 3. Fixed protocol for all iteration-2 experiments

Unless explicitly changed by an experiment:

```text
noise_type: non_gaussian for first full screen
signal_len: from dataset_config.json, expected 1024
nperseg: 128
hop_length: 32
noverlap: 96
batch_size: 512, or highest stable GPU value
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

Important: `B02 mask3` is now the main comparison baseline for iteration 2, while `A01 fixed_trainer` remains a historical reference.

---

## 4. Metrics and decision thresholds

Each experiment must report:

```text
best validation SNR
validation loss at best SNR
test SNR on same train noise type
cross-test SNR on opposite noise type
mean of NG->G and NG->NG
worst of NG->G and NG->NG
MSE, MAE, RMSE
epoch of best SNR
epoch of best loss
final LR
training time
peak GPU memory
NaN/failure status
```

### 4.1 Ranking metrics

Primary ranking:

```text
NG train -> NG test SNR
```

Secondary ranking:

```text
NG train -> G test SNR
mean(NG->G, NG->NG)
worst(NG->G, NG->NG)
```

### 4.2 Decision thresholds

| Decision | Rule |
|---|---|
| Promising vs `A01` | `NG->NG >= A01 + 0.3 dB` and `NG->G >= A01 - 0.2 dB` |
| Promising vs `B02` | `NG->NG >= B02 + 0.2 dB` and `NG->G >= B02 - 0.1 dB` |
| New default candidate | beats `B02` on mean and worst-case cross-noise score |
| Major improvement | `NG->NG >= B02 + 0.7 dB` |
| Repeatability required | any candidate above `B02 + 0.2 dB` must be rerun with seeds `42,43,44` |
| Reject | NaNs, unstable masks/specs, or >2x training time with no SNR gain |

---

## 5. Required infrastructure changes before running iteration 2

### 5.1 Introduce a variant-safe U-Net interface

Create a consistent return contract for all U-Net variants.

Recommended output dictionary:

```python
{
    "out_wave": Tensor | None,   # [B, T]
    "out_spec": Tensor | None,   # [B, F, TT], complex if spectral
    "out_mag": Tensor | None,    # [B, 1, F, TT]
    "mask": Tensor | None,       # mask if applicable
    "debug": dict,
}
```

Required behavior:

1. Evaluation should always use `out_wave`.
2. Spectral magnitude models may internally compute `out_wave` by applying noisy phase and `torch.istft`.
3. Complex models must compute `out_wave` from predicted complex STFT.
4. 1D models directly output `out_wave` or residual-clean reconstruction.
5. No trainer code should guess whether the model output is a mask, magnitude, complex mask, or waveform.

### 5.2 Add a U-Net variant registry

Create:

```text
models/unet_registry.py
```

Required API:

```python
def build_unet_variant(model_config: dict) -> torch.nn.Module:
    ...
```

Supported `architecture` values:

```text
spectral_unet
spectral_resunet
complex_crm_unet
complex_stft_unet
waveunet1d
```

Every run must save:

```text
model_config.json
experiment_config.json
training_args.json
```

The evaluator must reconstruct the model only from `model_config.json`.

### 5.3 Add a shared STFT/iSTFT module

Create:

```text
models/stft_projector.py
```

Required API:

```python
class STFTProjector(nn.Module):
    def stft(self, x: torch.Tensor) -> torch.Tensor: ...
    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor: ...
    def mag_phase(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
```

Requirements:

1. Use `torch.stft` and `torch.istft` only.
2. Use the same `nperseg`, `hop_length`, `win_length`, `center`, and Hann window in train and eval.
3. Preserve signal length exactly.
4. Add tests for `istft(stft(x))` reconstruction error.

### 5.4 Add a loss factory

Create:

```text
train/unet_losses.py
```

Required loss profiles:

```text
mag
mag_time
time_mrstft
mag_time_mrstft
mag_time_mrstft_snr
complex_time
complex_time_mrstft
waveform_time_mrstft
waveform_time_mrstft_snr
```

All composite losses must log components separately:

```text
train/loss_total
train/loss_mag
train/loss_time
train/loss_mrstft
train/loss_snr
train/loss_complex
```

### 5.5 Add mask/spec diagnostics

Every spectral experiment must record these diagnostics on the validation set:

```text
mask_mean
mask_std
mask_p50, mask_p90, mask_p99
fraction_mask_gt_1
fraction_mask_gt_2
fraction_mask_gt_3
out_mag_max
out_mag_p99
waveform_peak_abs
nan_count
inf_count
```

For complex experiments also record:

```text
crm_real_mean, crm_real_std
crm_imag_mean, crm_imag_std
crm_abs_p50, crm_abs_p90, crm_abs_p99
fraction_crm_abs_gt_2
fraction_crm_abs_gt_5
out_spec_abs_p99
```

### 5.6 Add separate U-Net experiment evaluator

Create or update:

```text
train/compare_unet_experiments.py
```

Do not break the general `compare_report.py`.

Required outputs:

```text
iteration2_summary.csv
iteration2_summary.json
iteration2_report.md
figures/iteration2_snr_bar.png
figures/iteration2_cross_noise_scatter.png
figures/iteration2_per_snr_curves.png
figures/iteration2_mask_diagnostics.png
```

---

## 6. Iteration-2 experiment suite A: combination tests

File to create:

```text
train/unet_experiment_configs/combo_v2.json
```

### 6.1 Experiment matrix

| ID | Purpose | Changes from `B02 mask3` | Priority |
|---|---|---|---:|
| `M00_B02_repro` | Confirm new baseline in current code | `mask_scaled_sigmoid`, `mask_max=3`, mag loss | P0 |
| `M01_B01_mask2_repro` | Check if cap 2 is as good but safer | `mask_max=2` | P1 |
| `M02_B03_softplus_repro` | Keep softplus candidate alive | `mask_softplus`, `softplus_max=3` | P1 |
| `X01_mask3_time003` | Test light waveform alignment | B02 + waveform MSE, `time_loss_weight=0.03` | P0 |
| `X02_mask3_time010` | Test previous waveform weight with winning mask | B02 + waveform MSE, `time_loss_weight=0.10` | P1 |
| `X03_mask3_time003_mrstft001` | Conservative MR-STFT combo | B02 + time 0.03 + MR-STFT 0.01 | P0 |
| `X04_mask3_time005_mrstft002` | Medium MR-STFT combo | B02 + time 0.05 + MR-STFT 0.02 | P1 |
| `X05_mask3_snr_aux001` | Direct metric alignment | B02 + negative SNR aux 0.01 | P1 |
| `X06_mask3_hop16` | Combine best mask with more time frames | B02 + `hop_length=16` | P0 |
| `X07_mask3_aniso` | Combine best mask with time-preserving pooling | B02 + `pooling_mode=freq_only_first2` | P0 |
| `X08_mask3_hop16_aniso` | Combine both temporal-resolution changes | B02 + hop16 + anisotropic pooling | P1 |
| `X09_mask3_aniso_mrstft_low` | Best mask + temporal preservation + gentle loss | B02 + aniso + time 0.03 + MR-STFT 0.01 | P1 |
| `X10_softplus3_time003_mrstft001` | Test smoother mask with composite loss | B03 + time 0.03 + MR-STFT 0.01 | P1 |
| `X11_softplus3_aniso` | Test smoother mask with preserved time | B03 + anisotropic pooling | P2 |
| `X12_mask2_aniso` | Safer mask cap plus temporal preservation | B01 + anisotropic pooling | P2 |

Do not run `residual_mag` combinations in the first full iteration-2 run unless compute is abundant. Its validation ranking looked promising, but cross-evaluation did not.

### 6.2 Proposed `combo_v2.json`

```json
{
  "suite_name": "combo_v2",
  "description": "Iteration-2 combinations around B02 mask3 baseline",
  "global_defaults": {
    "architecture": "spectral_unet",
    "input_domain": "mag",
    "output_mode": "mask_scaled_sigmoid",
    "mask_max": 3.0,
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
    "time_loss_weight": 0.03,
    "mrstft_loss_weight": 0.01,
    "snr_loss_weight": 0.01
  },
  "experiments": [
    {
      "id": "M00_B02_repro",
      "description": "B02 mask3 baseline: scaled sigmoid mask [0,3] with magnitude loss"
    },
    {
      "id": "M01_B01_mask2_repro",
      "description": "B01 mask2 reproduction: scaled sigmoid mask [0,2]",
      "mask_max": 2.0
    },
    {
      "id": "M02_B03_softplus_repro",
      "description": "B03 softplus mask capped at 3",
      "output_mode": "mask_softplus",
      "softplus_max": 3.0
    },
    {
      "id": "X01_mask3_time003",
      "description": "B02 plus light waveform MSE",
      "loss_profile": "mag_time",
      "time_loss_weight": 0.03
    },
    {
      "id": "X02_mask3_time010",
      "description": "B02 plus stronger waveform MSE",
      "loss_profile": "mag_time",
      "time_loss_weight": 0.10
    },
    {
      "id": "X03_mask3_time003_mrstft001",
      "description": "B02 plus conservative waveform and MR-STFT loss",
      "loss_profile": "mag_time_mrstft",
      "time_loss_weight": 0.03,
      "mrstft_loss_weight": 0.01
    },
    {
      "id": "X04_mask3_time005_mrstft002",
      "description": "B02 plus medium waveform and MR-STFT loss",
      "loss_profile": "mag_time_mrstft",
      "time_loss_weight": 0.05,
      "mrstft_loss_weight": 0.02
    },
    {
      "id": "X05_mask3_snr_aux001",
      "description": "B02 plus negative SNR auxiliary loss",
      "loss_profile": "mag_time_mrstft_snr",
      "time_loss_weight": 0.03,
      "mrstft_loss_weight": 0.01,
      "snr_loss_weight": 0.01
    },
    {
      "id": "X06_mask3_hop16",
      "description": "B02 with hop length 16",
      "hop_length": 16
    },
    {
      "id": "X07_mask3_aniso",
      "description": "B02 with anisotropic pooling preserving time",
      "pooling_mode": "freq_only_first2"
    },
    {
      "id": "X08_mask3_hop16_aniso",
      "description": "B02 with hop length 16 and anisotropic pooling",
      "hop_length": 16,
      "pooling_mode": "freq_only_first2"
    },
    {
      "id": "X09_mask3_aniso_mrstft_low",
      "description": "B02 with anisotropic pooling and conservative composite loss",
      "pooling_mode": "freq_only_first2",
      "loss_profile": "mag_time_mrstft",
      "time_loss_weight": 0.03,
      "mrstft_loss_weight": 0.01
    },
    {
      "id": "X10_softplus3_time003_mrstft001",
      "description": "Softplus mask capped at 3 with conservative composite loss",
      "output_mode": "mask_softplus",
      "softplus_max": 3.0,
      "loss_profile": "mag_time_mrstft",
      "time_loss_weight": 0.03,
      "mrstft_loss_weight": 0.01
    },
    {
      "id": "X11_softplus3_aniso",
      "description": "Softplus mask capped at 3 with anisotropic pooling",
      "output_mode": "mask_softplus",
      "softplus_max": 3.0,
      "pooling_mode": "freq_only_first2"
    },
    {
      "id": "X12_mask2_aniso",
      "description": "Scaled sigmoid mask [0,2] with anisotropic pooling",
      "mask_max": 2.0,
      "pooling_mode": "freq_only_first2"
    }
  ]
}
```

---

## 7. Iteration-2 experiment suite B: new U-Net infrastructure

File to create:

```text
train/unet_experiment_configs/infra_v2.json
```

This suite is not only about immediate SNR. It must prove that the new architecture infrastructure is correct, stable, and easy to compare.

### 7.1 Spectral ResU-Net variants

Purpose: test whether better U-Net blocks improve over B02 without changing the core spectral magnitude-mask approach.

| ID | Architecture | Changes | Priority |
|---|---|---|---:|
| `I01_resunet_mask3` | `spectral_resunet` | residual conv blocks, mask3, mag loss | P0 |
| `I02_resunet_dilated_bottleneck_mask3` | `spectral_resunet` | residual blocks + dilated bottleneck rates `[1,2,4]` | P1 |
| `I03_resunet_attention_skip_mask3` | `spectral_resunet` | residual blocks + attention gates on skip paths | P2 |
| `I04_resunet_aniso_mask3` | `spectral_resunet` | residual blocks + anisotropic pooling | P1 |
| `I05_resunet_aniso_mrstft_mask3` | `spectral_resunet` | residual blocks + aniso + conservative composite loss | P2 |

Requirements:

1. Keep parameter count logged.
2. Keep output mode as `mask_scaled_sigmoid`, `mask_max=3`.
3. Use GroupNorm or BatchNorm consistently; if batch size is high, BatchNorm is acceptable, but GroupNorm should be available.
4. Do not increase depth beyond 3 until temporal bottleneck tests are complete.

### 7.2 Complex spectral U-Net repair and debug

The previous `F01 complex_crm` result was very poor. Treat this as a failed implementation or loss setup until the following tests pass.

#### Required complex smoke tests

Create:

```text
tests/test_complex_unet_pipeline.py
```

Required tests:

1. **Identity reconstruction test**  
   If CRM is initialized as `1 + 0j`, then output waveform should match noisy waveform after STFT/iSTFT within tolerance.

2. **Zero-output prevention test**  
   With zero-initialized final layer, the CRM model must return identity/noisy reconstruction, not silence.

3. **Oracle CRM test**  
   Compute target CRM from `clean_spec / (noisy_spec + eps)` and verify applying it reconstructs clean STFT with high SNR on a small batch.

4. **Loss-gradient test**  
   One forward/backward pass with `complex_time_mrstft` must produce finite gradients.

5. **Shape test**  
   Model output spec and waveform must match `[B, F, TT]` and `[B, signal_len]` exactly.

#### Safe CRM parameterization

Use bounded residual CRM around identity:

```python
mr = 1.0 + crm_scale * torch.tanh(raw_real)
mi =       crm_scale * torch.tanh(raw_imag)
crm = torch.complex(mr, mi)
out_spec = crm * noisy_spec
```

Start with:

```text
crm_scale: 0.5
```

Then test:

```text
crm_scale: 1.0
```

Avoid unconstrained raw complex masks in the first repaired experiment.

#### Complex experiments

| ID | Architecture | Changes | Priority |
|---|---|---|---:|
| `C10_crm_identity_safe_scale05` | `complex_crm_unet` | bounded residual CRM, scale 0.5, complex_time loss | P0 |
| `C11_crm_identity_safe_scale10` | `complex_crm_unet` | bounded residual CRM, scale 1.0, complex_time loss | P1 |
| `C12_crm_scale05_mrstft` | `complex_crm_unet` | C10 + MR-STFT weight 0.01 | P1 |
| `C13_complex_stft_direct` | `complex_stft_unet` | predict clean real/imag STFT directly | P2 |

Reject complex experiments if any smoke test fails. Do not include failed complex results in the main ranking table except under a separate `debug failures` section.

### 7.3 1D Wave-U-Net-style variants

Purpose: give U-Net the same time-domain signal access as the Transformer while staying inside the U-Net family.

| ID | Architecture | Changes | Priority |
|---|---|---|---:|
| `W01_waveunet1d_residual_small` | `waveunet1d` | residual noise prediction, base channels 32 | P0 |
| `W02_waveunet1d_residual_dilated` | `waveunet1d` | W01 + dilated bottleneck `[1,2,4,8]` | P1 |
| `W03_waveunet1d_residual_snr_aux` | `waveunet1d` | W02 + negative SNR aux 0.01 | P1 |
| `W04_waveunet1d_attention_bottleneck` | `waveunet1d` | W02 + small bottleneck self-attention or gated TCN | P2 |

Recommended output:

```python
pred_noise = model(noisy_wave[:, None, :]).squeeze(1)
out_wave = noisy_wave - pred_noise
```

Recommended loss:

```text
waveform_time_mrstft
```

with:

```text
time_loss_weight: 1.0
mrstft_loss_weight: 0.05
snr_loss_weight: 0.0 initially
```

Add `snr_loss_weight=0.01` only in `W03`.

---

## 8. Proposed `infra_v2.json`

```json
{
  "suite_name": "infra_v2",
  "description": "Iteration-2 infrastructure suite: ResU-Net, repaired complex U-Net, and Wave-U-Net",
  "global_defaults": {
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
    "grad_clip_norm": 1.0
  },
  "experiments": [
    {
      "id": "I01_resunet_mask3",
      "architecture": "spectral_resunet",
      "input_domain": "mag",
      "output_mode": "mask_scaled_sigmoid",
      "mask_max": 3.0,
      "pooling_mode": "isotropic",
      "loss_profile": "mag",
      "base_channels": 32,
      "residual_blocks": true
    },
    {
      "id": "I02_resunet_dilated_bottleneck_mask3",
      "architecture": "spectral_resunet",
      "input_domain": "mag",
      "output_mode": "mask_scaled_sigmoid",
      "mask_max": 3.0,
      "pooling_mode": "isotropic",
      "loss_profile": "mag",
      "base_channels": 32,
      "residual_blocks": true,
      "bottleneck": "dilated",
      "dilation_rates": [1, 2, 4]
    },
    {
      "id": "I03_resunet_attention_skip_mask3",
      "architecture": "spectral_resunet",
      "input_domain": "mag",
      "output_mode": "mask_scaled_sigmoid",
      "mask_max": 3.0,
      "pooling_mode": "isotropic",
      "loss_profile": "mag",
      "base_channels": 32,
      "residual_blocks": true,
      "skip_attention": true
    },
    {
      "id": "I04_resunet_aniso_mask3",
      "architecture": "spectral_resunet",
      "input_domain": "mag",
      "output_mode": "mask_scaled_sigmoid",
      "mask_max": 3.0,
      "pooling_mode": "freq_only_first2",
      "loss_profile": "mag",
      "base_channels": 32,
      "residual_blocks": true
    },
    {
      "id": "I05_resunet_aniso_mrstft_mask3",
      "architecture": "spectral_resunet",
      "input_domain": "mag",
      "output_mode": "mask_scaled_sigmoid",
      "mask_max": 3.0,
      "pooling_mode": "freq_only_first2",
      "loss_profile": "mag_time_mrstft",
      "time_loss_weight": 0.03,
      "mrstft_loss_weight": 0.01,
      "base_channels": 32,
      "residual_blocks": true
    },
    {
      "id": "C10_crm_identity_safe_scale05",
      "architecture": "complex_crm_unet",
      "input_domain": "real_imag_mag",
      "output_mode": "complex_crm_identity_residual",
      "crm_scale": 0.5,
      "pooling_mode": "isotropic",
      "loss_profile": "complex_time",
      "time_loss_weight": 0.1,
      "mrstft_loss_weight": 0.0,
      "grad_clip_norm": 0.5
    },
    {
      "id": "C11_crm_identity_safe_scale10",
      "architecture": "complex_crm_unet",
      "input_domain": "real_imag_mag",
      "output_mode": "complex_crm_identity_residual",
      "crm_scale": 1.0,
      "pooling_mode": "isotropic",
      "loss_profile": "complex_time",
      "time_loss_weight": 0.1,
      "mrstft_loss_weight": 0.0,
      "grad_clip_norm": 0.5
    },
    {
      "id": "C12_crm_scale05_mrstft",
      "architecture": "complex_crm_unet",
      "input_domain": "real_imag_mag",
      "output_mode": "complex_crm_identity_residual",
      "crm_scale": 0.5,
      "pooling_mode": "isotropic",
      "loss_profile": "complex_time_mrstft",
      "time_loss_weight": 0.1,
      "mrstft_loss_weight": 0.01,
      "grad_clip_norm": 0.5
    },
    {
      "id": "C13_complex_stft_direct",
      "architecture": "complex_stft_unet",
      "input_domain": "real_imag_mag",
      "output_mode": "complex_stft",
      "pooling_mode": "isotropic",
      "loss_profile": "complex_time_mrstft",
      "time_loss_weight": 0.1,
      "mrstft_loss_weight": 0.01,
      "grad_clip_norm": 0.5
    },
    {
      "id": "W01_waveunet1d_residual_small",
      "architecture": "waveunet1d",
      "input_domain": "waveform",
      "output_mode": "waveform_residual_noise",
      "base_channels": 32,
      "depth": 4,
      "loss_profile": "waveform_time_mrstft",
      "time_loss_weight": 1.0,
      "mrstft_loss_weight": 0.05
    },
    {
      "id": "W02_waveunet1d_residual_dilated",
      "architecture": "waveunet1d",
      "input_domain": "waveform",
      "output_mode": "waveform_residual_noise",
      "base_channels": 32,
      "depth": 4,
      "bottleneck": "dilated",
      "dilation_rates": [1, 2, 4, 8],
      "loss_profile": "waveform_time_mrstft",
      "time_loss_weight": 1.0,
      "mrstft_loss_weight": 0.05
    },
    {
      "id": "W03_waveunet1d_residual_snr_aux",
      "architecture": "waveunet1d",
      "input_domain": "waveform",
      "output_mode": "waveform_residual_noise",
      "base_channels": 32,
      "depth": 4,
      "bottleneck": "dilated",
      "dilation_rates": [1, 2, 4, 8],
      "loss_profile": "waveform_time_mrstft_snr",
      "time_loss_weight": 1.0,
      "mrstft_loss_weight": 0.05,
      "snr_loss_weight": 0.01
    },
    {
      "id": "W04_waveunet1d_attention_bottleneck",
      "architecture": "waveunet1d",
      "input_domain": "waveform",
      "output_mode": "waveform_residual_noise",
      "base_channels": 32,
      "depth": 4,
      "bottleneck": "attention_tcn",
      "loss_profile": "waveform_time_mrstft",
      "time_loss_weight": 1.0,
      "mrstft_loss_weight": 0.05
    }
  ]
}
```

---

## 9. Minimum viable run if compute is limited

Run these first:

```text
M00_B02_repro
X03_mask3_time003_mrstft001
X06_mask3_hop16
X07_mask3_aniso
X09_mask3_aniso_mrstft_low
I01_resunet_mask3
I02_resunet_dilated_bottleneck_mask3
C10_crm_identity_safe_scale05
W01_waveunet1d_residual_small
W02_waveunet1d_residual_dilated
```

This gives coverage of:

```text
best current mask
best mask + loss
best mask + temporal resolution
new 2D ResU-Net
repaired complex U-Net
1D time-domain U-Net
```

---

## 10. Execution plan

### Phase 0 — smoke tests only

```bash
python -m pytest tests/test_unet_variants.py tests/test_complex_unet_pipeline.py -q
```

Then run all experiments for 2 epochs on a tiny subset:

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/combo_v2.json \
  --noise-types non_gaussian \
  --epochs 2 \
  --partial-train 0.005 \
  --seed 42 \
  --device cuda
```

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/infra_v2.json \
  --noise-types non_gaussian \
  --epochs 2 \
  --partial-train 0.005 \
  --seed 42 \
  --device cuda
```

Do not continue until every experiment outputs `[B, signal_len]`, saves configs, and has finite losses.

### Phase 1 — fast screen

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/combo_v2.json \
  --noise-types non_gaussian \
  --epochs 15 \
  --partial-train 0.1 \
  --seed 42 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/infra_v2.json \
  --noise-types non_gaussian \
  --epochs 15 \
  --partial-train 0.1 \
  --seed 42 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

Drop experiments that are unstable, slower than 2x without a validation SNR signal, or below `M00_B02_repro` by more than `0.5 dB` after the fast screen.

### Phase 2 — full Non-Gaussian screen

Create `combo_v2_selected.json` and `infra_v2_selected.json` from the fast-screen winners, then run:

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/combo_v2_selected.json \
  --noise-types non_gaussian \
  --epochs 50 \
  --seed 42 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/infra_v2_selected.json \
  --noise-types non_gaussian \
  --epochs 50 \
  --seed 42 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

### Phase 3 — cross-evaluate top variants

Create:

```text
train/unet_experiment_configs/top_iteration2_v2.json
```

Include:

```text
M00_B02_repro
best combo candidate
best spectral infrastructure candidate
best complex candidate if it passes smoke tests and beats baseline
best 1D candidate
```

Run both train noise types:

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/top_iteration2_v2.json \
  --noise-types gaussian,non_gaussian \
  --epochs 50 \
  --seed 42 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

### Phase 4 — repeatability

For any candidate beating B02 by at least `0.2 dB` on NG→NG:

```bash
python train/run_unet_experiment_suite.py \
  --dataset data_generation/datasets/deep_space_polygauss_qpsk_bs1024_n400000_c054e749 \
  --config train/unet_experiment_configs/best_iteration2_repeatability.json \
  --noise-types non_gaussian \
  --epochs 50 \
  --seeds 42,43,44 \
  --device cuda \
  --wandb-project signal-denoising-v2
```

Report mean and standard deviation. Do not promote a default based on one seed.

---

## 11. Implementation details for key variants

### 11.1 `mask_scaled_sigmoid` default

For `B02` and most combos:

```python
mask = mask_max * torch.sigmoid(raw)
out_mag = mask * noisy_mag
```

Use:

```text
mask_max = 3.0
```

Log the fraction of mask values above `1`, `2`, and `3`.

### 11.2 Conservative waveform loss

For spectral magnitude models:

```python
phase = noisy_spec / (noisy_spec.abs() + 1e-8)
out_spec = out_mag.squeeze(1) * phase
out_wave = stft_projector.istft(out_spec, length=signal_len)
```

Recommended conservative composite loss:

```python
loss = mag_loss(out_mag, clean_mag)
loss += 0.03 * mse(out_wave, clean_wave)
loss += 0.01 * multi_res_stft_loss(out_wave, clean_wave)
```

Rationale: previous `C01/C02` weights may have been too strong or may have conflicted with the `[0,1]` mask. Use lower weights first with `mask_max=3`.

### 11.3 Anisotropic pooling with mirror upsampling

For `freq_only_first2`:

```python
pool1 = MaxPool2d((2, 1))
pool2 = MaxPool2d((2, 1))
pool3 = MaxPool2d((2, 2))
```

Decoder upsampling must mirror this exactly:

```python
up3 = ConvTranspose2d(..., kernel_size=(2, 2), stride=(2, 2))
up2 = ConvTranspose2d(..., kernel_size=(2, 1), stride=(2, 1))
up1 = ConvTranspose2d(..., kernel_size=(2, 1), stride=(2, 1))
```

Validate final spectrogram shape equals input shape before computing loss.

### 11.4 ResU-Net block

Recommended block:

```python
class ResConvBlock(nn.Module):
    def __init__(self, cin, cout, norm="group", groups=8):
        ...
    def forward(self, x):
        residual = self.proj(x)
        y = self.conv_norm_act_1(x)
        y = self.conv_norm_act_2(y)
        return F.leaky_relu(y + residual, negative_slope=0.1)
```

Do not combine ResU-Net, hop16, aniso, and composite loss all at once in the first infrastructure test. Add combinations only after single infrastructure blocks pass.

### 11.5 Wave-U-Net residual output

Use residual noise prediction:

```python
pred_noise = model(noisy_wave[:, None, :]).squeeze(1)
out_wave = noisy_wave - pred_noise
```

This gives a safe identity path: if the model predicts zero noise, output equals noisy input.

---

## 12. Required tests

Create or update:

```text
tests/test_unet_variants.py
tests/test_stft_projector.py
tests/test_complex_unet_pipeline.py
tests/test_waveunet1d.py
```

Required tests:

```text
spectral variants return out_wave shape [B, signal_len]
complex variants return finite complex spec and waveform
waveunet1d variants return out_wave shape [B, signal_len]
STFT->iSTFT reconstruction length and error check
model_config save/load reconstructs exact architecture
one batch train step produces finite gradients for every variant
loss factory logs all expected components
summarizer reads mixed spectral/complex/waveform result files
```

---

## 13. Reporting requirements

The iteration-2 report must include:

| Rank | Exp ID | Architecture | Train noise | Test G | Test NG | Mean | Worst | Δ vs B02 NG | Δ vs A01 NG | Epoch best | Time | VRAM | Notes |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

Also include:

1. Best combo over B02.
2. Best new infrastructure over B02.
3. Best waveform/1D candidate.
4. Whether complex U-Net passed smoke tests.
5. Mask diagnostics for B02 and best combo.
6. CRM diagnostics for complex candidates.
7. Per-SNR curves for:
   - `A01 fixed_trainer`;
   - `M00_B02_repro`;
   - best combo;
   - best ResU-Net;
   - best complex U-Net if valid;
   - best Wave-U-Net.
8. Repeatability results for any winner.

---

## 14. Acceptance criteria

Iteration 2 is complete when:

1. `combo_v2.json` and `infra_v2.json` are implemented.
2. All selected variants use the new dictionary-return model API or an equivalent explicit wrapper.
3. `M00_B02_repro` reproduces the iteration-1 B02 result within expected seed/run variation.
4. At least one combo suite and one infrastructure suite finish a full Non-Gaussian training run.
5. Complex CRM is either repaired and valid, or clearly reported as failing smoke tests before full training.
6. A top-variant cross-evaluation is run for both Gaussian and Non-Gaussian training.
7. Any winner beating B02 by at least `0.2 dB` is repeated with seeds `42,43,44`.
8. The final report identifies whether the next default should be:
   - `B02 mask3`,
   - a `B02` combination,
   - a spectral ResU-Net,
   - a repaired complex spectral U-Net,
   - or a 1D Wave-U-Net.

---

## 15. Expected outcomes

### Outcome A: `B02 + loss` wins

Interpretation: the mask range was the main blocker, and waveform/MR-STFT loss only helps after the model is allowed to amplify magnitude.

Next action: make `mask_max=3` and conservative `mag_time_mrstft` the new default.

### Outcome B: `B02 + hop/aniso` wins

Interpretation: temporal resolution was a secondary bottleneck hidden by the `[0,1]` mask.

Next action: use `mask_max=3` with the winning temporal-resolution setting.

### Outcome C: ResU-Net wins

Interpretation: the current U-Net capacity/block design is limiting, not just mask range.

Next action: promote `spectral_resunet` with `mask_max=3` as the spectral default.

### Outcome D: repaired complex U-Net wins

Interpretation: noisy phase was a major ceiling, and the previous `F01` failure was implementation/loss-related.

Next action: prioritize phase-aware spectral U-Net.

### Outcome E: Wave-U-Net wins

Interpretation: time-domain phase/timing information matters more than spectral magnitude tuning.

Next action: keep U-Net family, but move main U-Net experiments to 1D waveform residual models.

### Outcome F: nothing beats B02

Interpretation: the main available gain was mask range. Further progress probably requires either a better complex pipeline, stronger time-domain U-Net, or architectural ideas beyond simple U-Net parameter combinations.

Next action: keep `B02 mask3` as the default U-Net and focus on repaired complex/time-domain infrastructure.
