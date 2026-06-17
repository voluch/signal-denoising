import argparse
import json
import sys
import uuid
import time
import os
import gc
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False

from tqdm import tqdm

from models.autoencoder_unet import UnetAutoencoder
from models.unet_registry import build_unet_variant, param_count
from models.stft_projector import STFTProjector
from train.unet_losses import build_loss_fn, multi_res_stft_loss, negative_snr_loss
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from train.losses import select_loss
from train.snr_curve import evaluate_per_snr, print_snr_table, plot_snr_curve, log_snr_curve_wandb, save_training_curves

MODEL_NAME = "UnetAutoencoder"


class UnetAutoencoderTrainer:
    def __init__(
        self,
        dataset_path: Path,
        noise_type: str = "non_gaussian",
        batch_size: int = 512,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        signal_len: int = 1024,
        fs: int = 8192,
        nperseg: int = 128,
        noverlap=None,
        hop_length: int = 32,
        random_state: int = 42,
        wandb_project: str = "",
        device=None,
        data_fraction: float = 1.0,
        output_dir=None,
        run_id: str | None = None,
        exp_id: str | None = None,
        description: str | None = None,
        # Spectral / mask params
        input_domain: str = "mag",
        output_mode: str = "mask_scaled_sigmoid",
        mask_max: float = 3.0,
        softplus_max: float = 3.0,
        pooling_mode: str = "isotropic",
        # Loss params
        loss_profile: str = "mag",
        loss_name=None,
        time_loss_weight: float = 0.03,
        mrstft_loss_weight: float = 0.01,
        snr_loss_weight: float = 0.01,
        # Training control
        checkpoint_metric: str = "val_snr",
        scheduler_metric: str = "val_snr",
        min_epochs: int = 25,
        early_stop_patience: int = 15,
        weight_decay: float = 1e-4,
        grad_clip_norm: float = 1.0,
        save_every_epoch: bool = False,
        optimizer_name: str = "adamw",
        disable_early_stop: bool = False,
        # Architecture selection (new in iteration 2)
        architecture: str = "spectral_unet",
        base_channels: int = 32,
        depth_layers: int = 3,
        bottleneck: str = "standard",
        dilation_rates=(1, 2, 4),
        skip_attention: bool = False,
        crm_scale: float = 0.5,
    ):
        self.dataset_path = Path(dataset_path)
        self.noise_type = noise_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.signal_len = signal_len
        self.fs = fs
        self.nperseg = nperseg

        if hop_length is not None:
            self.noverlap = nperseg - hop_length
        else:
            self.noverlap = noverlap if noverlap is not None else (nperseg * 3 // 4)

        self.hop_length = self.nperseg - self.noverlap
        self.random_state = random_state
        from train.repro_utils import set_global_seed
        set_global_seed(random_state)
        self.data_fraction = data_fraction

        self.output_dir = Path(output_dir) if output_dir is not None else None
        from train.device_utils import get_device
        self.device = get_device(device)

        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.exp_id = exp_id
        self.description = description
        self.run_date = datetime.now().strftime("%Y%m%d")
        self.dataset_uid = self.dataset_path.name.split("_")[-1]

        # Experiment params
        self.architecture = architecture
        self.input_domain = input_domain
        self.output_mode = output_mode
        self.mask_max = mask_max
        self.softplus_max = softplus_max
        self.pooling_mode = pooling_mode
        self.loss_profile = loss_profile
        self.loss_name = loss_name
        self.time_loss_weight = time_loss_weight
        self.mrstft_loss_weight = mrstft_loss_weight
        self.snr_loss_weight = snr_loss_weight
        self.checkpoint_metric = checkpoint_metric
        self.scheduler_metric = scheduler_metric
        self.min_epochs = min_epochs
        self.early_stop_patience = early_stop_patience
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.save_every_epoch = save_every_epoch
        self.optimizer_name = optimizer_name.lower()
        self.disable_early_stop = disable_early_stop
        # Architecture-specific
        self.base_channels = base_channels
        self.depth_layers = depth_layers
        self.bottleneck = bottleneck
        self.dilation_rates = tuple(dilation_rates)
        self.skip_attention = skip_attention
        self.crm_scale = crm_scale

        # STFT projector (shared by all spectral models)
        self.stft_proj = STFTProjector(nperseg, self.hop_length, signal_len)

        # Loss factory (covers all profiles including legacy ones)
        self.loss_fn = build_loss_fn(loss_profile, time_loss_weight, mrstft_loss_weight, snr_loss_weight)

        if WANDB_OK and wandb_project:
            if not wandb.api.api_key:
                api_key = os.getenv("WANDB_API_KEY")
                if api_key:
                    wandb.login(key=api_key)
            run_name = f"{MODEL_NAME}_{noise_type}_{self.dataset_uid}_{self.run_id}"
            wandb.init(
                project=wandb_project, name=run_name, reinit=True,
                config={
                    "model": MODEL_NAME, "noise_type": noise_type,
                    "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                    "random_state": random_state, "nperseg": nperseg, "hop_length": self.hop_length,
                    "dataset": self.dataset_path.name, "run_id": self.run_id,
                    "architecture": architecture, "input_domain": input_domain,
                    "output_mode": output_mode, "pooling_mode": pooling_mode,
                    "loss_profile": loss_profile,
                },
            )
            print(f"[W&B] Logging enabled → project='{wandb_project}', run='{run_name}'")
        else:
            print("[W&B] Logging disabled")

        self.train_loader, self.val_loader, self.test_loader, self.input_shape = self._load_data()

        # Build model via registry
        model_config = self._build_model_config()
        self.model_config = model_config
        self.model = build_unet_variant(
            model_config,
            input_shape=None if architecture == "waveunet1d" else self.input_shape,
        ).to(self.device)
        print(f"[Model] {architecture} | params={param_count(self.model):,}")

    # ── model config dict ─────────────────────────────────────────────────────

    def _build_model_config(self) -> dict:
        cfg = {
            "architecture": self.architecture,
            "input_domain": self.input_domain,
            "output_mode": self.output_mode,
            "mask_max": self.mask_max,
            "softplus_max": self.softplus_max,
            "pooling_mode": self.pooling_mode,
            "base_channels": self.base_channels,
            "depth": self.depth_layers,
            "bottleneck": self.bottleneck,
            "dilation_rates": list(self.dilation_rates),
            "skip_attention": self.skip_attention,
            "crm_scale": self.crm_scale,
            "nperseg": self.nperseg,
            "hop_length": self.hop_length,
            "signal_len": self.signal_len,
        }
        if hasattr(self, "input_shape"):
            cfg["input_shape"] = list(self.input_shape)
        return cfg

    # ── data ──────────────────────────────────────────────────────────────────

    def _load_data(self):
        noisy = np.load(self.dataset_path / "train" / f"{self.noise_type}_signals.npy")
        clean = np.load(self.dataset_path / "train" / "clean_signals.npy")
        if self.data_fraction < 1.0:
            n = max(1, int(len(noisy) * self.data_fraction))
            noisy, clean = noisy[:n], clean[:n]

        noisy_raw = torch.tensor(noisy, dtype=torch.float32)
        clean_raw = torch.tensor(clean, dtype=torch.float32)

        # Compute STFT shape on CPU
        dummy_spec = self.stft_proj.stft(noisy_raw[:1])
        input_shape = (dummy_spec.shape[1], dummy_spec.shape[2])

        dataset = TensorDataset(noisy_raw, clean_raw)
        total = len(dataset)
        val_len = int(0.25 * total)
        test_len = int(0.25 * total)
        train_len = total - val_len - test_len
        train_set, val_set, test_set = random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.random_state),
        )
        from train.device_utils import get_dataloader_kwargs
        dl_kw = get_dataloader_kwargs(self.device)
        return (
            DataLoader(train_set, batch_size=self.batch_size, shuffle=True, **dl_kw),
            DataLoader(val_set, batch_size=self.batch_size, **dl_kw),
            DataLoader(test_set, batch_size=self.batch_size, **dl_kw),
            input_shape,
        )

    # ── forward helpers ───────────────────────────────────────────────────────

    def _preprocess_input_spectral(
        self, spec_noisy: torch.Tensor, mag_noisy: torch.Tensor
    ) -> torch.Tensor:
        if self.input_domain == "mag":
            return mag_noisy
        elif self.input_domain == "log1p_mag":
            return torch.log1p(mag_noisy)
        elif self.input_domain == "real_imag":
            return torch.stack([spec_noisy.real, spec_noisy.imag], dim=1)
        elif self.input_domain == "real_imag_mag":
            return torch.stack([spec_noisy.real, spec_noisy.imag, mag_noisy.squeeze(1)], dim=1)
        return mag_noisy

    def _forward_model(
        self, noisy_raw: torch.Tensor
    ) -> tuple[dict, torch.Tensor | None, torch.Tensor | None]:
        """
        Run model forward pass and return:
            (out_dict_with_out_wave, spec_noisy_or_None, mag_noisy_or_None)
        out_dict always contains 'out_wave'.
        """
        if self.architecture == "waveunet1d":
            out = self.model(noisy_raw.unsqueeze(1))
            out["out_wave"] = noisy_raw - out["pred_noise"].squeeze(1)
            return out, None, None

        spec_noisy = self.stft_proj.stft(noisy_raw)
        mag_noisy = spec_noisy.abs().unsqueeze(1)
        x_in = self._preprocess_input_spectral(spec_noisy, mag_noisy)
        out = self.model(x_in, noisy_mag=mag_noisy, noisy_spec=spec_noisy)

        # Ensure out_wave is always present
        if "out_wave" not in out:
            if "out_spec" in out:
                os = out["out_spec"]
                if os.dim() == 4:
                    os = os.squeeze(1)
                out["out_wave"] = self.stft_proj.istft(os, self.signal_len)
            elif "out_mag" in out:
                phase = spec_noisy / (spec_noisy.abs() + 1e-8)
                out["out_wave"] = self.stft_proj.istft(
                    out["out_mag"].squeeze(1) * phase, self.signal_len
                )

        return out, spec_noisy, mag_noisy

    # ── loss ─────────────────────────────────────────────────────────────────

    def _get_loss(self, noisy_raw: torch.Tensor, clean_raw: torch.Tensor) -> torch.Tensor:
        out, spec_noisy, _ = self._forward_model(noisy_raw)
        clean_spec = self.stft_proj.stft(clean_raw) if spec_noisy is not None else None
        total, _ = self.loss_fn(out, clean_raw, clean_spec)
        return total

    # ── inference ─────────────────────────────────────────────────────────────

    def denoise_batch(self, noisy_raw: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            out, _, _ = self._forward_model(noisy_raw)
            return out["out_wave"]

    def denoise_numpy(self, noisy: np.ndarray) -> np.ndarray:
        results = []
        for i in range(0, len(noisy), self.batch_size):
            batch = torch.tensor(noisy[i:i + self.batch_size], dtype=torch.float32, device=self.device)
            results.append(self.denoise_batch(batch).cpu().numpy())
        return np.concatenate(results)

    # ── mask diagnostics ──────────────────────────────────────────────────────

    def _collect_diagnostics(self, out: dict) -> dict:
        diag: dict = {}
        mask = out.get("mask")
        if mask is not None:
            with torch.no_grad():
                m = mask.detach().float().flatten()
                diag["mask_mean"] = round(m.mean().item(), 5)
                diag["mask_std"] = round(m.std().item(), 5)
                diag["mask_p50"] = round(m.quantile(0.50).item(), 5)
                diag["mask_p90"] = round(m.quantile(0.90).item(), 5)
                diag["mask_p99"] = round(m.quantile(0.99).item(), 5)
                diag["frac_mask_gt1"] = round((m > 1).float().mean().item(), 5)
                diag["frac_mask_gt2"] = round((m > 2).float().mean().item(), 5)
                diag["frac_mask_gt3"] = round((m > 3).float().mean().item(), 5)

        out_mag = out.get("out_mag")
        if out_mag is not None:
            with torch.no_grad():
                om = out_mag.detach().float().flatten()
                diag["out_mag_max"] = round(om.max().item(), 5)
                diag["out_mag_p99"] = round(om.quantile(0.99).item(), 5)

        out_wave = out.get("out_wave")
        if out_wave is not None:
            with torch.no_grad():
                ow = out_wave.detach().float()
                diag["waveform_peak_abs"] = round(ow.abs().max().item(), 5)
                diag["nan_count"] = int(ow.isnan().sum().item())
                diag["inf_count"] = int(ow.isinf().sum().item())

        crm = out.get("crm")
        if crm is not None:
            with torch.no_grad():
                ca = crm.detach().abs().flatten()
                diag["crm_abs_p50"] = round(ca.quantile(0.50).item(), 5)
                diag["crm_abs_p90"] = round(ca.quantile(0.90).item(), 5)
                diag["crm_abs_p99"] = round(ca.quantile(0.99).item(), 5)
                diag["frac_crm_abs_gt2"] = round((ca > 2).float().mean().item(), 5)
                diag["frac_crm_abs_gt5"] = round((ca > 5).float().mean().item(), 5)

        return diag

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self) -> dict:
        start_time = time.time()
        print(f"\n[Train] {self.exp_id or self.run_id} | arch={self.architecture} | noise={self.noise_type}")

        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        sched_mode = "max" if self.scheduler_metric == "val_snr" else "min"
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sched_mode, patience=5, factor=0.5,
            threshold=0.02, threshold_mode="abs", cooldown=2, min_lr=1e-5,
        )

        best_val_loss = float("inf")
        best_val_snr = float("-inf")
        best_epoch_snr = 0
        train_history, val_snr_history = [], []
        no_improve = 0

        if self.output_dir:
            run_dir = self.output_dir
        else:
            run_dir = self.dataset_path / "runs" / f"unet_exp_{self.run_date}_{self.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        last_diag: dict = {}

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d}", leave=False)
            for noisy_raw, clean_raw in pbar:
                noisy_raw = noisy_raw.to(self.device)
                clean_raw = clean_raw.to(self.device)
                loss = self._get_loss(noisy_raw, clean_raw)
                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.5f}")

            # Validation
            self.model.eval()
            val_loss = 0.0
            all_true, all_pred = [], []
            diag_collected = False
            with torch.no_grad():
                for noisy_raw, clean_raw in self.val_loader:
                    noisy_raw_d = noisy_raw.to(self.device)
                    clean_raw_d = clean_raw.to(self.device)
                    val_loss += self._get_loss(noisy_raw_d, clean_raw_d).item()

                    pred = self.denoise_batch(noisy_raw_d).cpu().numpy()
                    all_pred.append(pred)
                    all_true.append(clean_raw.numpy())

                    # Collect diagnostics once per epoch from first batch
                    if not diag_collected:
                        out_tmp, _, _ = self._forward_model(noisy_raw_d)
                        last_diag = self._collect_diagnostics(out_tmp)
                        diag_collected = True

            val_loss /= len(self.val_loader)
            val_snr = float(SignalToNoiseRatio.calculate(
                np.concatenate(all_true), np.concatenate(all_pred)
            ))

            if sched_mode == "max":
                scheduler.step(val_snr)
            else:
                scheduler.step(val_loss)

            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d} | loss={epoch_loss/len(self.train_loader):.5f} "
                f"| val_loss={val_loss:.5f} | val_SNR={val_snr:.2f} dB | lr={lr_now:.2e}"
            )

            if WANDB_OK and hasattr(wandb, "run") and wandb.run:
                log_dict = {
                    "train/loss": epoch_loss / len(self.train_loader),
                    "val/loss": val_loss, "val/snr": val_snr, "lr": lr_now,
                }
                log_dict.update({f"diag/{k}": v for k, v in last_diag.items()})
                wandb.log(log_dict, step=epoch)

            if val_snr > best_val_snr:
                best_val_snr = val_snr
                best_epoch_snr = epoch
                torch.save(self.model.state_dict(), run_dir / "model_best_snr.pth")
                no_improve = 0
            else:
                no_improve += 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), run_dir / "model_best_loss.pth")

            if self.save_every_epoch:
                torch.save(self.model.state_dict(), run_dir / f"model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), run_dir / "model_last.pth")

            train_history.append(epoch_loss / len(self.train_loader))
            val_snr_history.append(val_snr)

            if not self.disable_early_stop:
                if epoch >= self.min_epochs and no_improve >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Final evaluation
        self.model.load_state_dict(
            torch.load(run_dir / "model_best_snr.pth", map_location=self.device, weights_only=True)
        )
        test_metrics = self._evaluate_test()
        elapsed = time.time() - start_time

        per_snr = {}
        test_dir = self.dataset_path / "test"
        if test_dir.exists():
            per_snr = evaluate_per_snr(
                lambda x: self.denoise_numpy(x), test_dir, self.noise_type, batch_size=self.batch_size
            )
            (run_dir / "figures").mkdir(parents=True, exist_ok=True)
            plot_snr_curve(per_snr, MODEL_NAME, save_path=run_dir / "figures" / "snr_curve.png")
            save_training_curves(
                train_history, val_snr_history,
                run_dir / "figures" / "training_curves.png", MODEL_NAME, self.noise_type,
            )

        # Save model_config.json (allows model reconstruction)
        self.model_config["input_shape"] = list(self.input_shape)
        with open(run_dir / "model_config.json", "w") as f:
            json.dump(self.model_config, f, indent=2)

        # Save experiment_config.json
        exp_config = {
            "exp_id": self.exp_id,
            "description": self.description,
            "run_id": self.run_id,
            "dataset": self.dataset_path.name,
            "dataset_uid": self.dataset_uid,
            "noise_type": self.noise_type,
            "architecture": self.architecture,
            "input_domain": self.input_domain,
            "output_mode": self.output_mode,
            "mask_max": self.mask_max,
            "softplus_max": self.softplus_max,
            "pooling_mode": self.pooling_mode,
            "loss_profile": self.loss_profile,
            "loss_name": self.loss_name,
            "optimizer": self.optimizer_name,
            "learning_rate": self.lr,
            "epochs_run": best_epoch_snr,
            "epochs_max": self.epochs,
            "batch_size": self.batch_size,
            "nperseg": self.nperseg,
            "hop_length": self.hop_length,
            "seed": self.random_state,
            "checkpoint_metric": self.checkpoint_metric,
            "val_snr": float(best_val_snr),
            "test_snr": float(test_metrics.get("SNR", 0)),
            "test_metrics": test_metrics,
            "training_time_s": round(elapsed, 1),
            "param_count": param_count(self.model),
            "mask_diagnostics": last_diag,
            "timestamp": datetime.now().isoformat(),
        }
        with open(run_dir / "experiment_config.json", "w") as f:
            json.dump(exp_config, f, indent=2)

        # Save training_args.json
        training_args = {
            "lr": self.lr, "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "min_epochs": self.min_epochs, "early_stop_patience": self.early_stop_patience,
            "scheduler_metric": self.scheduler_metric,
            "time_loss_weight": self.time_loss_weight,
            "mrstft_loss_weight": self.mrstft_loss_weight,
            "snr_loss_weight": self.snr_loss_weight,
        }
        with open(run_dir / "training_args.json", "w") as f:
            json.dump(training_args, f, indent=2)

        if WANDB_OK and hasattr(wandb, "run") and wandb.run:
            wandb.finish()

        return {
            "model": MODEL_NAME, "noise_type": self.noise_type,
            "val_snr": best_val_snr, "test_metrics": test_metrics,
            "per_snr_results": per_snr,
            "weights_path": str(run_dir / "model_best_snr.pth"),
        }

    def _evaluate_test(self) -> dict:
        all_true, all_pred = [], []
        for noisy_raw, clean_raw in self.test_loader:
            pred = self.denoise_batch(noisy_raw.to(self.device)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(clean_raw.numpy())
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        return {
            "MSE":  float(MeanSquaredError.calculate(y_true, y_pred)),
            "MAE":  float(MeanAbsoluteError.calculate(y_true, y_pred)),
            "RMSE": float(RootMeanSquaredError.calculate(y_true, y_pred)),
            "SNR":  float(SignalToNoiseRatio.calculate(y_true, y_pred)),
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--noise-type", default="non_gaussian")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--nperseg", type=int, default=128)
    p.add_argument("--hop-length", type=int, default=32)
    p.add_argument("--signal-len", type=int, default=1024)
    p.add_argument("--fs", type=int, default=8192)
    # Spectral params
    p.add_argument("--input-domain", default="mag")
    p.add_argument("--output-mode", default="mask_scaled_sigmoid")
    p.add_argument("--mask-max", type=float, default=3.0)
    p.add_argument("--softplus-max", type=float, default=3.0)
    p.add_argument("--pooling-mode", default="isotropic")
    # Loss params
    p.add_argument("--loss-profile", default="mag")
    p.add_argument("--loss-name", default=None)
    p.add_argument("--time-loss-weight", type=float, default=0.03)
    p.add_argument("--mrstft-loss-weight", type=float, default=0.01)
    p.add_argument("--snr-loss-weight", type=float, default=0.01)
    # Training control
    p.add_argument("--checkpoint-metric", default="val_snr")
    p.add_argument("--scheduler-metric", default="val_snr")
    p.add_argument("--min-epochs", type=int, default=25)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--partial-train", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--wandb-project", default="")
    p.add_argument("--device", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--exp-id", default=None)
    p.add_argument("--description", default=None)
    p.add_argument("--optimizer", default="adamw")
    p.add_argument("--disable-early-stop", action="store_true")
    # Architecture params (new in iteration 2)
    p.add_argument("--architecture", default="spectral_unet")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--bottleneck", default="standard")
    p.add_argument("--dilation-rates", type=json.loads, default=[1, 2, 4])
    p.add_argument("--skip-attention", action="store_true")
    p.add_argument("--crm-scale", type=float, default=0.5)
    # Ignored flags from infra_v2 configs (kept for forward compat)
    p.add_argument("--residual-blocks", action="store_true")
    args = p.parse_args()

    trainer = UnetAutoencoderTrainer(
        dataset_path=Path(args.dataset),
        noise_type=args.noise_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        signal_len=args.signal_len,
        fs=args.fs,
        nperseg=args.nperseg,
        hop_length=args.hop_length,
        input_domain=args.input_domain,
        output_mode=args.output_mode,
        mask_max=args.mask_max,
        softplus_max=args.softplus_max,
        pooling_mode=args.pooling_mode,
        loss_profile=args.loss_profile,
        loss_name=args.loss_name,
        time_loss_weight=args.time_loss_weight,
        mrstft_loss_weight=args.mrstft_loss_weight,
        snr_loss_weight=args.snr_loss_weight,
        checkpoint_metric=args.checkpoint_metric,
        scheduler_metric=args.scheduler_metric,
        min_epochs=args.min_epochs,
        early_stop_patience=args.early_stop_patience,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        data_fraction=args.partial_train,
        random_state=args.seed,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        device=args.device,
        run_id=args.run_id,
        exp_id=args.exp_id,
        description=args.description,
        optimizer_name=args.optimizer,
        disable_early_stop=args.disable_early_stop,
        architecture=args.architecture,
        base_channels=args.base_channels,
        depth_layers=args.depth,
        bottleneck=args.bottleneck,
        dilation_rates=args.dilation_rates,
        skip_attention=args.skip_attention,
        crm_scale=args.crm_scale,
    )
    trainer.train()