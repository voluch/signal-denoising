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
from metrics import MeanSquaredError, MeanAbsoluteError, RootMeanSquaredError, SignalToNoiseRatio
from train.losses import select_loss
from train.snr_curve import evaluate_per_snr, print_snr_table, plot_snr_curve, log_snr_curve_wandb, save_training_curves

MODEL_NAME = 'UnetAutoencoder'

def negative_snr_loss(pred, target, eps=1e-8):
    """Time-domain SNR loss (negative to minimize)."""
    noise = pred - target
    signal_power = torch.mean(target ** 2, dim=1) + eps
    noise_power = torch.mean(noise ** 2, dim=1) + eps
    return -10.0 * torch.log10(signal_power / noise_power).mean()

def _stft_mag_torch(x: torch.Tensor, nperseg: int, noverlap: int) -> torch.Tensor:
    hop = nperseg - noverlap
    win = torch.hann_window(nperseg, periodic=True, device=x.device, dtype=x.dtype)
    return torch.abs(torch.stft(x, n_fft=nperseg, hop_length=hop, win_length=nperseg,
                                window=win, center=True, return_complex=True))

def multi_res_stft_loss(x_hat: torch.Tensor, x: torch.Tensor,
                        configs=((32, 16), (64, 32), (16, 8))) -> torch.Tensor:
    total = 0.0
    for n, ov in configs:
        S_hat = _stft_mag_torch(x_hat, n, ov)
        S     = _stft_mag_torch(x,     n, ov)
        l1 = torch.mean(torch.abs(torch.log1p(S_hat) - torch.log1p(S)))
        sc = (torch.linalg.norm(S_hat - S, ord='fro', dim=(1, 2)) /
              (torch.linalg.norm(S, ord='fro', dim=(1, 2)) + 1e-12)).mean()
        total = total + l1 + 0.5 * sc
    return total / len(configs)

class UnetAutoencoderTrainer:
    def __init__(self, dataset_path: Path, noise_type="non_gaussian",
                 batch_size=512, epochs=50, learning_rate=1e-3,
                 signal_len=1024, fs=8192, nperseg=128, noverlap=None, hop_length=32,
                 random_state=42, wandb_project="", device=None, data_fraction=1.0, 
                 output_dir=None, run_id: str | None = None,
                 # New experiment parameters
                 input_domain="mag", output_mode="mask_sigmoid", mask_max=1.0, softplus_max=3.0,
                 pooling_mode="isotropic", loss_profile="mag", loss_name="mse",
                 time_loss_weight=1.0, mrstft_loss_weight=1.0, snr_loss_weight=1.0,
                 checkpoint_metric="val_snr", scheduler_metric="val_snr",
                 min_epochs=25, early_stop_patience=15, weight_decay=1e-4, grad_clip_norm=1.0,
                 save_every_epoch=False):
        
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
        
        self.random_state = random_state
        from train.repro_utils import set_global_seed
        set_global_seed(random_state)
        self.data_fraction = data_fraction
        
        self.output_dir = Path(output_dir) if output_dir is not None else None
        from train.device_utils import get_device
        self.device = get_device(device)

        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.run_date = datetime.now().strftime("%Y%m%d")
        self.dataset_uid = self.dataset_path.name.split('_')[-1]
        
        # Experiment params
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

        if WANDB_OK and wandb_project:
            if not wandb.api.api_key:
                api_key = os.getenv("WANDB_API_KEY")
                if api_key:
                    wandb.login(key=api_key)

            run_name = f"{MODEL_NAME}_{noise_type}_{self.dataset_uid}_{self.run_id}"
            wandb.init(project=wandb_project, name=run_name, reinit=True, config={
                "model": MODEL_NAME, "noise_type": noise_type,
                "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                "random_state": random_state, "fs": fs, "nperseg": nperseg, "hop_length": hop_length,
                "dataset": self.dataset_path.name, "run_id": self.run_id,
                "input_domain": input_domain, "output_mode": output_mode,
                "pooling_mode": pooling_mode, "loss_profile": loss_profile,
                "checkpoint_metric": checkpoint_metric, "scheduler_metric": scheduler_metric,
            })
            print(f"[W&B] Logging enabled → project='{wandb_project}', run='{run_name}'")
        else:
            print(f"[W&B] Logging disabled")

        self.train_loader, self.val_loader, self.test_loader, self.input_shape = self._load_data()
        
        # Determine in_channels based on input_domain
        in_channels = 1
        if input_domain in ["real_imag", "real_imag_mag"]:
            in_channels = 2 if input_domain == "real_imag" else 3
            
        self.model = UnetAutoencoder(
            input_shape=self.input_shape,
            in_channels=in_channels,
            out_channels=1, # mag output usually
            pooling_mode=pooling_mode,
            output_mode=output_mode,
            mask_max=mask_max,
            softplus_max=softplus_max,
        ).to(self.device)

    def _stft_batch(self, x: torch.Tensor) -> torch.Tensor:
        win = torch.hann_window(self.nperseg, device=x.device)
        return torch.stft(x, n_fft=self.nperseg,
                          hop_length=self.nperseg - self.noverlap,
                          win_length=self.nperseg, window=win,
                          center=True, pad_mode='reflect',
                          onesided=True, return_complex=True)

    def _istft_batch(self, spec: torch.Tensor) -> torch.Tensor:
        win = torch.hann_window(self.nperseg, device=spec.device)
        return torch.istft(spec, n_fft=self.nperseg,
                           hop_length=self.nperseg - self.noverlap,
                           win_length=self.nperseg, window=win,
                           center=True, onesided=True, length=self.signal_len)

    def _preprocess_input(self, noisy_raw: torch.Tensor) -> torch.Tensor:
        spec = self._stft_batch(noisy_raw)
        mag = spec.abs().unsqueeze(1)
        
        if self.input_domain == "mag":
            return mag
        elif self.input_domain == "log1p_mag":
            return torch.log1p(mag)
        elif self.input_domain == "real_imag":
            return torch.stack([spec.real, spec.imag], dim=1)
        elif self.input_domain == "real_imag_mag":
            return torch.stack([spec.real, spec.imag, mag.squeeze(1)], dim=1)
        else:
            return mag

    def _load_data(self):
        noisy = np.load(self.dataset_path / "train" / f"{self.noise_type}_signals.npy")
        clean = np.load(self.dataset_path / "train" / "clean_signals.npy")
        if self.data_fraction < 1.0:
            n = max(1, int(len(noisy) * self.data_fraction))
            noisy, clean = noisy[:n], clean[:n]
        
        noisy_raw = torch.tensor(noisy, dtype=torch.float32)
        clean_raw = torch.tensor(clean, dtype=torch.float32)

        # Get input shape from a dummy STFT
        dummy_spec = self._stft_batch(noisy_raw[:1])
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
            DataLoader(val_set,   batch_size=self.batch_size, **dl_kw),
            DataLoader(test_set,  batch_size=self.batch_size, **dl_kw),
            input_shape,
        )

    def denoise_batch(self, noisy_raw: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x = self._preprocess_input(noisy_raw)
            spec = self._stft_batch(noisy_raw)
            mag = spec.abs().unsqueeze(1)
            
            out = self.model(x, noisy_mag=mag, noisy_spec=spec)
            
            if "out_spec" in out:
                return self._istft_batch(out["out_spec"].squeeze(1))
            elif "out_mag" in out:
                phase = spec / (spec.abs() + 1e-8)
                out_spec = out["out_mag"].squeeze(1) * phase
                return self._istft_batch(out_spec)
            else:
                return noisy_raw

    def _get_loss(self, noisy_raw, clean_raw):
        x_in = self._preprocess_input(noisy_raw)
        spec_noisy = self._stft_batch(noisy_raw)
        mag_noisy = spec_noisy.abs().unsqueeze(1)
        
        spec_clean = self._stft_batch(clean_raw)
        mag_clean = spec_clean.abs().unsqueeze(1)
        
        out = self.model(x_in, noisy_mag=mag_noisy, noisy_spec=spec_noisy)
        
        # Base spectral loss
        if self.loss_profile in ["mag", "mag_time", "mag_time_mrstft", "snr_aux"]:
            out_mag = out.get("out_mag")
            if out_mag is None and "mask" in out:
                out_mag = out["mask"] * mag_noisy
            
            if self.loss_name == "mse":
                loss = F.mse_loss(out_mag, mag_clean)
            elif self.loss_name == "l1":
                loss = F.l1_loss(out_mag, mag_clean)
            elif self.loss_name == "huber":
                loss = F.huber_loss(out_mag, mag_clean)
            else:
                loss = F.mse_loss(out_mag, mag_clean)
        elif self.loss_profile == "complex_time":
            out_spec = out["out_spec"]
            # Complex STFT loss
            loss = F.mse_loss(torch.view_as_real(out_spec), torch.view_as_real(spec_clean.unsqueeze(1)))
        elif self.loss_profile == "time_mrstft":
            loss = 0.0 # Will be handled by time domain losses
        else:
            loss = 0.0

        # Time domain losses
        if self.loss_profile in ["mag_time", "mag_time_mrstft", "time_mrstft", "complex_time", "snr_aux"]:
            if "out_spec" in out:
                out_wave = self._istft_batch(out["out_spec"].squeeze(1))
            else:
                out_mag = out.get("out_mag")
                if out_mag is None and "mask" in out:
                    out_mag = out["mask"] * mag_noisy
                phase_noisy = spec_noisy / (mag_noisy.squeeze(1) + 1e-8)
                out_wave = self._istft_batch(out_mag.squeeze(1) * phase_noisy)
            
            if self.loss_profile != "snr_aux" or self.time_loss_weight > 0:
                loss += self.time_loss_weight * F.mse_loss(out_wave, clean_raw)
            
            if "mrstft" in self.loss_profile:
                loss += self.mrstft_loss_weight * multi_res_stft_loss(out_wave, clean_raw)
            
            if self.loss_profile == "snr_aux":
                loss += self.snr_loss_weight * negative_snr_loss(out_wave, clean_raw)

        return loss

    def train(self) -> dict:
        start_time = time.time()
        print(f"\n🚀 Training UNet Experiment: {self.run_id}")
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if self.scheduler_metric == "val_snr":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=5, factor=0.5, threshold=0.02, threshold_mode="abs", cooldown=2, min_lr=1e-5
            )
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5, factor=0.5, cooldown=2, min_lr=1e-5
            )

        best_val_loss = float("inf")
        best_val_snr  = float("-inf")
        train_history, val_snr_history = [], []
        no_improve = 0
        
        # Checkpoint directory
        if self.output_dir:
            run_dir = self.output_dir
        else:
            run_dir = self.dataset_path / "runs" / f"unet_exp_{self.run_date}_{self.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d}", leave=False)
            for noisy_raw, clean_raw in pbar:
                noisy_raw, clean_raw = noisy_raw.to(self.device), clean_raw.to(self.device)
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
            with torch.no_grad():
                for noisy_raw, clean_raw in self.val_loader:
                    noisy_raw_dev, clean_raw_dev = noisy_raw.to(self.device), clean_raw.to(self.device)
                    val_loss += self._get_loss(noisy_raw_dev, clean_raw_dev).item()
                    
                    pred = self.denoise_batch(noisy_raw_dev).cpu().numpy()
                    all_pred.append(pred)
                    all_true.append(clean_raw.numpy())
            
            val_loss /= len(self.val_loader)
            val_snr = float(SignalToNoiseRatio.calculate(np.concatenate(all_true), np.concatenate(all_pred)))
            
            if self.scheduler_metric == "val_snr":
                scheduler.step(val_snr)
            else:
                scheduler.step(val_loss)
            
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:02d} | loss={epoch_loss/len(self.train_loader):.5f} | val_loss={val_loss:.5f} | val_SNR={val_snr:.2f} dB | lr={lr_now:.2e}")
            
            if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
                wandb.log({"train/loss": epoch_loss/len(self.train_loader), "val/loss": val_loss, "val/snr": val_snr, "lr": lr_now}, step=epoch)

            # Checkpointing
            if val_snr > best_val_snr:
                best_val_snr = val_snr
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

            if epoch >= self.min_epochs and no_improve >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Final evaluation
        self.model.load_state_dict(torch.load(run_dir / "model_best_snr.pth"))
        test_metrics = self._evaluate_test()
        
        # Per-SNR curves
        per_snr = {}
        test_dir = self.dataset_path / "test"
        if test_dir.exists():
            per_snr = evaluate_per_snr(lambda x: self.denoise_batch(torch.tensor(x, device=self.device)).cpu().numpy(), test_dir, self.noise_type, batch_size=self.batch_size)
            plot_snr_curve(per_snr, MODEL_NAME, save_path=run_dir / "figures" / "snr_curve.png")
            save_training_curves(train_history, val_snr_history, run_dir / "figures" / "training_curves.png", MODEL_NAME, self.noise_type)

        # Save experiment config
        exp_config = {
            "run_id": self.run_id,
            "dataset": self.dataset_path.name,
            "noise_type": self.noise_type,
            "input_domain": self.input_domain,
            "output_mode": self.output_mode,
            "pooling_mode": self.pooling_mode,
            "loss_profile": self.loss_profile,
            "val_snr": best_val_snr,
            "test_snr": test_metrics.get("SNR"),
        }
        with open(run_dir / "experiment_config.json", "w") as f:
            json.dump(exp_config, f, indent=2)

        if WANDB_OK and hasattr(wandb, 'run') and wandb.run:
            wandb.finish()

        return {
            'model': MODEL_NAME, 'noise_type': self.noise_type,
            'val_snr': best_val_snr, 'test_metrics': test_metrics,
            'per_snr_results': per_snr, 'weights_path': str(run_dir / "model_best_snr.pth"),
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
            "MSE":  MeanSquaredError.calculate(y_true, y_pred),
            "MAE":  MeanAbsoluteError.calculate(y_true, y_pred),
            "RMSE": RootMeanSquaredError.calculate(y_true, y_pred),
            "SNR":  SignalToNoiseRatio.calculate(y_true, y_pred),
        }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--noise-type", default="non_gaussian")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--nperseg", type=int, default=128)
    p.add_argument("--hop-length", type=int, default=32)
    p.add_argument("--signal-len", type=int, default=1024)
    p.add_argument("--fs", type=int, default=8192)
    p.add_argument("--input-domain", default="mag")
    p.add_argument("--output-mode", default="mask_sigmoid")
    p.add_argument("--mask-max", type=float, default=1.0)
    p.add_argument("--softplus-max", type=float, default=3.0)
    p.add_argument("--pooling-mode", default="isotropic")
    p.add_argument("--loss-profile", default="mag")
    p.add_argument("--loss-name", default="mse")
    p.add_argument("--time-loss-weight", type=float, default=1.0)
    p.add_argument("--mrstft-loss-weight", type=float, default=1.0)
    p.add_argument("--snr-loss-weight", type=float, default=1.0)
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
        run_id=args.run_id
    )
    trainer.train()
