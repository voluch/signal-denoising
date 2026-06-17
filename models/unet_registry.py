"""Registry for all U-Net variants. Entry point: build_unet_variant(model_config)."""
import json
from pathlib import Path

import torch
import torch.nn as nn

from models.autoencoder_unet import UnetAutoencoder
from models.spectral_resunet import SpectralResUnet
from models.complex_unet import ComplexCRMUnet, ComplexSTFTUnet
from models.waveunet1d import WaveUNet1D

SUPPORTED_ARCHITECTURES = {
    "spectral_unet",
    "spectral_resunet",
    "complex_crm_unet",
    "complex_stft_unet",
    "waveunet1d",
}


def _in_channels(cfg: dict) -> int:
    domain = cfg.get("input_domain", "mag")
    if domain == "real_imag":
        return 2
    if domain == "real_imag_mag":
        return 3
    return 1


def build_unet_variant(model_config: dict, input_shape: tuple | None = None) -> nn.Module:
    """
    Instantiate a U-Net variant from a model_config dict.

    Args:
        model_config: dict with at least 'architecture' key (defaults to 'spectral_unet').
        input_shape: (F, TT) spectrogram shape required by 2-D models.
                     Not used for 'waveunet1d'.

    Returns:
        Instantiated nn.Module.
    """
    arch = model_config.get("architecture", "spectral_unet")
    if arch not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'. Supported: {sorted(SUPPORTED_ARCHITECTURES)}")

    if arch == "spectral_unet":
        return UnetAutoencoder(
            input_shape=input_shape,
            in_channels=_in_channels(model_config),
            out_channels=1,
            base_channels=model_config.get("base_channels", 32),
            depth=model_config.get("depth", 3),
            pooling_mode=model_config.get("pooling_mode", "isotropic"),
            output_mode=model_config.get("output_mode", "mask_scaled_sigmoid"),
            mask_max=model_config.get("mask_max", 3.0),
            softplus_max=model_config.get("softplus_max", 3.0),
        )

    if arch == "spectral_resunet":
        return SpectralResUnet(
            input_shape=input_shape,
            in_channels=_in_channels(model_config),
            base_channels=model_config.get("base_channels", 32),
            depth=model_config.get("depth", 3),
            pooling_mode=model_config.get("pooling_mode", "isotropic"),
            output_mode=model_config.get("output_mode", "mask_scaled_sigmoid"),
            mask_max=model_config.get("mask_max", 3.0),
            softplus_max=model_config.get("softplus_max", 3.0),
            bottleneck=model_config.get("bottleneck", "standard"),
            dilation_rates=tuple(model_config.get("dilation_rates", [1, 2, 4])),
            skip_attention=model_config.get("skip_attention", False),
        )

    if arch == "complex_crm_unet":
        return ComplexCRMUnet(
            input_shape=input_shape,
            in_channels=3,
            base_channels=model_config.get("base_channels", 32),
            depth=model_config.get("depth", 3),
            pooling_mode=model_config.get("pooling_mode", "isotropic"),
            crm_scale=model_config.get("crm_scale", 0.5),
        )

    if arch == "complex_stft_unet":
        return ComplexSTFTUnet(
            input_shape=input_shape,
            in_channels=3,
            base_channels=model_config.get("base_channels", 32),
            depth=model_config.get("depth", 3),
            pooling_mode=model_config.get("pooling_mode", "isotropic"),
        )

    if arch == "waveunet1d":
        return WaveUNet1D(
            signal_len=model_config.get("signal_len", 1024),
            base_channels=model_config.get("base_channels", 32),
            depth=model_config.get("depth", 4),
            bottleneck=model_config.get("bottleneck", "standard"),
            dilation_rates=tuple(model_config.get("dilation_rates", [1, 2, 4, 8])),
        )

    raise ValueError(f"Architecture '{arch}' matched but not handled.")


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_from_dir(exp_dir: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """
    Reconstruct and load a saved model from an experiment directory.

    Looks for model_config.json and model_best_snr.pth in exp_dir.
    Returns (model, model_config).
    """
    cfg_path = exp_dir / "model_config.json"
    weights_path = exp_dir / "model_best_snr.pth"
    if not cfg_path.exists():
        raise FileNotFoundError(f"model_config.json not found in {exp_dir}")
    with open(cfg_path) as f:
        model_config = json.load(f)
    input_shape = tuple(model_config.get("input_shape", [65, 33]))
    model = build_unet_variant(model_config, input_shape)
    if weights_path.exists():
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    return model.to(device), model_config
