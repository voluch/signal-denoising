import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Constants for fixed-length signals and STFT parameters
FS = 1024               # Sampling rate
SIGNAL_LEN = 1024       # All signals will be exactly this long
NPERSEG = 256           # STFT window length
NOVERLAP = NPERSEG // 2 # STFT overlap
PAD = NPERSEG // 2

# ─────────────────────────────────────────────────────────────────────────────
# 1) DATA GENERATION & TRANSFORMS

def generate_signal():
    """
    Generate a clean sinusoidal signal of length SIGNAL_LEN,
    plus its noisy version (Gaussian noise).
    """
    t = np.linspace(0, 1, SIGNAL_LEN, endpoint=False)
    clean = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)
    noise = np.random.normal(0, 0.5, clean.shape)
    noisy = clean + noise
    return clean, noisy


def signal_to_spectrogram(signal):
    """
    1) дзеркально допадимо по краях
    2) зробимо STFT
    """
    sig = np.pad(signal, PAD, mode='reflect')
    _, _, Zxx = stft(sig, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    mag = np.abs(Zxx)
    return mag, Zxx

def spectrogram_to_signal(Zxx_complex):
    """
    1) зробимо ISTFT
    2) обріжемо паддінг назад
    """
    _, rec = istft(Zxx_complex, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    # обрізаємо віддзеркалені PAD точок на початку й кінці
    rec = rec[PAD : PAD + SIGNAL_LEN]
    return rec

# ─────────────────────────────────────────────────────────────────────────────
# 2) DATASET

class SpectrogramDataset(Dataset):
    """
    Holds paired (noisy_spec, clean_spec) tensors,
    all spectrograms share the same shape.
    """
    def __init__(self, n_samples):
        clean_list, noisy_list = [], []
        for _ in range(n_samples):
            clean, noisy = generate_signal()
            clean_mag, _ = signal_to_spectrogram(clean)
            noisy_mag, _ = signal_to_spectrogram(noisy)
            clean_list.append(clean_mag)
            noisy_list.append(noisy_mag)
        # Convert to single numpy arrays, then to torch tensors
        clean_arr = np.stack(clean_list, axis=0)   # (N, F, T)
        noisy_arr = np.stack(noisy_list, axis=0)
        # add channel dim
        self.clean = torch.from_numpy(clean_arr).unsqueeze(1).float()
        self.noisy = torch.from_numpy(noisy_arr).unsqueeze(1).float()

    def __len__(self):
        return self.clean.shape[0]

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]

# ─────────────────────────────────────────────────────────────────────────────
# 3) MODEL

class SpectrogramVAE(nn.Module):
    def __init__(self, freq_bins, time_frames, latent_dim=128):
        super().__init__()
        self.freq_bins, self.time_frames = freq_bins, time_frames
        self.latent_dim = latent_dim

        # Encoder conv stack
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), nn.ReLU()
        )
        # Run dummy through encoder to get shape
        with torch.no_grad():
            dummy = torch.zeros(1,1,freq_bins,time_frames)
            enc_out = self.encoder_conv(dummy)
            _, c_enc, h_enc, w_enc = enc_out.shape
            self._flat_dim = c_enc * h_enc * w_enc
            self._h_enc, self._w_enc = h_enc, w_enc

        # Latent space
        self.fc_mu     = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

        # Decoder fc + deconv stack
        self.fc_dec = nn.Linear(latent_dim, self._flat_dim)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,3,2,1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,3,2,1,output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,1,3,2,1,output_padding=1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        # Decode
        d = self.fc_dec(z).view(-1, 128, self._h_enc, self._w_enc)
        out = self.decoder_deconv(d)
        # Ensure exact size
        if out.shape[-2:] != (self.freq_bins, self.time_frames):
            out = nn.functional.interpolate(
                out,
                size=(self.freq_bins, self.time_frames),
                mode='bilinear',
                align_corners=False
            )
        return out, mu, logvar

# ─────────────────────────────────────────────────────────────────────────────
# 4) TRAINING FUNCTION

def train_vae(n_samples=1000, batch_size=16, lr=1e-3, epochs=30):
    dataset = SpectrogramDataset(n_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    fb, tf = dataset.clean.shape[2], dataset.clean.shape[3]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectrogramVAE(fb, tf).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss(reduction='sum')

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            recon, mu, logvar = model(noisy)
            # reconstruction loss
            rec_loss = mse_loss(recon, clean)
            # KL divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (rec_loss + kl_loss) / noisy.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d}/{epochs}  Loss: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), 'spectrogram_vae.pth')
    return model, device

# ─────────────────────────────────────────────────────────────────────────────
# 5) INFERENCE ON NEW SIGNAL

def predict_new(raw_signal, model, device):
    model.eval()
    # spectrogram + phase
    mag, Zxx = signal_to_spectrogram(raw_signal)
    phase = np.angle(Zxx)
    # to tensor
    inp = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        denoised_mag = model(inp).squeeze(0).squeeze(0).cpu().numpy()
    # reconstruct complex spectrogram and invert
    denoised_Zxx = denoised_mag * np.exp(1j*phase)
    rec = spectrogram_to_signal(denoised_Zxx)
    # ensure 1D
    return np.asarray(rec).flatten()

def denoise_with_vae(raw_signal, model, device):
    mag, Zxx = signal_to_spectrogram(raw_signal)
    phase = np.angle(Zxx)
    inp = torch.from_numpy(mag).unsqueeze(0).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        out_mag, _, _ = model(inp)
    out_mag = out_mag.squeeze().cpu().numpy()
    denoised_Zxx = out_mag * np.exp(1j*phase)
    return spectrogram_to_signal(denoised_Zxx)

# ─────────────────────────────────────────────────────────────────────────────
# 6) RUN

if __name__ == '__main__':
    # train (or load) model
    vae_model, dev = train_vae(n_samples=1000, batch_size=32, lr=1e-3, epochs=30)


    # generate test signal
    clean, noisy = generate_signal()
    denoised = denoise_with_vae(noisy, vae_model, dev)

    mse_val = np.mean((denoised - clean) ** 2)
    print(f"Test MSE: {mse_val:.6f}")

    t = np.linspace(0, 1, SIGNAL_LEN, endpoint=False)
    plt.figure(figsize=(10, 5))
    plt.plot(t, clean, label='Clean')
    plt.plot(t, noisy, label='Noisy', alpha=0.5)
    plt.plot(t, denoised, label='Denoised', linestyle='--')
    plt.legend()
    # plt.xlim(0.4, 0.5)
    # plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.show()
