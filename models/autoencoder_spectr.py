import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from scipy.signal import stft, istft


# Генерація тестового сигналу
def generate_signal(t):
    signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # Сума синусоїд
    noise = np.random.normal(0, 0.5, signal.shape)  # Шум
    return signal, signal + noise


# Перетворення сигналу в спектрограму
def signal_to_spectrogram(signal, fs=1024):
    nperseg = min(256, len(signal))  # Динамічне встановлення nperseg
    noverlap = max(0, nperseg // 2)  # noverlap має бути меншим за nperseg
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Zxx), Zxx


# Відновлення сигналу з спектрограми
def spectrogram_to_signal(Zxx, fs=1024):
    nperseg = min(256, Zxx.shape[1])  # Динамічне встановлення nperseg
    noverlap = max(0, nperseg // 2)
    _, reconstructed_signal = istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return reconstructed_signal


# Dataset для роботи з автоенкодером
class SpectrogramDataset(Dataset):
    def __init__(self, clean_signals, noisy_signals):
        self.clean_spectrograms = np.array([signal_to_spectrogram(sig)[2] for sig in clean_signals])
        self.noisy_spectrograms = np.array([signal_to_spectrogram(sig)[2] for sig in noisy_signals])

        # Перетворення у тензори
        self.clean_spectrograms = torch.tensor(self.clean_spectrograms, dtype=torch.float32).unsqueeze(1)
        self.noisy_spectrograms = torch.tensor(self.noisy_spectrograms, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.clean_spectrograms)

    def __getitem__(self, idx):
        return self.noisy_spectrograms[idx], self.clean_spectrograms[idx]


# Автоенкодер для спектрограм
class SpectrogramAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(SpectrogramAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.Sigmoid()
        )
        self.input_shape = input_shape

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Обрізаємо або інтерполюємо, щоб повернути оригінальний розмір
        if decoded.shape[-2:] != self.input_shape:
            decoded = nn.functional.interpolate(decoded, size=self.input_shape, mode='bilinear', align_corners=False)
        return decoded


# Тренування автоенкодера
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    t = np.linspace(0, 1, 1024, endpoint=False)
    clean_signals, noisy_signals = zip(*[generate_signal(t) for _ in range(100)])
    dataset = SpectrogramDataset(clean_signals, noisy_signals)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = dataset.clean_spectrograms.shape[2:]
    model = SpectrogramAutoencoder(input_shape).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

    # Тестування моделі
    model.eval()
    with torch.no_grad():
        sample_noisy, sample_clean = dataset[0]
        sample_noisy, sample_clean = sample_noisy.unsqueeze(0).to(device), sample_clean.unsqueeze(0).to(device)
        denoised_spectrogram = model(sample_noisy).squeeze(0).cpu().numpy()

        # Відновлення сигналу з очищеної спектрограми
        _, _, _, orig_phase = signal_to_spectrogram(clean_signals[0])  # Отримуємо фазову інформацію
        reconstructed_signal = spectrogram_to_signal(denoised_spectrogram * np.exp(1j * np.angle(orig_phase)))

        # Перетворення sample_clean назад у часовий ряд
        _, _, _, clean_phase = signal_to_spectrogram(sample_clean.squeeze(0).cpu().numpy())
        clean_reconstructed = spectrogram_to_signal(
            sample_clean.squeeze(0).cpu().numpy() * np.exp(1j * np.angle(clean_phase)))

        # Обчислення MSE між оригінальним і відновленим часовими сигналами
        mse = criterion(torch.tensor(reconstructed_signal), torch.tensor(clean_reconstructed)).item()
        print(f"Test MSE: {mse:.6f}")

        # Візуалізація результату
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(sample_clean.squeeze().cpu().numpy(), aspect='auto', cmap='hot')
        plt.title("Original Spectrogram")
        plt.subplot(1, 3, 2)
        plt.imshow(sample_noisy.squeeze().cpu().numpy(), aspect='auto', cmap='hot')
        plt.title("Noisy Spectrogram")
        plt.subplot(1, 3, 3)
        plt.imshow(denoised_spectrogram.squeeze(), aspect='auto', cmap='hot')
        plt.title("Denoised Spectrogram")
        plt.show()

        # Візуалізація часових сигналів
        plt.subplot(2, 1, 2)
        plt.plot(clean_signals[0], label="Original Signal")
        if len(reconstructed_signal) < len(clean_signals[0]):
            reconstructed_signal = np.pad(reconstructed_signal, (0, len(clean_signals[0]) - len(reconstructed_signal)),
                                      mode='edge')
        else:
            reconstructed_signal = reconstructed_signal[:len(clean_signals[0])]
        plt.plot(reconstructed_signal.squeeze(), label="Denoised Signal", linestyle='dashed')
        plt.legend()
        plt.title("Time Domain Signals")

        plt.show()
