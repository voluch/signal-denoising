import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class SignalDataset(Dataset):
    def __init__(self, clean_signals, noisy_signals):
        """
        Ініціалізація датасету для сигналів.

        :param clean_signals: Чисті сигнали (numpy масив).
        :param noisy_signals: Зашумлені сигнали (numpy масив).
        """
        self.clean_signals = torch.tensor((clean_signals - np.mean(clean_signals)) / np.std(clean_signals), dtype=torch.float32).unsqueeze(1)
        self.noisy_signals = torch.tensor((noisy_signals - np.mean(noisy_signals)) / np.std(noisy_signals), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        return self.noisy_signals[idx], self.clean_signals[idx]

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Обрізаємо або додаємо нулі до виходу для відповідності вхідному розміру
        if decoded.size(2) != x.size(2):
            decoded = nn.functional.pad(decoded, (0, x.size(2) - decoded.size(2)))

        return decoded

class MeanSquaredError:
    @staticmethod
    def calculate(original_signal, denoised_signal):
        """
        Обчислює середньоквадратичну помилку (MSE) між оригінальним і очищеним сигналами.

        :param original_signal: Оригінальний сигнал (1D масив або тензор).
        :param denoised_signal: Очищений сигнал (1D масив або тензор).
        :return: Значення MSE.
        """
        mse = torch.mean((original_signal - denoised_signal) ** 2).item()
        return mse

# Приклад використання Convolutional Autoencoder
if __name__ == "__main__":
    # Перевірка наявності GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Генерація синтетичних сигналів для прикладу
    np.random.seed(0)
    t = np.linspace(0, 1, 1024, endpoint=False)
    clean_signals = np.array([np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) for _ in range(100)])
    noise = np.random.normal(0, 0.5, clean_signals.shape)
    noisy_signals = clean_signals + noise

    # Підготовка датасету та DataLoader
    dataset = SignalDataset(clean_signals, noisy_signals)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Ініціалізація моделі, оптимізатора та функції втрат
    model = ConvolutionalAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Тренування моделі
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Прямий прохід
            outputs = model(noisy)
            loss = criterion(outputs, clean)

            # Зворотній прохід
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step(epoch_loss / len(dataloader))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.6f}")

    # Тестування моделі
    model.eval()
    with torch.no_grad():
        sample_noisy, sample_clean = dataset[0]
        sample_noisy, sample_clean = sample_noisy.to(device), sample_clean.to(device)
        sample_noisy = sample_noisy.unsqueeze(0)  # Додаємо розмір для batch
        denoised_signal = model(sample_noisy).squeeze(0).squeeze(0).cpu().numpy()

        # Обчислення MSE
        mse = MeanSquaredError.calculate(sample_clean.squeeze(0).cpu(), torch.tensor(denoised_signal))
        print(f"Test MSE: {mse:.6f}")

        # Візуалізація результату
        plt.figure(figsize=(12, 6))
        plt.plot(sample_clean.squeeze(0).cpu().numpy(), label="Original Signal")
        plt.plot(sample_noisy.squeeze(0).squeeze(0).cpu().numpy(), label="Noisy Signal")
        plt.plot(denoised_signal, label="Denoised Signal")
        plt.legend()
        plt.title("Signal Denoising with Convolutional Autoencoder")
        plt.show()
