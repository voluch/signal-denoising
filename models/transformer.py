import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

class SignalDataset(Dataset):
    def __init__(self, clean_signals, noisy_signals):
        """
        Dataset для зберігання сигналів.
        :param clean_signals: Чисті сигнали (numpy array).
        :param noisy_signals: Зашумлені сигнали (numpy array).
        """
        self.clean_signals = torch.tensor(clean_signals, dtype=torch.float32)
        self.noisy_signals = torch.tensor(noisy_signals, dtype=torch.float32)

    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        return self.noisy_signals[idx], self.clean_signals[idx]

class TransformerDenoiser(nn.Module):
    def __init__(self, signal_length, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512):
        """
        Трансформер для знешумлення сигналів.

        :param signal_length: Довжина вхідного сигналу.
        :param d_model: Розмірність моделі (embedding size).
        :param nhead: Кількість голов у багатоголовій самоувазі.
        :param num_encoder_layers: Кількість шарів енкодера.
        :param num_decoder_layers: Кількість шарів декодера.
        :param dim_feedforward: Розмірність feedforward шару.
        """
        super(TransformerDenoiser, self).__init__()
        self.signal_length = signal_length
        self.embedding = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, signal_length, d_model))

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, noisy_signal):
        """
        Прямий прохід трансформера.

        :param noisy_signal: Вхідний зашумлений сигнал (batch_size, signal_length, 1).
        :return: Очищений сигнал (batch_size, signal_length, 1).
        """
        # Додаємо позиційне кодування
        embedded = self.embedding(noisy_signal) + self.positional_encoding

        # Трансформерний блок
        denoised = self.transformer(embedded, embedded)

        # Перетворення в один канал
        output = self.output_layer(denoised)
        return output

# Тренування моделі
if __name__ == "__main__":
    # Генерація даних
    np.random.seed(42)
    torch.manual_seed(42)

    t = np.linspace(0, 1, 1024, endpoint=False)
    clean_signals = np.array([np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) for _ in range(100)])
    noise = np.random.normal(0, 0.5, clean_signals.shape)
    noisy_signals = clean_signals + noise

    # Підготовка датасету
    dataset = SignalDataset(noisy_signals[:, :, None], clean_signals[:, :, None])
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Ініціалізація моделі
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerDenoiser(signal_length=1024).to(device)

    # Оптимізатор і функція втрат
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Тренування
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Прямий прохід
            outputs = model(noisy)
            loss = criterion(outputs, clean)

            # Зворотний прохід
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

    # Тестування
    model.eval()
    with torch.no_grad():
        sample_noisy, sample_clean = dataset[0]
        sample_noisy, sample_clean = sample_noisy.to(device).unsqueeze(0), sample_clean.to(device).unsqueeze(0)
        denoised_signal = model(sample_noisy).squeeze(0).cpu().numpy()

        # Обчислення MSE
        mse = criterion(torch.tensor(denoised_signal).to(device), sample_clean).item()
        print(f"Test MSE: {mse:.6f}")

        # Візуалізація
        plt.figure(figsize=(12, 6))
        plt.plot(sample_clean.squeeze(0).cpu().numpy(), label="Original Signal")
        plt.plot(sample_noisy.squeeze(0).cpu().numpy(), label="Noisy Signal")
        plt.plot(denoised_signal, label="Denoised Signal")
        plt.legend()
        plt.title("Signal Denoising with Transformer")
        plt.show()
