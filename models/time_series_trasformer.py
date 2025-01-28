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

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=4, dim_feedforward=512):
        """
        Трансформер для знешумлення часових рядів.

        :param input_dim: Розмір вхідного сигналу.
        :param d_model: Розмірність моделі (embedding size).
        :param nhead: Кількість голов у багатоголовій самоувазі.
        :param num_layers: Кількість шарів трансформера.
        :param dim_feedforward: Розмірність feedforward шару.
        """
        super(TimeSeriesTransformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, d_model))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        Прямий прохід трансформера.

        :param x: Вхідний зашумлений сигнал (batch_size, sequence_length, input_dim).
        :return: Очищений сигнал (batch_size, sequence_length, input_dim).
        """
        x = self.input_projection(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        return x

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
    model = TimeSeriesTransformer(input_dim=1).to(device)

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

            # Зворотній прохід
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
        plt.title("Signal Denoising with Time Series Transformer")
        plt.show()
