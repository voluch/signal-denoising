import torch
import torch.nn as nn


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
        assert x.dim() == 3, f"Expected input shape (B, T, F), got {x.shape}"
        if x.shape[1] < x.shape[2]:  # Shape is likely (B, F, T)
            x = x.permute(0, 2, 1)  # Change to (B, T, F)
        x = self.input_projection(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        return x
