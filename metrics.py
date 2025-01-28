import numpy as np


class MeanSquaredError:
    @staticmethod
    def calculate(original_signal, denoised_signal):
        """
        Обчислює середньоквадратичну помилку (MSE) між оригінальним і очищеним сигналами.

        :param original_signal: Оригінальний сигнал (1D масив).
        :param denoised_signal: Очищений сигнал (1D масив).
        :return: Значення MSE.
        """
        mse = np.mean((original_signal - denoised_signal) ** 2)
        return mse
