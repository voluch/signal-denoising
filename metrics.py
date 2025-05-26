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


class MeanAbsoluteError:
    @staticmethod
    def calculate(original_signal, denoised_signal):
        """
        Обчислює середню абсолютну помилку (MAE) між оригінальним і очищеним сигналами.

        :param original_signal: Оригінальний сигнал (1D масив).
        :param denoised_signal: Очищений сигнал (1D масив).
        :return: Значення MAE.
        """
        mae = np.mean(np.abs(original_signal - denoised_signal))
        return mae


class RootMeanSquaredError:
    @staticmethod
    def calculate(original_signal, denoised_signal):
        """
        Обчислює корінь середньоквадратичної помилки (RMSE) між оригінальним і очищеним сигналами.

        :param original_signal: Оригінальний сигнал (1D масив).
        :param denoised_signal: Очищений сигнал (1D масив).
        :return: Значення RMSE.
        """
        rmse = np.sqrt(np.mean((original_signal - denoised_signal) ** 2))
        return rmse


class SignalToNoiseRatio:
    @staticmethod
    def calculate(original_signal, denoised_signal):
        """
        Обчислює співвідношення сигнал/шум (SNR) між оригінальним і очищеним сигналами в децибелах (dB).

        :param original_signal: Оригінальний сигнал (1D масив).
        :param denoised_signal: Очищений сигнал (1D масив).
        :return: Значення SNR у dB.
        """
        noise = original_signal - denoised_signal
        signal_power = np.sum(original_signal ** 2)
        noise_power = np.sum(noise ** 2)

        # Захист від ділення на нуль
        if noise_power == 0:
            return np.inf

        snr = 10 * np.log10(signal_power / noise_power)
        return snr
