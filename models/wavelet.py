import numpy as np
import pywt
import matplotlib.pyplot as plt
from metrics import MeanSquaredError

class WaveletDenoising:
    def __init__(self, wavelet='db1', level=None):
        """
        Ініціалізація класу для знешумлення сигналів за допомогою вейвлетів.

        :param wavelet: Назва вейвлета (наприклад, 'db1', 'sym5').
        :param level: Рівень декомпозиції (None означає автоматичний вибір).
        """
        self.wavelet = wavelet
        self.level = level

    def denoise(self, signal):
        """
        Виконує вейвлет-знешумлення сигналу.

        :param signal: Вхідний зашумлений сигнал (1D масив).
        :return: Очищений сигнал.
        """
        # Виконуємо вейвлет-декомпозицію
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)

        # Порогова обробка коефіцієнтів
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Оцінка шуму
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        # М'яке порогове обнулення
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

        # Реконструкція сигналу
        denoised_signal = pywt.waverec(coeffs_thresholded, self.wavelet)

        return denoised_signal

    def visualize(self, original_signal, noisy_signal, denoised_signal):
        """
        Візуалізує оригінальний, зашумлений і очищений сигнал.

        :param original_signal: Оригінальний сигнал (1D масив).
        :param noisy_signal: Зашумлений сигнал (1D масив).
        :param denoised_signal: Очищений сигнал (1D масив).
        """
        plt.figure(figsize=(15, 8))

        plt.subplot(3, 1, 1)
        plt.plot(original_signal, label="Original Signal")
        plt.title("Original Signal")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(noisy_signal, label="Noisy Signal")
        plt.title("Noisy Signal")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(denoised_signal, label="Denoised Signal")
        plt.title("Denoised Signal")
        plt.legend()

        plt.tight_layout()
        plt.show()

# Приклад використання
if __name__ == "__main__":
    # Генерація тестового сигналу
    np.random.seed(0)
    t = np.linspace(0, 1, 1000, endpoint=False)
    clean_signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    noise = np.random.normal(0, 0.5, len(t))
    noisy_signal = clean_signal + noise

    # Ініціалізація та використання вейвлет-знешумлення
    wavelet_denoiser = WaveletDenoising(wavelet='db4', level=4)
    denoised_signal = wavelet_denoiser.denoise(noisy_signal)

    # Обчислення MSE
    mse = MeanSquaredError.calculate(clean_signal, denoised_signal)
    print(f"Mean Squared Error (MSE): {mse:.6f}")

    # Візуалізація результатів
    wavelet_denoiser.visualize(clean_signal, noisy_signal, denoised_signal)
