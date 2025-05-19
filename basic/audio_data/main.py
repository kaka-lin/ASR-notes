import os

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Create the directory if it doesn't exist
    os.makedirs("basic/audio_data/images", exist_ok=True)

    # Set the output path for all plots
    waveform_save_path = "basic/audio_data/images/waveform.png"
    spectrum_save_path = "basic/audio_data/images/spectrum.png"
    spectrogram_save_path = "basic/audio_data/images/spectrogram.png"
    mel_spectrogram_save_path = "basic/audio_data/images/mel_spectrogram.png"

    # Load the audio file
    array, sampling_rate = librosa.load(librosa.ex("trumpet"))

    # 1. Plot the waveform
    plt.figure().set_figwidth(12)
    librosa.display.waveshow(array, sr=sampling_rate)
    plt.savefig(waveform_save_path)
    plt.show()
    print(f"已將波形圖儲存至 {waveform_save_path}")

    # 2. Plot the spectrum
    # get the first 4096 samples
    dft_input = array[:4096]

    # calculate the DFT
    window = np.hanning(len(dft_input))
    windowed_input = dft_input * window
    dft = np.fft.rfft(windowed_input)

    # get the amplitude spectrum in decibels
    amplitude = np.abs(dft)
    amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

    # get the frequency bins
    frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

    plt.figure().set_figwidth(12)
    plt.plot(frequency, amplitude_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.xscale("log")
    plt.savefig(spectrum_save_path)
    plt.show()

    # 3. Plot the spectrogram
    D = librosa.stft(array)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure().set_figwidth(12)
    librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
    plt.colorbar()
    plt.savefig(spectrogram_save_path)
    plt.show()

    # 4. Plot the mel spectrogram
    S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure().set_figwidth(12)
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
    plt.colorbar()
    plt.savefig(mel_spectrogram_save_path)
    plt.show()
