import numpy as np
import wave
import matplotlib.pyplot as plt

frame_size = 256


# Function to read a .wav file
def read_wav(file_path):
    with wave.open(file_path, "r") as wav_file:
        signal = wav_file.readframes(-1)
        signal = np.frombuffer(signal, dtype=np.int16)
        frame_rate = wav_file.getframerate()
    return signal, frame_rate


# Function to plot the waveform with text annotations
def plot_waveform_with_text(signal, frame_rate, recognized_text):
    # Create time axis
    time = np.linspace(0, len(signal) / frame_rate, num=len(signal))

    # Plot the waveform
    plt.figure(figsize=(15, 6))
    plt.plot(time, signal, label="Audio Signal", color="blue")
    plt.title("Waveform with Recognized Text")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Annotate the waveform with text
    text_duration = len(signal) / frame_rate  # Total duration of the audio
    text_positions = np.linspace(
        0, text_duration, num=len(recognized_text)
    )  # Spread text over time

    for i, char in enumerate(recognized_text):
        plt.text(text_positions[i], max(signal) * 0.8, char, fontsize=12, color="red")

    plt.legend()
    plt.show()


def calculate_energy(signal, frame_size):
    energy = np.sum(np.square(signal.reshape(-1, frame_size)), axis=1)
    return energy


def calculate_zero_crossings_per_frame(signal, frame_size):
    frames = signal.reshape(-1, frame_size)
    zero_crossings = np.array(
        [np.sum(np.abs(np.diff(np.sign(frame))) > 0) for frame in frames]
    )
    return zero_crossings


def dynamic_thresholds(
    energy, zero_crossings, energy_factor=1.5, zero_crossing_factor=0.8
):
    # Compute dynamic thresholds based on signal characteristics
    threshold_energy = np.mean(energy) * energy_factor  # Scale by energy factor
    threshold_zero_crossings = (
        np.mean(zero_crossings) * zero_crossing_factor
    )  # Scale by zero-crossing factor
    return threshold_energy, threshold_zero_crossings


def detect_voiced_unvoiced(
    signal, frame_size, energy_factor=1.5, zero_crossing_factor=0.8
):
    energy = calculate_energy(signal, frame_size)
    zero_crossings = calculate_zero_crossings_per_frame(signal, frame_size)

    # Compute dynamic thresholds
    threshold_energy, threshold_zero_crossings = dynamic_thresholds(
        energy, zero_crossings, energy_factor, zero_crossing_factor
    )

    # Voiced = High energy AND Low zero-crossings
    voiced = (energy > threshold_energy) & (zero_crossings < threshold_zero_crossings)

    return voiced, threshold_energy, threshold_zero_crossings


import numpy as np
import matplotlib.pyplot as plt


def plot_voiced_unvoiced(signal, voiced, frame_size, frame_rate):
    # Plot the signal with different colors for voiced (True) and unvoiced (False)
    time = np.linspace(0, len(signal) / frame_rate, num=len(signal))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Voiced/Unvoiced Signal Detection")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    # Initialize flags to track whether the labels have been added
    voiced_label_added = False
    unvoiced_label_added = False

    # Plot each frame with corresponding color (voiced or unvoiced)
    num_frames = len(voiced)
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        if voiced[i]:
            # Plot voiced segments, adding label only for the first voiced segment
            if not voiced_label_added:
                ax.plot(
                    time[start:end],
                    signal[start:end],
                    label="Voiced Zone",
                    color="green",
                )
                voiced_label_added = True
            else:
                ax.plot(
                    time[start:end], signal[start:end], color="green"
                )  # Voiced = Green
        else:
            # Plot unvoiced segments, adding label only for the first unvoiced segment
            if not unvoiced_label_added:
                ax.plot(
                    time[start:end],
                    signal[start:end],
                    label="Unvoiced Zone",
                    color="red",
                )
                unvoiced_label_added = True
            else:
                ax.plot(
                    time[start:end], signal[start:end], color="red"
                )  # Unvoiced = Red

    # Display the legend
    ax.legend()

    plt.show()
