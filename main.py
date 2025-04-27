import speech_recognition as sr
import numpy as np
import wave
import matplotlib.pyplot as plt

import audio_processing as ap

# Main program
r = sr.Recognizer()
m = sr.Microphone()

try:
    with m as source:
        print("Adjusting for ambient noise... Please wait.")
        r.adjust_for_ambient_noise(source)
        print(f"Minimum energy threshold set to {r.energy_threshold}")

    while True:
        print("Say something...")
        with m as source:
            audio = r.listen(source)

            # Save the audio as a .wav file
            with open("output.wav", "wb") as f:
                f.write(audio.get_wav_data())

            # Read and process the .wav file
            signal, frame_rate = ap.read_wav("output.wav")
            frame_size = 256

            # Recognize the speech
            print("Recognizing speech...")
            try:
                recognized_text = r.recognize_whisper(audio)
                print(f"Recognized Text: {recognized_text}")

                num_frames = len(signal) // frame_size
                truncated_signal = signal[: num_frames * frame_size]

                # Plot the waveform with the recognized text
                ap.plot_waveform_with_text(signal, frame_rate, recognized_text)

                # Detect voiced/unvoiced regions with dynamic thresholds
                voiced_unvoiced, threshold_energy, threshold_zero_crossings = (
                    ap.detect_voiced_unvoiced(truncated_signal, frame_size)
                )

                # Plot the result with color-coded voiced/unvoiced frames
                ap.plot_voiced_unvoiced(
                    truncated_signal, voiced_unvoiced, frame_size, frame_rate
                )

            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                print(
                    f"Could not request results from Google Speech Recognition service; {e}"
                )

except KeyboardInterrupt:
    print("Exiting...")
