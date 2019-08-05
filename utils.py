import wave
import pyaudio
import numpy as np

def open_stream(fs):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    return p, stream


def generate_tone(fs, freq, duration):
    npsin = np.sin(2 * np.pi * np.arange(fs*duration) * freq / fs)
    samples = npsin.astype(np.float32)
    return 0.1 * samples