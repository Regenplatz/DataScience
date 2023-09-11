#!/usr/bin/env python

__author__ = "WhyKiki"
__version__ = "1.0.0"


import numpy as np
import simpleaudio as sa
from scipy import signal
from scipy.io import wavfile
from scipy.fft import fftfreq, rfft, irfft
import matplotlib.pyplot as plt
import time


def readSample(pathToAudioSample):
    """
    Read audio sample and assign data, frequency, duration, time to variables
    :param pathToAudioSample: String, path to audio file
    :return: frequency, data, duration, time
    """
    ## read sample
    fs, data = wavfile.read(pathToAudioSample)

    ## assign duration and time
    duration = len(data)/fs
    t = np.arange(0, duration, 1/fs)

    return fs, data, duration, t


def plotAmplitude(t, data):
    """
    Plot audio sequence as amplitude
    :return: None
    """
    plt.figure(figsize=(16, 8))
    plt.plot(t, data)
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")


def plotCorrelation(start, end, data, fs, t):
    """
    Plot audio correlations
    :param start: float, start of audio signal in seconds
    :param end: float, end of audio signal in seconds
    :return: dictionary of numpy arrays, the section of interested extracted from the whole audio sample
    """
    sec_start = start
    sec_end = end

    ## extract time sequence of interest from audio sample
    section = data[int(sec_start * fs):int(sec_end * fs)]

    ## evaluate signal correlations
    corr = signal.fftconvolve(data, section, mode='same')

    ## plot correlations
    plt.plot(t, corr)
    plt.ylabel('correlation')
    plt.xlabel('time [sec]')

    section_dict = {"audio_section": section,
                    "sec_times": [sec_start, sec_end]}

    return section_dict


def plotSpectrogram(t, f, S):
    """
    Plot spectrogram
    :param t: time
    :param f: frequency
    :param S: amplitude
    :return: None
    """
    plt.pcolormesh(t, f, S)
    plt.ylabel('frequency [Hz]')
    plt.xlabel('time [sec]')


def transformAudio(audio_section, time_start, time_end, lower_bond, upper_bond, fs, play=True):
    """
    Transform audio sample to frequency domain and afterwards back to time domain
    :param audio_section: numpy array, audio sample (which might be an extract
                          of the whole sample or the whole sample itself)
    :param time_start: number, start of audio sample in seconds
    :param time_end: number, end of audio sample in seconds
    :param lower_bond: integer, lower frequency limit to be set
    :param upper_bond: integer, upper frequency limit to be set
    :param play: boolean, if audio sample should be played or not
    :return: None
    """

    ## transform to frequency domain
    S = rfft(audio_section)
    f = fftfreq(audio_section.size, 1 / fs)[:audio_section.size // 2]

    ## amplitude boundaries
    low = np.max(np.argwhere(f.real < lower_bond))
    high = np.min(np.argwhere(f.real > upper_bond))

    ## cut out everything outside boundaries
    S[:low] = 0
    S[high:] = 0

    plt.figure(figsize=(16, 10))
    ## plot frequencies
    plt.subplot(2, 1, 1)
    plt.plot(f, S[:-1])
    plt.xlim(lower_bond - 10, upper_bond + 10)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("amplitude")
    plt.show

    ## transform back to time domain
    sf = irfft(S)
    sf = sf * (2 ** 15 - 1) / np.max(np.abs(sf))

    ## evaluate time window for plotting
    section_duration = len(audio_section) / fs
    time_section = np.arange(0, section_duration, 1 / fs)

    ## plot time series
    plt.subplot(2, 1, 2)
    plt.plot(time_section + time_start, sf)
    plt.xlim(time_start, time_end)
    plt.ylim(-30000, 30000)
    plt.xlabel("time [s]")
    plt.ylabel("amplitude")
    plt.tight_layout()
    plt.show

    if play:
        sa.play_buffer(sf.astype(np.int16), 1, 2, fs)
        time.sleep(section_duration)


def main():
    pass


if __name__ == "__main__":
    main()
