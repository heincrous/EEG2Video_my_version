import numpy as np
import math
from scipy.fftpack import fft

def DE_PSD(data, fre, time_window):
    """
    Compute Differential Entropy (DE) and Power Spectral Density (PSD) features.

    Parameters
    ----------
    data : np.ndarray
        EEG segment with shape (n_channels, n_timepoints).
    fre : int
        Sampling frequency (Hz).
    time_window : int
        Window length in seconds.

    Returns
    -------
    de : np.ndarray
        Differential Entropy features, shape (n_channels, 5).
    psd : np.ndarray
        Power Spectral Density features, shape (n_channels, 5).
    """

    STFTN = 200
    fStart = [1, 4, 8, 14, 31]
    fEnd = [4, 8, 14, 31, 99]
    fs = fre
    Hlength = time_window * fs

    # Precompute frequency bin ranges
    fStartNum = [int(start/fs*STFTN) for start in fStart]
    fEndNum = [int(end/fs*STFTN) for end in fEnd]

    n, m = data.shape
    psd = np.zeros((n, len(fStart)))
    de = np.zeros((n, len(fStart)))

    # Hanning window
    Hwindow = np.array([0.5 - 0.5 * np.cos(2 * np.pi * k / (Hlength+1)) for k in range(1, Hlength+1)])

    for j in range(n):
        temp = data[j]
        Hdata = temp * Hwindow
        FFTdata = fft(Hdata, STFTN)
        magFFTdata = np.abs(FFTdata[:STFTN//2])

        for p in range(len(fStart)):
            E = np.mean(magFFTdata[fStartNum[p]-1:fEndNum[p]+1]**2)
            psd[j, p] = E
            de[j, p] = math.log(100*E, 2)

    return de, psd
