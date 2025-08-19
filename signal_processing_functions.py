import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
# import torch.nn as nn
import numpy as np

from scipy.signal import argrelextrema
# from sklearn.impute import KNNImputer
from scipy.interpolate import CubicSpline

def lowpass_filter(data, fpass, fstop, apass, astop, fs):
    fpass_norm = fpass / (fs / 2)
    fstop_norm = fstop / (fs / 2)
    order, wn = signal.buttord(fpass_norm, fstop_norm, apass, astop, analog=False)
    b = signal.firwin(order + 1, wn, window='hamming', pass_zero=True)
    filtered_data = signal.filtfilt(b, [1], data)
    return filtered_data
