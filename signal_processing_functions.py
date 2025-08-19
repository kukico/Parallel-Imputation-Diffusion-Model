import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
# import torch.nn as nn
import numpy as np


# from pyod.models.copod import COPOD
# from reparo import HotDeckImputation
from scipy.signal import argrelextrema
# from sklearn.impute import KNNImputer
from scipy.interpolate import CubicSpline
# from sklearn.decomposition import FastICA
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from PyEMD import EMD,EEMD,CEEMDAN

def lowpass_filter(data, fpass, fstop, apass, astop, fs):
    fpass_norm = fpass / (fs / 2)
    fstop_norm = fstop / (fs / 2)
    order, wn = signal.buttord(fpass_norm, fstop_norm, apass, astop, analog=False)
    b = signal.firwin(order + 1, wn, window='hamming', pass_zero=True)
    filtered_data = signal.filtfilt(b, [1], data)
    return filtered_data

# def get_envelopes(data):
#     max_extrema = argrelextrema(data, np.greater)[0]
#     min_extrema = argrelextrema(data, np.less)[0]

#     upper_env = CubicSpline(max_extrema, data[max_extrema], bc_type='natural')(np.arange(len(data)))
#     lower_env = CubicSpline(min_extrema, data[min_extrema], bc_type='natural')(np.arange(len(data)))
#     env = (upper_env + lower_env) / 2
#     return env

# def combine_close_indices(indices,threshold1,threshold2):
#     indices = sorted(indices)  # Sort the indices
#     combined_indices = []
#     start = indices[0] - threshold1
#     end = indices[0] + threshold1
    
#     for i in range(1, len(indices)):
#         if indices[i] - end <= threshold2:  # Check if the difference is small
#             end = indices[i] + threshold1
#         else:
#             combined_indices.append([start, end])
#             start = indices[i] - threshold1
#             end = indices[i] + threshold1
    
#     # Append the last range
#     combined_indices.append([start, end])
    
#     return combined_indices

# def detect_outliers(data,threshold_outlier,threshold_combine1,threshold_combine2,n_jobs=1):
#     data_replica = data.reshape(-1, 1)
#     clf = COPOD(n_jobs=n_jobs, contamination=threshold_outlier)
#     clf.fit(data_replica)
#     y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
#     outlier_indices = np.where(y_train_pred == 1)[0]
#     combined_indices = combine_close_indices(outlier_indices,threshold_combine1,threshold_combine2)
#     return np.array(combined_indices)