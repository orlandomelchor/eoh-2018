import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from scipy import stats
import pickle
from scipy.optimize import curve_fit
import time
from scipy.signal import savgol_filter
from scipy import asarray as ar,exp

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

#finds n number of harmonics
#spacing is the width of the peak
def peaks(arr, n, spacing=10):
    L=np.argsort(-arr)
    bad_idx=[]
    for i in range(len(L)-1):
        if abs(L[i]-L[i+1])<spacing:
            bad_idx.append(i+1)
    L=np.delete(L,bad_idx)
    return np.array(arr[L[:n]])
