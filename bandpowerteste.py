from runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f > fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"
unit = "A1"
test = "2019_07_01"
N_Samples = 25600
f_s = 25600
index = 0

with RunIn_File(r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5") as file:
    test = file["A1"][0]
    print(bandpower(test,25600,1000,1100))


# algum erro aqui, não consigo fazer a bandpower nem para uma função genérica