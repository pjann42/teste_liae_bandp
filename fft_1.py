from runInDB_utils import RunIn_File
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy

path = r"\\LIAE-SANTINHO\Backups\Amaciamento_DatabaseMIMICRI\ModelA.hdf5"
unit = "A1"
test = "2019_07_01"
N_Samples = 25600
f_s = 25600
index = range(1)

def FFT_ensaio(file:RunIn_File, unidade:str, teste:str, index:int):

    # Retorna FFT da vibração lateral no index desejado

    hm = file[unidade][teste]# path ->  unidade -> ensaio -> index

    dados = hm.getMeasurements(varName=["vibrationRAWLateral"], indexes = [index])[0]["vibrationRAWLateral"]
    

    N = len(dados)
    fs = N
    T = 1/fs
    t = np.arange(0,N/fs,T)

    # fft

    f = np.fft.fftfreq(N,T)
    transf = np.fft.fft(dados) #transformada
    tr = np.abs(transf)

    #plot

    plt.xlabel('Index')
    plt.ylabel('Frequência')
    plt.title('Espectro de Frequência')
    plt.grid(True)
    plt.plot(f,tr)
    plt.plot(f[f>=0],tr[f>=0])
    plt.show()
    return (f[f>=0],tr[f>=0])


with RunIn_File(path) as file:

    fft = FFT_ensaio(file,unit,test,0) #ta dando problema no último parâmetro quando usa o arquivo

