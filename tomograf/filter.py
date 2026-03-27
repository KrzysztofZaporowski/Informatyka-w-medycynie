import numpy as np

def filtruj_sinogram(sinogram):
    liczba_detektorow, liczba_skanow = sinogram.shape
    przefiltrowany_sinogram = np.zeros_like(sinogram)

    czestotliwosci = np.fft.fftfreq(liczba_detektorow)    
    filtr_ramp = np.abs(czestotliwosci) * 2.0 

    for skan in range(liczba_skanow):
        odczyt_kat = sinogram[:, skan]
        odczyt_fft = np.fft.fft(odczyt_kat)
        odczyt_przefiltrowany_fft = odczyt_fft * filtr_ramp

        odczyt_przefiltrowany = np.real(np.fft.ifft(odczyt_przefiltrowany_fft))

        przefiltrowany_sinogram[:, skan] = odczyt_przefiltrowany

    return przefiltrowany_sinogram