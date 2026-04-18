import numpy as np

# Ram-Lak kernel, który jest używany do filtracji sinogramu w metodzie splotu.
def _filtr_ramlak_jadro(liczba_detektorow):
    # Nieparzysta dlugosc jadra ulatwia centrowanie przy trybie 'same'.
    dlugosc = min(129, liczba_detektorow if liczba_detektorow % 2 == 1 else liczba_detektorow - 1)
    if dlugosc < 3:
        dlugosc = 3

    srodek = dlugosc // 2
    indeksy = np.arange(-srodek, srodek + 1)
    jadro = np.zeros_like(indeksy, dtype=np.float64)

    jadro[srodek] = 1.0 / 4.0
    niezerowe = indeksy != 0
    nieparzyste = np.abs(indeksy) % 2 == 1
    maska = niezerowe & nieparzyste
    jadro[maska] = -1.0 / (np.pi**2 * indeksy[maska] ** 2)

    return jadro

# Transformata Fouriera
def _filtruj_sinogram_fft(sinogram):
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

# Splot
def _filtruj_sinogram_convolve(sinogram):
    liczba_detektorow, liczba_skanow = sinogram.shape
    przefiltrowany_sinogram = np.zeros_like(sinogram)
    jadro = _filtr_ramlak_jadro(liczba_detektorow)

    for skan in range(liczba_skanow):
        odczyt_kat = sinogram[:, skan]
        przefiltrowany_sinogram[:, skan] = np.convolve(odczyt_kat, jadro, mode='same')

    return przefiltrowany_sinogram

# Główna funkcja filtrująca, która wybiera metodę na podstawie argumentu 'metoda'.
def filtruj_sinogram(sinogram, metoda='convolve'):
    if metoda == 'fft':
        return _filtruj_sinogram_fft(sinogram)
    if metoda == 'convolve':
        return _filtruj_sinogram_convolve(sinogram)

    raise ValueError("Nieznana metoda filtracji. Uzyj 'fft' albo 'convolve'.")