import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

def wyznacz_pozycje_czujnikow(stopnie_alfa, n, rozpietosc_stopnie, promien_r, srodek_x, srodek_y):
    alfa = np.radians(stopnie_alfa)
    phi = np.radians(rozpietosc_stopnie)

    # 1. Pozycja Emitera (E)
    # Wzór: x_E = r * cos(alfa), y_E = r * sin(alfa)
    # Dodajemy srodek_x i srodek_y, żeby układ obracał się wokół środka obrazka, a nie punktu (0,0) monitora
    xE = promien_r * np.cos(alfa) + srodek_x
    yE = promien_r * np.sin(alfa) + srodek_y
    emiter = (xE, yE)

    # 2. Pozycja Detektorow
    detektory = []

    if n == 1: 
        kat_kroku = 0
    else:
        kat_kroku = phi / (n - 1)  # Kąt między detektorami

    for i in range(n):
        kat_detektora = alfa + np.pi - phi / 2 + i * kat_kroku  # Detektory rozmieszczone symetrycznie wokół emitera

        xD = promien_r * np.cos(kat_detektora) + srodek_x
        yD = promien_r * np.sin(kat_detektora) + srodek_y
        detektory.append((xD, yD))

    return emiter, detektory

def bresenham(x0, y0, x1, y1):
    # Zwraca listę współrzędnych pikseli leżących na odcinku 
    # między punktami (x0, y0) a (x1, y1).

    points = []
    
    # Różnice między punktami
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    # Kierunek poruszania się (krok w osi X i Y)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    # Zmienna przechowująca błąd (pomaga decydować, czy przesunąć się w X czy w Y)
    err = dx - dy

    while True:
        points.append((x0, y0)) # Dodajemy aktualny piksel do listy
        
        # Jeśli dotarliśmy do punktu końcowego, przerywamy pętlę
        if x0 == x1 and y0 == y1:
            break
            
        # Obliczamy podwójny błąd (optymalizacja, żeby nie używać ułamków)
        e2 = 2 * err
        
        # Korygujemy pozycję X
        if e2 > -dy:
            err -= dy
            x0 += sx
            
        # Korygujemy pozycję Y
        if e2 < dx:
            err += dx
            y0 += sy

    return points

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

def policz_mse(obraz_oryginalny, obraz_rekonstrukcji):
    ori_norm = obraz_oryginalny / np.max(obraz_oryginalny)

    if np.max(obraz_rekonstrukcji) > 0:
        rek_norm = obraz_rekonstrukcji / np.max(obraz_rekonstrukcji)
    else:
        rek_norm = obraz_rekonstrukcji

    mse = np.mean((ori_norm - rek_norm) ** 2)
    return mse

def zapisz_dicom(obraz, nazwa_pliku, dane_pacjenta, data_badania, komentarze):
    obraz = (obraz * 255).astype(np.uint8)

    imie, nazwisko, pesel = dane_pacjenta
    ds = Dataset()
    ds.PatientName = f"{imie} {nazwisko}"
    ds.PatientID = pesel
    ds.StudyDate = data_badania
    ds.StudyDescription = komentarze
    ds.Rows, ds.Columns = obraz.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    
    ds.PixelData = obraz.tobytes()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.save_as(nazwa_pliku, write_like_original=False)

    wczytany_dicom = pydicom.dcmread(nazwa_pliku)

    plt.imshow(wczytany_dicom.pixel_array, cmap='gray')
    plt.title(f"Zapisany DICOM\nPacjent: {wczytany_dicom.PatientName}, ID: {wczytany_dicom.PatientID}")
    plt.show()
    
def rekonstrukcja_obrazu(wymiar_x, wymiar_y, liczba_detektorow, liczba_skanow, rozpietosc, sinogram):
    srodek_x = wymiar_x / 2
    srodek_y = wymiar_y / 2
    promien = max(wymiar_x, wymiar_y) * 0.7
    delta_alfa = 360 / liczba_skanow

    rekonstruowany_obraz = np.zeros((wymiar_x, wymiar_y))

    for skan in range(liczba_skanow):
        aktualny_kat = skan * delta_alfa

        emiter, detektory = wyznacz_pozycje_czujnikow(aktualny_kat, liczba_detektorow, rozpietosc, promien, srodek_x, srodek_y)

        for i, detektor in enumerate(detektory):
            x0, y0 = int(emiter[0]), int(emiter[1])
            x1, y1 = int(detektor[0]), int(detektor[1])

            linia_pikseli = bresenham(x0, y0, x1, y1)

            for piksel in linia_pikseli:
                x, y = piksel
                if 0 <= x < wymiar_x and 0 <= y < wymiar_y:
                    rekonstruowany_obraz[x, y] += sinogram[i, skan]

    # Normalizacja obrazu do zakresu [0, 1]
    max_val = np.max(rekonstruowany_obraz)

    if max_val > 0:
        rekonstruowany_obraz = rekonstruowany_obraz / max_val

    return rekonstruowany_obraz

def stworz_sinogram(wymiar_x, wymiar_y, liczba_detektorow, liczba_skanow, rozpietosc, image):
    srodek_x = wymiar_x / 2
    srodek_y = wymiar_y / 2
    promien = max(wymiar_x, wymiar_y) * 0.7
    delta_alfa = 360 / liczba_skanow

    sinogram = np.zeros((liczba_detektorow, liczba_skanow))

    for skan in range(liczba_skanow):
        aktualny_kat = skan * delta_alfa

        emiter, detektory = wyznacz_pozycje_czujnikow(aktualny_kat, liczba_detektorow, rozpietosc, promien, srodek_x, srodek_y)

        for i, detektor in enumerate(detektory):
            x0, y0 = int(emiter[0]), int(emiter[1])
            x1, y1 = int(detektor[0]), int(detektor[1])

            linia_pikseli = bresenham(x0, y0, x1, y1)

            suma_jasnosci = 0

            for piksel in linia_pikseli:
                x, y = piksel
                if 0 <= x < wymiar_x and 0 <= y < wymiar_y:
                    suma_jasnosci += image[x, y]

            sinogram[i, skan] = suma_jasnosci

    return sinogram