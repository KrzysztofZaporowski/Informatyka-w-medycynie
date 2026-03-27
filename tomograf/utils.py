import numpy as np
import bresenham

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

def policz_mse(obraz_oryginalny, obraz_rekonstrukcji):
    ori_norm = obraz_oryginalny / np.max(obraz_oryginalny)

    if np.max(obraz_rekonstrukcji) > 0:
        rek_norm = obraz_rekonstrukcji / np.max(obraz_rekonstrukcji)
    else:
        rek_norm = obraz_rekonstrukcji

    mse = np.mean((ori_norm - rek_norm) ** 2)
    return mse
    
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

            linia_pikseli = bresenham.bresenham(x0, y0, x1, y1)

            for piksel in linia_pikseli:
                x, y = piksel
                if 0 <= x < wymiar_x and 0 <= y < wymiar_y:
                    rekonstruowany_obraz[x, y] += sinogram[i, skan]

    # Normalizacja obrazu do zakresu [0, 1]
    rekonstruowany_obraz[rekonstruowany_obraz < 0] = 0
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

            linia_pikseli = bresenham.bresenham(x0, y0, x1, y1)

            suma_jasnosci = 0

            for piksel in linia_pikseli:
                x, y = piksel
                if 0 <= x < wymiar_x and 0 <= y < wymiar_y:
                    suma_jasnosci += image[x, y]

            sinogram[i, skan] = suma_jasnosci

    return sinogram