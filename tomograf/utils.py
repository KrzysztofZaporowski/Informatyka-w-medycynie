import numpy as np
import matplotlib.pyplot as plt

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
