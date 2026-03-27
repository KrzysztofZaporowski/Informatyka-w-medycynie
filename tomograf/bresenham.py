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