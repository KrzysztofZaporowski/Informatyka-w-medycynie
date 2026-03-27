def plot_line_high(x0, y0, x1, y1):
    points = []

    # Różnice między punktami
    dx = x1 - x0
    dy = y1 - y0

    xi = 1

    if dx < 0:
        xi = -1
        dx = -dx
    D = (2 * dx) - dy
    x = x0
    
    for y in range(y0, y1 + 1):
        points.append((x, y))
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2*dx
    return points