from gauss1D import gauss1D

def gauss2D(sigma_x, sigma_y, kernel_size):
    Gx = gauss1D(sigma_x, kernel_size).reshape(-1, 1)
    Gy = gauss1D(sigma_y, kernel_size).reshape(1, -1)
    G = Gx @ Gy
    return G
