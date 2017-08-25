"""Compute the analytical Terzaghi 1D consolidation solution."""

def expterm(Tv, j):
    """Computing the exponential factor of the series."""
    return np.exp(-(2*j-1)**2*np.pi**2/4*Tv)

def costerm(znorm, j):
    """Computing the cosine factor of the series."""
    return np.cos((2*j-1)*np.pi/2*znorm)

def seriesterm(Tv, znorm, j):
    """One term of the series expansion for a given j."""
    return 4/np.pi*(-1)**(j-1)/(2*j-1)*costerm(znorm,j)*expterm(Tv, j)

def terzaghi(Tv, znorm, maxj):
    """Complete solution for a given time factor at a given depth."""
    return sum(seriesterm(Tv, znorm, j) for j in range(1, maxj+1))