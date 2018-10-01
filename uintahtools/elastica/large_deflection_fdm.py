"""Module for computing the large deflection of an elastic beam by the finite difference method."""

import numpy as np


def small_deflection(load, number_of_cells, beam):
    xs = [x / number_of_cells * beam.l for x in range(number_of_cells)]
    ys = [-load / (24.0 * beam.E * beam.I) * deflection(x, beam.l) for x in xs]

    return xs, ys


def deflection(x, l):
    return x * x * (6.0 * l * l - 4.0 * l * x + x * x)


def large_deflection(load, number_of_cells, beam):
    n_cells = number_of_cells + 1
    gridsize = beam.l / n_cells

    theta = np.zeros(n_cells)
    theta[0] = 0
    theta[1] = theta[0]

    beta = compute_constant_factor(load, gridsize, beam)

    for n in range(1, n_cells - 1):
        theta[n + 1] = angle(theta, n, beta, gridsize)

    return integrate_theta(theta, gridsize)


def integrate_theta(theta, gridsize):
    xs = np.zeros(len(theta))
    ys = np.zeros(len(theta))

    cells = len(theta)

    print(simpsons_rule(0, 1, cells, np.sin, theta))

    return xs, ys


def simpsons_rule(a, b, cells, func, theta):
    l = (b - a) / cells

    result = 0
    for n in range(cells):
        if n == 0 or n == cells - 1:
            result += func(theta[n])
        elif n % 2:
            result += 4.0 * func(theta[n])
        else:
            result += 2.0 * func(theta[n])
    result *= l / 3.0
    return result


def angle(theta, n, beta, gridsize):
    return 2.0 * theta[n] - beta * n * \
        gridsize * np.cos(theta[n]) - theta[n - 1]


def compute_constant_factor(load, gridsize, beam):
    return load * gridsize * gridsize / (beam.E * beam.I)
