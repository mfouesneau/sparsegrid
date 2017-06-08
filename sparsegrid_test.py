"""
Some unit tests and usage examples for Smolyak Sparse Grid Interpolation

2D Chebyshev polynomial sparse grid interpolation of 2D test function in fun_nd
2D Clenshaw-Curtis piece-wise linear basis sparse grid interpolation of 2D test function in fun_nd

Adapted from Michael Tompkins, https://github.com/geofizx/Sparse-Grid-Interpolation
"""
from __future__ import print_function

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sparsegrid import SparseInterpolator


def make_data():
    """ Generate fake data """
    dim = 2    # Dimensionality of function to interpolate
    n = 6      # Maximum degree of interpolation to consider - early stopping may use less degree exactness
    shape = Nx, Ny = 21, 21   # values on x and y

    Ntot = Nx * Ny
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X,  Y = np.meshgrid(x, y)
    gridout = np.asarray([X.reshape(Ntot), Y.reshape(Ntot)]).T

    intval = np.asarray([[0.0, 1.0], [0.0, 1.0]]).T

    return dim, n, gridout, intval, shape, X, Y


def fun_nd(x):
    """ Example function """
    func2d = ((
        0.5 / np.pi * x[:, 0] -
        .51 / (.4 * np.pi ** 2) * x[:, 0] ** 2 +
        x[:, 1] - (.6)) ** 2 + (1 - 1 / (.8 * np.pi)) * np.cos(x[:, 0]) + .10)
    return func2d



def plot_result(gridout, output, shape, X, Y, func):
    # Compare results with true function
    ref_vals = np.asarray(func(gridout)).reshape(shape)
    out_vals = output.reshape(shape)

    fig = plt.figure(figsize=(16, 6))

    ax = fig.add_subplot(131, projection='3d',title="True Function")
    ax.plot_surface(X, Y, ref_vals, rstride=1, cstride=1, cmap=plt.cm.magma)

    ax = fig.add_subplot(132, projection='3d',title="Interpolation")
    ax.plot_surface(X, Y, out_vals,  rstride=1, cstride=1, cmap=plt.cm.magma)

    ax = fig.add_subplot(133, projection='3d', title="Interpolation Error")
    ax.plot_surface(X, Y, (out_vals - ref_vals), rstride=1, cstride=1, cmap=plt.cm.magma)
    ax.set_zlim(0.0, ax.get_zlim()[1] * 2)
    plt.tight_layout()


if __name__ == '__main__':

    # which tests will run
    chebyshev = True
    clenshaw = True

    dim, n, gridout, intval, shape, X, Y = make_data()
    func = fun_nd


    if chebyshev is True:

        # Run Chebyshev polynomial sparse-grid interpolation of 2D test function in fun_nd
        interpolation_type = "CH"
        interp = SparseInterpolator(n, dim, interpolation_type, intval)

        output = interp.fit(func, gridout)
        output1 = interp.evaluate(gridout)
        print((output - output1).ptp())

        plot_result(gridout, output, shape, X, Y, func)
        plt.title("Chebyshev Polynomial Basis")

    if clenshaw is True:

        # Run Clenshaw-Curtis Piece-wise linear sparse-grid Interpolation of 2D test function in fun_nd
        interpolation_type = "CC"
        interp = SparseInterpolator(n, dim, interpolation_type, intval)

        output = interp.fit(func, gridout)
        output1 = interp.evaluate(gridout)
        print((output - output1).ptp())

        plot_result(gridout, output, shape, X, Y, func)
        plt.title("Clenshaw Basis")

    plt.show()
