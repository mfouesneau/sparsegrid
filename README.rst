Smolyak Sparse Grid Interpolation
=================================

This library is an implementation of Smolyak’s Sparse Grid Algorithm for solving
integration and interpolation problems in d-dim spaces with far fewer function
evaluations than needed with traditional tensor production
integration / interpolation.

Implementation
--------------
This library is mostly a sandbox for testing sparse grid applications

I deeply vampirized Michael Tompkins implementation https://github.com/geofizx/Sparse-Grid-Interpolation
The major updates are based on code cleaning and interface. Note that I did not
optimize the code for speed (yet).

This library currently implements Smolyak's algorithm for two polynomial bases:

* Clenshaw-Curtis - Piecewise Linear Basis Functions
* Chebyshev Polynomials - Cos Basis Functions

All goes through the same interface class `SparseInterpolator`.


Method
------

Sergey Smolyak introduced a numerical technique where the number of grid points
needed to approximate grew polynomially instead of exponentially. The idea
behind this technique is that some elements produced by tensor-product rules are
more important for representing multidimensional functions than the others.

The tensor-product typically takes as parameters the dimension of the grid and
the number of points, :math:`n` to be evaluated at in each dimension which produces a
grid with :math:`n^d` points. Note that for non-regular grid where the number of points
is different in each dimension, the resulting number of points is eventually
similar.

The Smolyak grid takes an *accuracy* parameter and the number of
dimensions :math:`d` as parameters. The number
of Smolyak grid points is then deterministic:

.. math::

    n(d = 1) = 1 + 2 d, 
    n(d = 2) = 1 + 4d + 4d(d-1),
    ...

Notice that the number of grid points grows linearly with :math:`d = 1`, and
quadratically with :math:`d=2` ...

The standard construction of a Smolyak grid uses nested sets of points.
One typically uses the extrema of the *Chebyshev Polynomials*, which are known as
the *Chebyshev-Gauss-Lobatto points*. 

Smolyak interpolation consists of two objects: a grid and a interpolating
polynomial. Typically, the sparse grid is generated using nested sets of the
extrema of the Chebychev polynomials. Similarly, the polynomial is constructed
using nested sets of uni-dimensional basis polynomials (generally Chebychev
polynomials of the first kind). There have been numerous applications of this
procedure to economic models. 

A few implementation examples are:

* Krueger and Kubler (2004)
* Klemke and Wohlmuth (2005)
* Malin, Krueger, and Kubler (2007)
* Malin, Krueger, and Kubler (2011)
* Gordon (2011)

References
----------

* Barthelmann, V., E. Novak, and K. Ritter, 2000, High dimensional polynomial interpolation on sparse grids, Adv. in Comput. Math., 12, 273–288.
* Gordon (2011)
* Krueger and Kubler (2004)
* Klemke and Wohlmuth (2005)
* Malin, Krueger, and Kubler (2007)
* Malin, Krueger, and Kubler (2011)
* Smolyak, S., 1963, Quadrature and interpolation formulas for tensor products of certain classes of functions, Soviet Math. Dokl., 4, 240-243.
* Waldvogel, J., 2003, Fast construction of the Fejér and Clenshaw-Curtis quadrature rules, BIT Numerical Mathematics, 43(1), 1-18.

