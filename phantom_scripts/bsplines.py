import numpy as np

# A self-contained reference implementation of the B-spline basis functions
# and their first derivatives, in addition to some utiliy functions.

def float_is_zero(v, eps=1e-6):
    """ Test if floating point number is zero. """
    return abs(v) < eps

def special_div(num, den):
    """ Return num/dev with the special rule
    that 0/0 is 0. """
    if float_is_zero(num) and float_is_zero(den):
        return 0.0
    else:
        return num / den

def B(j, p, x, knots):
    """ Compute B-splines using recursive definition. """        
    if p == 0:
        if knots[j] <= x < knots[j+1]:
            return 1.0
        else:
            return 0.0
    else:
        left = special_div((x-knots[j])*B(j,p-1,x,knots), knots[j+p]-knots[j])
        right = special_div((knots[j+1+p]-x)*B(j+1,p-1,x,knots), knots[j+1+p]-knots[j+1])
        return left + right

def render_spline(p, knots, control_points, ts):
    """
    Compute points on a spline function using the straightforward
    implementation of the recurrence relation for the B-splines.
    """
    ys = []
    for t in ts:
        y = 0.0 
        for j in range(0, len(control_points)):
            y += B(j, p, t, knots)*control_points[j]
        ys.append(y)
    return ys


def uniform_regular_knot_vector(n, p, t0=0.0, t1=1.0):
    """
    Create a p+1-regular uniform knot vector for
    a given number of control points
    Throws if n is too small
    """
    # The minimum length of a p+1-regular knot vector
    # is 2*(p+1)
    if n < p+1:
        raise RuntimeError("Too small n for a uniform regular knot vector")

    # p+1 copies of t0 left and p+1 copies of t1 right
    # but one of each in linspace
    return [t0]*p + list(np.linspace(t0, t1, n+1-p)) + [t1]*p

def control_points(p, knots):
    """
    Return the control point abscissa for the control polygon
    of a one-dimensional spline.
    """
    knots = np.array(knots)
    abscissas = []
    for i in range(len(knots)-p-1):
        part = knots[(i+1):(i+1+p)]
        abscissas.append(np.mean(part))
    return abscissas


def B_derivative(j, p, x, knots):
    """
    Evaluate the derivative of Bj,p(x)
    p must be greater than or equal to 1
    """
    if p < 1: raise RuntimeError("p must be greater than or equal to 1")
    left = special_div(B(j, p-1, x, knots), (knots[j+p]-knots[j]) )
    right = special_div(B(j+1,p-1, x, knots), (knots[j+p+1] - knots[j+1]) )
    return (left - right)*p
