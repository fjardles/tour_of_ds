"""
Laplace Transform Implementation

This module provides functions for computing Laplace transforms,
both symbolic and numerical.

The Laplace transform is an integral transform that converts a function
of a real variable t (often time) to a function of a complex variable s
(complex frequency).

For a function f(t), the Laplace transform is defined as:
L{f(t)} = F(s) = ∫[0,∞] f(t) * e^(-st) dt
"""

import numpy as np
from scipy import integrate
import sympy as sp


def laplace_transform_symbolic(f, t, s):
    """
    Compute the symbolic Laplace transform of a function.
    
    Parameters:
    -----------
    f : sympy expression
        The function to transform (expressed in terms of t)
    t : sympy symbol
        The time variable
    s : sympy symbol
        The complex frequency variable
    
    Returns:
    --------
    sympy expression
        The Laplace transform F(s)
    
    Examples:
    ---------
    >>> import sympy as sp
    >>> t, s = sp.symbols('t s', real=True, positive=True)
    >>> f = sp.exp(-2*t)
    >>> F = laplace_transform_symbolic(f, t, s)
    >>> print(F)
    1/(s + 2)
    """
    return sp.laplace_transform(f, t, s, noconds=True)


def laplace_transform_numerical(f, s_val, t_max=10, **kwargs):
    """
    Compute the numerical Laplace transform of a function.
    
    This function numerically evaluates the integral:
    L{f(t)} = ∫[0,t_max] f(t) * e^(-s*t) dt
    
    Parameters:
    -----------
    f : callable
        Function to transform, should accept time t as input
    s_val : float or complex
        Value of the complex frequency variable s
    t_max : float, optional
        Upper limit of integration (default: 10)
    **kwargs : dict
        Additional arguments passed to scipy.integrate.quad
    
    Returns:
    --------
    complex
        The numerical value of the Laplace transform at s_val
    
    Examples:
    ---------
    >>> f = lambda t: np.exp(-2*t)
    >>> F_s = laplace_transform_numerical(f, s_val=3)
    >>> print(f"F(3) ≈ {F_s:.6f}")
    F(3) ≈ 0.200000
    """
    def integrand(t):
        return f(t) * np.exp(-s_val * t)
    
    if np.iscomplex(s_val):
        # Handle complex s by separating real and imaginary parts
        def real_part(t):
            return np.real(integrand(t))
        
        def imag_part(t):
            return np.imag(integrand(t))
        
        real_result, _ = integrate.quad(real_part, 0, t_max, **kwargs)
        imag_result, _ = integrate.quad(imag_part, 0, t_max, **kwargs)
        return complex(real_result, imag_result)
    else:
        result, _ = integrate.quad(integrand, 0, t_max, **kwargs)
        return result


def inverse_laplace_transform_symbolic(F, s, t):
    """
    Compute the symbolic inverse Laplace transform.
    
    Parameters:
    -----------
    F : sympy expression
        The function in s-domain to transform back to time domain
    s : sympy symbol
        The complex frequency variable
    t : sympy symbol
        The time variable
    
    Returns:
    --------
    sympy expression
        The inverse Laplace transform f(t)
    
    Examples:
    ---------
    >>> import sympy as sp
    >>> t, s = sp.symbols('t s', real=True, positive=True)
    >>> F = 1/(s + 2)
    >>> f = inverse_laplace_transform_symbolic(F, s, t)
    >>> print(f)
    exp(-2*t)*Heaviside(t)
    """
    return sp.inverse_laplace_transform(F, s, t)


# Common Laplace transform pairs as a reference
LAPLACE_PAIRS = {
    "unit_step": {
        "time": "1",
        "laplace": "1/s"
    },
    "exponential": {
        "time": "exp(-a*t)",
        "laplace": "1/(s+a)"
    },
    "sine": {
        "time": "sin(w*t)",
        "laplace": "w/(s^2 + w^2)"
    },
    "cosine": {
        "time": "cos(w*t)",
        "laplace": "s/(s^2 + w^2)"
    },
    "ramp": {
        "time": "t",
        "laplace": "1/s^2"
    },
    "power": {
        "time": "t^n",
        "laplace": "n!/s^(n+1)"
    }
}


def get_laplace_pairs():
    """
    Return a dictionary of common Laplace transform pairs.
    
    Returns:
    --------
    dict
        Dictionary with common Laplace transform pairs
    """
    return LAPLACE_PAIRS
