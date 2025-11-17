"""
Examples demonstrating the use of Laplace Transform functions

This script shows how to use both symbolic and numerical Laplace transforms.
"""

import numpy as np
import sympy as sp
from bishop.code.laplace_transform import (
    laplace_transform_symbolic,
    laplace_transform_numerical,
    inverse_laplace_transform_symbolic,
    get_laplace_pairs
)


def example_symbolic_transforms():
    """Demonstrate symbolic Laplace transforms"""
    print("=" * 60)
    print("SYMBOLIC LAPLACE TRANSFORMS")
    print("=" * 60)
    
    # Define symbolic variables
    t, s = sp.symbols('t s', real=True, positive=True)
    a, w = sp.symbols('a w', real=True, positive=True)
    
    # Example 1: Exponential decay
    print("\n1. Exponential Decay: f(t) = e^(-2t)")
    f1 = sp.exp(-2*t)
    F1 = laplace_transform_symbolic(f1, t, s)
    print(f"   L{{e^(-2t)}} = {F1}")
    
    # Example 2: Sine function
    print("\n2. Sine Function: f(t) = sin(3t)")
    f2 = sp.sin(3*t)
    F2 = laplace_transform_symbolic(f2, t, s)
    print(f"   L{{sin(3t)}} = {F2}")
    
    # Example 3: Cosine function
    print("\n3. Cosine Function: f(t) = cos(3t)")
    f3 = sp.cos(3*t)
    F3 = laplace_transform_symbolic(f3, t, s)
    print(f"   L{{cos(3t)}} = {F3}")
    
    # Example 4: Polynomial
    print("\n4. Polynomial: f(t) = t^2")
    f4 = t**2
    F4 = laplace_transform_symbolic(f4, t, s)
    print(f"   L{{t^2}} = {F4}")
    
    # Example 5: Product of functions
    print("\n5. Product: f(t) = t * e^(-t)")
    f5 = t * sp.exp(-t)
    F5 = laplace_transform_symbolic(f5, t, s)
    print(f"   L{{t * e^(-t)}} = {F5}")
    
    print()


def example_numerical_transforms():
    """Demonstrate numerical Laplace transforms"""
    print("=" * 60)
    print("NUMERICAL LAPLACE TRANSFORMS")
    print("=" * 60)
    
    # Example 1: Exponential decay
    print("\n1. Exponential Decay: f(t) = e^(-2t) at s=3")
    f1 = lambda t: np.exp(-2*t)
    F1_num = laplace_transform_numerical(f1, s_val=3, t_max=20)
    F1_exact = 1/(3+2)  # = 0.2
    print(f"   Numerical: F(3) = {F1_num:.6f}")
    print(f"   Exact:     F(3) = {F1_exact:.6f}")
    print(f"   Error:     {abs(F1_num - F1_exact):.2e}")
    
    # Example 2: Sine function
    print("\n2. Sine Function: f(t) = sin(2t) at s=3")
    f2 = lambda t: np.sin(2*t)
    F2_num = laplace_transform_numerical(f2, s_val=3, t_max=50)
    F2_exact = 2/(3**2 + 2**2)  # = 2/13
    print(f"   Numerical: F(3) = {F2_num:.6f}")
    print(f"   Exact:     F(3) = {F2_exact:.6f}")
    print(f"   Error:     {abs(F2_num - F2_exact):.2e}")
    
    # Example 3: Complex frequency
    print("\n3. Unit Step: f(t) = 1 at s=2+1j")
    f3 = lambda t: 1.0
    s_complex = 2 + 1j
    F3_num = laplace_transform_numerical(f3, s_val=s_complex, t_max=20)
    F3_exact = 1/s_complex
    print(f"   Numerical: F(2+1j) = {F3_num:.6f}")
    print(f"   Exact:     F(2+1j) = {F3_exact:.6f}")
    
    print()


def example_inverse_transforms():
    """Demonstrate inverse Laplace transforms"""
    print("=" * 60)
    print("INVERSE LAPLACE TRANSFORMS")
    print("=" * 60)
    
    # Define symbolic variables
    t, s = sp.symbols('t s', real=True, positive=True)
    
    # Example 1: Simple pole
    print("\n1. F(s) = 1/(s+2)")
    F1 = 1/(s+2)
    f1 = inverse_laplace_transform_symbolic(F1, s, t)
    print(f"   L^(-1){{1/(s+2)}} = {f1}")
    
    # Example 2: Quadratic denominator
    print("\n2. F(s) = 3/(s^2 + 9)")
    F2 = 3/(s**2 + 9)
    f2 = inverse_laplace_transform_symbolic(F2, s, t)
    print(f"   L^(-1){{3/(s^2 + 9)}} = {f2}")
    
    # Example 3: Higher order pole
    print("\n3. F(s) = 2/s^3")
    F3 = 2/s**3
    f3 = inverse_laplace_transform_symbolic(F3, s, t)
    print(f"   L^(-1){{2/s^3}} = {f3}")
    
    print()


def example_laplace_pairs():
    """Display common Laplace transform pairs"""
    print("=" * 60)
    print("COMMON LAPLACE TRANSFORM PAIRS")
    print("=" * 60)
    
    pairs = get_laplace_pairs()
    print("\nFunction Name    | Time Domain       | Laplace Domain")
    print("-" * 60)
    for name, pair in pairs.items():
        print(f"{name:15} | {pair['time']:17} | {pair['laplace']}")
    
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("LAPLACE TRANSFORM EXAMPLES")
    print("=" * 60 + "\n")
    
    example_symbolic_transforms()
    example_numerical_transforms()
    example_inverse_transforms()
    example_laplace_pairs()
    
    print("=" * 60)
    print("Examples completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
