# Laplace Transform Implementation

This module provides comprehensive support for computing Laplace transforms, both symbolically and numerically.

## Overview

The Laplace transform is an integral transform that converts a function of a real variable `t` (often time) to a function of a complex variable `s` (complex frequency). It is widely used in signal processing, control theory, and solving differential equations.

For a function `f(t)`, the Laplace transform is defined as:

```
L{f(t)} = F(s) = ∫[0,∞] f(t) * e^(-st) dt
```

## Features

- **Symbolic Laplace Transforms**: Compute exact symbolic transforms using SymPy
- **Numerical Laplace Transforms**: Compute numerical approximations for arbitrary functions
- **Inverse Laplace Transforms**: Transform back from frequency to time domain
- **Common Transform Pairs**: Reference table of frequently used transforms

## Installation

Install the required dependencies:

```bash
pip install numpy scipy sympy
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Symbolic Laplace Transform

```python
import sympy as sp
from bishop.code.laplace_transform import laplace_transform_symbolic

# Define symbolic variables
t, s = sp.symbols('t s', real=True, positive=True)

# Example: Laplace transform of e^(-2t)
f = sp.exp(-2*t)
F = laplace_transform_symbolic(f, t, s)
print(F)  # Output: 1/(s + 2)
```

### Numerical Laplace Transform

```python
import numpy as np
from bishop.code.laplace_transform import laplace_transform_numerical

# Define a function
f = lambda t: np.exp(-2*t)

# Compute Laplace transform at s=3
F_s = laplace_transform_numerical(f, s_val=3, t_max=20)
print(F_s)  # Output: 0.2 (which is 1/5)
```

### Inverse Laplace Transform

```python
import sympy as sp
from bishop.code.laplace_transform import inverse_laplace_transform_symbolic

# Define symbolic variables
t, s = sp.symbols('t s', real=True, positive=True)

# Example: Inverse Laplace transform of 1/(s+2)
F = 1/(s + 2)
f = inverse_laplace_transform_symbolic(F, s, t)
print(f)  # Output: exp(-2*t)*Heaviside(t)
```

### Common Transform Pairs

```python
from bishop.code.laplace_transform import get_laplace_pairs

pairs = get_laplace_pairs()
for name, pair in pairs.items():
    print(f"{name}: {pair['time']} → {pair['laplace']}")
```

## Examples

Run the comprehensive examples:

```bash
cd /home/runner/work/tour_of_ds/tour_of_ds
PYTHONPATH=. python bishop/code/laplace_examples.py
```

## Running Tests

The module includes comprehensive unit tests:

```bash
cd /home/runner/work/tour_of_ds/tour_of_ds
PYTHONPATH=. python -m pytest tests/test_laplace_transform.py -v
```

## API Reference

### Functions

#### `laplace_transform_symbolic(f, t, s)`

Compute the symbolic Laplace transform of a function.

**Parameters:**
- `f` (sympy expression): The function to transform (in terms of t)
- `t` (sympy symbol): The time variable
- `s` (sympy symbol): The complex frequency variable

**Returns:**
- sympy expression: The Laplace transform F(s)

#### `laplace_transform_numerical(f, s_val, t_max=10, **kwargs)`

Compute the numerical Laplace transform of a function.

**Parameters:**
- `f` (callable): Function to transform, accepts time t as input
- `s_val` (float or complex): Value of the complex frequency variable s
- `t_max` (float, optional): Upper limit of integration (default: 10)
- `**kwargs`: Additional arguments passed to scipy.integrate.quad

**Returns:**
- complex: The numerical value of the Laplace transform at s_val

#### `inverse_laplace_transform_symbolic(F, s, t)`

Compute the symbolic inverse Laplace transform.

**Parameters:**
- `F` (sympy expression): The function in s-domain
- `s` (sympy symbol): The complex frequency variable
- `t` (sympy symbol): The time variable

**Returns:**
- sympy expression: The inverse Laplace transform f(t)

#### `get_laplace_pairs()`

Return a dictionary of common Laplace transform pairs.

**Returns:**
- dict: Dictionary with common Laplace transform pairs

## Common Transform Pairs

| Time Domain | Laplace Domain |
|-------------|----------------|
| 1 | 1/s |
| e^(-at) | 1/(s+a) |
| sin(ωt) | ω/(s²+ω²) |
| cos(ωt) | s/(s²+ω²) |
| t | 1/s² |
| t^n | n!/s^(n+1) |

## Dependencies

- NumPy: For numerical computations
- SciPy: For numerical integration
- SymPy: For symbolic mathematics

## References

- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
- Oppenheim, A. V., Willsky, A. S., & Nawab, S. H. (1997). Signals and Systems.
