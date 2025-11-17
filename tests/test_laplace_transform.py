"""
Tests for Laplace Transform Implementation
"""

import unittest
import numpy as np
import sympy as sp
from bishop.code.laplace_transform import (
    laplace_transform_symbolic,
    laplace_transform_numerical,
    inverse_laplace_transform_symbolic,
    get_laplace_pairs,
    LAPLACE_PAIRS
)


class TestLaplaceTransformSymbolic(unittest.TestCase):
    """Test cases for symbolic Laplace transforms"""
    
    def setUp(self):
        """Set up symbolic variables"""
        self.t = sp.symbols('t', real=True, positive=True)
        self.s = sp.symbols('s', real=True, positive=True)
    
    def test_exponential_decay(self):
        """Test Laplace transform of exponential decay e^(-a*t)"""
        a = sp.Symbol('a', real=True, positive=True)
        f = sp.exp(-a * self.t)
        F = laplace_transform_symbolic(f, self.t, self.s)
        expected = 1 / (self.s + a)
        self.assertEqual(sp.simplify(F - expected), 0)
    
    def test_unit_step(self):
        """Test Laplace transform of unit step function"""
        f = sp.Heaviside(self.t)
        F = laplace_transform_symbolic(f, self.t, self.s)
        expected = 1 / self.s
        self.assertEqual(sp.simplify(F - expected), 0)
    
    def test_sine_function(self):
        """Test Laplace transform of sin(w*t)"""
        w = sp.Symbol('w', real=True, positive=True)
        f = sp.sin(w * self.t)
        F = laplace_transform_symbolic(f, self.t, self.s)
        expected = w / (self.s**2 + w**2)
        self.assertEqual(sp.simplify(F - expected), 0)
    
    def test_cosine_function(self):
        """Test Laplace transform of cos(w*t)"""
        w = sp.Symbol('w', real=True, positive=True)
        f = sp.cos(w * self.t)
        F = laplace_transform_symbolic(f, self.t, self.s)
        expected = self.s / (self.s**2 + w**2)
        self.assertEqual(sp.simplify(F - expected), 0)
    
    def test_polynomial(self):
        """Test Laplace transform of t^n"""
        # Test for t (n=1)
        f = self.t
        F = laplace_transform_symbolic(f, self.t, self.s)
        expected = 1 / self.s**2
        self.assertEqual(sp.simplify(F - expected), 0)
        
        # Test for t^2 (n=2)
        f = self.t**2
        F = laplace_transform_symbolic(f, self.t, self.s)
        expected = 2 / self.s**3
        self.assertEqual(sp.simplify(F - expected), 0)


class TestLaplaceTransformNumerical(unittest.TestCase):
    """Test cases for numerical Laplace transforms"""
    
    def test_exponential_decay(self):
        """Test numerical Laplace transform of e^(-2*t)"""
        f = lambda t: np.exp(-2 * t)
        s_val = 3
        F_s = laplace_transform_numerical(f, s_val, t_max=20)
        expected = 1 / (s_val + 2)  # 1/5 = 0.2
        self.assertAlmostEqual(F_s, expected, places=5)
    
    def test_unit_step(self):
        """Test numerical Laplace transform of unit step"""
        f = lambda t: 1.0
        s_val = 2
        F_s = laplace_transform_numerical(f, s_val, t_max=20)
        expected = 1 / s_val  # 1/2 = 0.5
        self.assertAlmostEqual(F_s, expected, places=5)
    
    def test_sine_function(self):
        """Test numerical Laplace transform of sin(w*t)"""
        w = 2.0
        f = lambda t: np.sin(w * t)
        s_val = 3.0
        F_s = laplace_transform_numerical(f, s_val, t_max=50)
        expected = w / (s_val**2 + w**2)  # 2/(9+4) = 2/13
        self.assertAlmostEqual(F_s, expected, places=3)
    
    def test_cosine_function(self):
        """Test numerical Laplace transform of cos(w*t)"""
        w = 2.0
        f = lambda t: np.cos(w * t)
        s_val = 3.0
        F_s = laplace_transform_numerical(f, s_val, t_max=50)
        expected = s_val / (s_val**2 + w**2)  # 3/(9+4) = 3/13
        self.assertAlmostEqual(F_s, expected, places=3)
    
    def test_complex_s(self):
        """Test numerical Laplace transform with complex s"""
        f = lambda t: np.exp(-t)
        s_val = 2 + 1j
        F_s = laplace_transform_numerical(f, s_val, t_max=20)
        # For e^(-t), L{f} = 1/(s+1) = 1/(2+1j+1) = 1/(3+1j)
        expected = 1 / (s_val + 1)
        self.assertAlmostEqual(F_s.real, expected.real, places=4)
        self.assertAlmostEqual(F_s.imag, expected.imag, places=4)


class TestInverseLaplaceTransform(unittest.TestCase):
    """Test cases for inverse Laplace transforms"""
    
    def setUp(self):
        """Set up symbolic variables"""
        self.t = sp.symbols('t', real=True, positive=True)
        self.s = sp.symbols('s', real=True, positive=True)
    
    def test_inverse_simple(self):
        """Test inverse Laplace transform of 1/(s+a)"""
        a = sp.Symbol('a', real=True, positive=True)
        F = 1 / (self.s + a)
        f = inverse_laplace_transform_symbolic(F, self.s, self.t)
        # Result should be e^(-a*t)*Heaviside(t)
        expected = sp.exp(-a * self.t) * sp.Heaviside(self.t)
        self.assertEqual(sp.simplify(f - expected), 0)
    
    def test_inverse_polynomial(self):
        """Test inverse Laplace transform of 1/s^2"""
        F = 1 / self.s**2
        f = inverse_laplace_transform_symbolic(F, self.s, self.t)
        # Result should be t*Heaviside(t)
        expected = self.t * sp.Heaviside(self.t)
        self.assertEqual(sp.simplify(f - expected), 0)


class TestLaplacePairs(unittest.TestCase):
    """Test cases for Laplace transform pairs reference"""
    
    def test_get_laplace_pairs(self):
        """Test that get_laplace_pairs returns the correct dictionary"""
        pairs = get_laplace_pairs()
        self.assertIsInstance(pairs, dict)
        self.assertIn("unit_step", pairs)
        self.assertIn("exponential", pairs)
        self.assertIn("sine", pairs)
        self.assertIn("cosine", pairs)
    
    def test_laplace_pairs_constant(self):
        """Test that LAPLACE_PAIRS constant is accessible"""
        self.assertIsInstance(LAPLACE_PAIRS, dict)
        self.assertEqual(LAPLACE_PAIRS["unit_step"]["laplace"], "1/s")


class TestIntegration(unittest.TestCase):
    """Integration tests combining symbolic and numerical methods"""
    
    def setUp(self):
        """Set up symbolic variables"""
        self.t = sp.symbols('t', real=True, positive=True)
        self.s = sp.symbols('s', real=True, positive=True)
    
    def test_symbolic_vs_numerical(self):
        """Compare symbolic and numerical results for e^(-2*t)"""
        # Symbolic
        f_sym = sp.exp(-2 * self.t)
        F_sym = laplace_transform_symbolic(f_sym, self.t, self.s)
        F_at_3 = F_sym.subs(self.s, 3)
        
        # Numerical
        f_num = lambda t: np.exp(-2 * t)
        F_num = laplace_transform_numerical(f_num, s_val=3, t_max=20)
        
        self.assertAlmostEqual(float(F_at_3), F_num, places=5)


if __name__ == '__main__':
    unittest.main()
