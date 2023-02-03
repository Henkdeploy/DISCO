def inc(x):
    return x + 1


def test_answer():
    assert inc(3) == 5
    
    
import unittest

def square(n):
    return n*n

def cube(n):
    return n*n*n

class TestCase(unittest.TestCase):
    def test_square(self):
        self.assertEqual(square(4), 16)

    def test_cube(self):
        self.assertEqual(cube(4), 16)    