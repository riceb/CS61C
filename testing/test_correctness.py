"""
Feel free to add more test classes and/or tests that are not provided by the skeleton code!
Make sure you follow these naming conventions: https://docs.pytest.org/en/reorganize-docs/goodpractices.html#test-discovery
for your new tests/classes/python files or else they might be skipped.
"""
from utils import *
"""
For each operation, you should write tests to test correctness on matrices of different sizes.
Hint: use rand_dp_nc_matrix to generate dumbpy and numc matrices with the same data and use
      cmp_dp_nc_matrix to compare the results
"""
class TestAddCorrectness:
    def test_small_add(self):
        # TODO: YOUR CODE HERE
        a, b = rand_dp_nc_matrix(5, 5)
        c, d = rand_dp_nc_matrix(5, 5)
        e = a+c
        f = b+d
        assert(cmp_dp_nc_matrix(e, f))
        

    def test_medium_add(self):
        # TODO: YOUR CODE HERE
        a, b = rand_dp_nc_matrix(1000, 900)
        c, d = rand_dp_nc_matrix(1000, 900)
        e = a+c
        f = b+d
        assert(cmp_dp_nc_matrix(e, f))
    

    def test_medium_add(self):
        # TODO: YOUR CODE HERE
        # a, b = rand_dp_nc_matrix(10000, 10001)
        # c, d = rand_dp_nc_matrix(10000, 10001)
        # e = a+c
        # f = b+d
        # assert(cmp_dp_nc_matrix(e, f))
        pass
        
        

class TestSubCorrectness:
    def test_small_sub(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_sub(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_sub(self):
        # TODO: YOUR CODE HERE
        pass

class TestAbsCorrectness:
    def test_small_abs(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_abs(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_abs(self):
        # TODO: YOUR CODE HERE
        pass

class TestNegCorrectness:
    def test_small_neg(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_neg(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_neg(self):
        # TODO: YOUR CODE HERE
        pass

class TestMulCorrectness:
    def test_small_mul(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_mul(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_mul(self):
        # TODO: YOUR CODE HERE
        pass

class TestPowCorrectness:
    def test_small_pow(self):
        # TODO: YOUR CODE HERE
        pass

    def test_medium_pow(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_pow(self):
        # TODO: YOUR CODE HERE
        pass

class TestGetCorrectness:
    def test_get(self):
        # TODO: YOUR CODE HERE
        pass

class TestSetCorrectness:
    def test_set(self):
        # TODO: YOUR CODE HERE
        m1, m2 = rand_dp_nc_matrix(10, 10)
        print(m1)
        m1[5][3] = 1.5
        assert(m1[5][3] != m2[5][3])
        print(m1)

a, b = rand_dp_nc_matrix(24, 29, 1)
print(b[2])
print(b.set(2, 3, 1.3))
print(b[2])

