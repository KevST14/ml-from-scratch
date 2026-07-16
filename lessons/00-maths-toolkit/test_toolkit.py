import math

import numpy as np
import pytest

from toolkit import dot, matvec, mean, numerical_derivative, numerical_gradient, std, variance

DATA = [2, 4, 4, 4, 5, 5, 7, 9]  # the worked example in section 2


class TestMean:
    def test_worked_example(self):
        assert mean(DATA) == pytest.approx(5.0)

    def test_single_value(self):
        assert mean([3.5]) == pytest.approx(3.5)

    def test_matches_numpy(self):
        xs = list(np.random.default_rng(0).normal(size=100))
        assert mean(xs) == pytest.approx(np.mean(xs))


class TestVariance:
    def test_worked_example(self):
        assert variance(DATA) == pytest.approx(4.0)
        assert std(DATA) == pytest.approx(2.0)

    def test_no_spread_means_zero(self):
        assert variance([7, 7, 7]) == pytest.approx(0.0)

    def test_matches_numpy(self):
        xs = list(np.random.default_rng(1).normal(size=100))
        assert variance(xs) == pytest.approx(np.var(xs))
        assert std(xs) == pytest.approx(np.std(xs))

    def test_shifting_the_data_does_not_change_the_spread(self):
        assert std([x + 10 for x in DATA]) == pytest.approx(std(DATA))


class TestDerivative:
    @pytest.mark.parametrize("x", [-2.0, 0.0, 1.0, 3.0])
    def test_x_squared_has_slope_2x(self, x):
        assert numerical_derivative(lambda t: t**2, x) == pytest.approx(2 * x, abs=1e-6)

    def test_straight_line_has_the_same_slope_everywhere(self):
        for x in [-10.0, 0.0, 4.2]:
            assert numerical_derivative(lambda t: 5 * t + 2, x) == pytest.approx(5.0)

    def test_chain_rule_example_from_section_6(self):
        assert numerical_derivative(lambda t: (3 * t - 1) ** 2, 1.0) == pytest.approx(12.0)

    def test_exponential_is_its_own_slope(self):
        assert numerical_derivative(math.exp, 1.0) == pytest.approx(math.e)


class TestGradient:
    def test_one_slope_per_parameter(self):
        grad = numerical_gradient(lambda p: p[0] ** 2 + 3 * p[1], [1.0, 2.0])
        assert grad == pytest.approx([2.0, 3.0])

    def test_check_yourself_example_from_section_5(self):
        grad = numerical_gradient(lambda p: 5 * p[0] + p[1] ** 2, [2.0, 3.0])
        assert grad == pytest.approx([5.0, 6.0])

    def test_leaves_the_parameters_as_it_found_them(self):
        params = [1.0, 2.0, 3.0]
        numerical_gradient(lambda p: p[0] * p[1] * p[2], params)
        assert params == [1.0, 2.0, 3.0]


class TestDot:
    def test_portfolio_value(self):
        holdings, prices = [10, 5, 2], [3, 10, 50]
        assert dot(holdings, prices) == pytest.approx(180)

    def test_check_yourself_example(self):
        assert dot([1, 2, 3], [4, 5, 6]) == pytest.approx(32)

    def test_different_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            dot([1, 2, 3], [1, 2])


class TestMatvec:
    def test_worked_example(self):
        assert matvec([[1, 2], [3, 4], [5, 6]], [10, 1]) == pytest.approx([12, 34, 56])

    def test_matches_numpy(self):
        rng = np.random.default_rng(2)
        A, x = rng.normal(size=(5, 3)), rng.normal(size=3)
        assert matvec(A.tolist(), x.tolist()) == pytest.approx(A @ x)

    def test_shapes_that_do_not_fit_are_rejected(self):
        with pytest.raises(ValueError):
            matvec([[1, 2], [3, 4]], [1, 2, 3])
