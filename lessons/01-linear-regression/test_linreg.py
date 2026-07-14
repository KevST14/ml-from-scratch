import numpy as np
import pytest

from linreg import fit, fit_closed_form, gradients, mse, predict


def make_data(n=200, seed=0):
    """Two features, known true parameters, a little noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    true_w = np.array([3.0, -2.0])
    true_b = 0.5
    y = X @ true_w + true_b + rng.normal(scale=0.1, size=n)
    return X, y, true_w, true_b


def reference_loss(X, y, w, b):
    return np.mean((X @ w + b - y) ** 2)


class TestPredict:
    def test_one_feature(self):
        X = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(predict(X, np.array([2.0]), 1.0), [3.0, 5.0, 7.0])

    def test_two_features(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(predict(X, np.array([2.0, -1.0]), 0.5), [2.5, -0.5, 1.5])

    def test_one_prediction_per_row(self):
        X, _, _, _ = make_data(n=50)
        assert predict(X, np.zeros(2), 0.0).shape == (50,)


class TestMSE:
    def test_worked_example_from_the_notebook(self):
        assert mse(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0])) == pytest.approx(5 / 3)

    def test_perfect_predictions_cost_nothing(self):
        y = np.array([1.5, -2.0, 3.0])
        assert mse(y, y) == 0.0

    def test_errors_in_either_direction_count(self):
        assert mse(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == pytest.approx(1.0)


class TestGradients:
    def test_shapes(self):
        X, y, _, _ = make_data()
        dw, db = gradients(X, y, np.zeros(2), 0.0)
        assert np.shape(dw) == (2,)
        assert np.ndim(db) == 0

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_matches_numerical_gradient(self, seed):
        X, y, _, _ = make_data()
        rng = np.random.default_rng(seed)
        w, b = rng.normal(size=2), float(rng.normal())
        dw, db = gradients(X, y, w, b)

        h = 1e-6
        for j in range(2):
            nudge = np.zeros(2)
            nudge[j] = h
            numerical = (reference_loss(X, y, w + nudge, b) - reference_loss(X, y, w - nudge, b)) / (2 * h)
            assert dw[j] == pytest.approx(numerical, rel=1e-5, abs=1e-7)
        numerical_b = (reference_loss(X, y, w, b + h) - reference_loss(X, y, w, b - h)) / (2 * h)
        assert db == pytest.approx(numerical_b, rel=1e-5, abs=1e-7)

    def test_zero_at_the_bottom_of_the_bowl(self):
        X, y, _, _ = make_data()
        with_ones = np.column_stack([X, np.ones(len(X))])
        best = np.linalg.lstsq(with_ones, y, rcond=None)[0]
        dw, db = gradients(X, y, best[:2], best[2])
        np.testing.assert_allclose(dw, 0, atol=1e-8)
        assert db == pytest.approx(0, abs=1e-8)


class TestFit:
    def test_recovers_the_true_parameters(self):
        X, y, true_w, true_b = make_data()
        w, b, _ = fit(X, y, lr=0.1, epochs=500)
        np.testing.assert_allclose(w, true_w, atol=0.05)
        assert b == pytest.approx(true_b, abs=0.05)

    def test_records_the_loss_every_epoch(self):
        X, y, _, _ = make_data()
        _, _, history = fit(X, y, lr=0.1, epochs=30)
        assert len(history) == 30
        assert history[0] == pytest.approx(reference_loss(X, y, np.zeros(2), 0.0))

    def test_loss_goes_down_with_a_sensible_learning_rate(self):
        X, y, _, _ = make_data()
        _, _, history = fit(X, y, lr=0.05, epochs=100)
        assert all(later <= earlier + 1e-12 for earlier, later in zip(history, history[1:]))
        assert history[-1] < history[0] / 100


class TestClosedForm:
    def test_exact_on_noise_free_data(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = 4.0 * X[:, 0] - 1.0
        w, b = fit_closed_form(X, y)
        np.testing.assert_allclose(w, [4.0])
        assert b == pytest.approx(-1.0)

    def test_recovers_the_true_parameters(self):
        X, y, true_w, true_b = make_data()
        w, b = fit_closed_form(X, y)
        np.testing.assert_allclose(w, true_w, atol=0.05)
        assert b == pytest.approx(true_b, abs=0.05)

    def test_gradient_descent_agrees_with_it(self):
        X, y, _, _ = make_data()
        w_gd, b_gd, _ = fit(X, y, lr=0.1, epochs=2000)
        w_exact, b_exact = fit_closed_form(X, y)
        np.testing.assert_allclose(w_gd, w_exact, atol=1e-6)
        assert b_gd == pytest.approx(b_exact, abs=1e-6)
