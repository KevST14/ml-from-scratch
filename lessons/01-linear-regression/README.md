# Lesson 01: Linear regression and gradient descent

> How does a machine learn a straight line?

**Status:** Not started

Work through [`lesson.ipynb`](lesson.ipynb). My implementation is in [`linreg.py`](linreg.py), checked by [`test_linreg.py`](test_linreg.py):

```bash
uv run pytest lessons/01-linear-regression
```

**Concepts:** model, parameters, loss function, mean squared error, loss landscape, gradient, gradient descent, learning rate, epoch, gradient checking, normal equation, feature scaling.

**Finance link:** the slope of this regression is a stock's beta.

## Checklist

- [ ] 1–2 Plot the data and fit lines by eye
- [ ] 3 `predict` and `mse`
- [ ] 4 Plot the loss landscape
- [ ] 5 `gradients`, plus a gradient check by hand
- [ ] 6 `fit`, a loss curve, and the descent path on the contour plot
- [ ] 7 Learning-rate experiment
- [ ] 8 `fit_closed_form`, compared with gradient descent
- [ ] 9 Feature scaling
- [ ] Stretch (at least one)

## In my own words

_What does "training a model" mean? What are a loss function, a gradient and a learning rate? When would you use the closed form instead of gradient descent?_

## What broke

_The learning-rate and feature-scaling experiments, plus anything else. What I predicted, what actually happened, and why._

## Key plot

_The loss curves, or the descent path on the contour plot._

## Questions I still have
