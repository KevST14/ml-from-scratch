# Lesson 01: Linear regression and gradient descent

> How does a machine learn a straight line?

**Status:** Not started

Work through [`lesson.ipynb`](lesson.ipynb). My implementation is in [`linreg.py`](linreg.py), checked by [`test_linreg.py`](test_linreg.py):

```bash
uv run pytest lessons/01-linear-regression
```

**Builds on:** [lesson 00](../00-maths-toolkit/) (averages, slopes, the chain rule, dot products, shapes).

**Concepts:** model, parameters, loss function, mean squared error, loss landscape, gradient, gradient descent, learning rate, epoch, gradient checking, normal equation, feature scaling.

**Finance link:** the slope of this regression is a stock's beta.

## Checklist

- [ ] 1–2 Plot the data and fit lines by eye
- [ ] 3 `predict` and `mse`, then the slider
- [ ] 4 Loss landscape: 1D curve and contour plot
- [ ] 5 `gradients`, checked against the nudge test
- [ ] 6 A second gradient-descent step by hand, then `fit`, a loss curve, and the path on the contour plot
- [ ] 7 Learning-rate experiment
- [ ] 8 `fit_closed_form`, compared with gradient descent and scikit-learn
- [ ] 9 Feature scaling
- [ ] 10 Interview questions answered below
- [ ] Stretch (at least one)

## Explain it to a friend

_Gradient descent in five sentences, with no symbols and no code, for someone who has never programmed._

## In my own words

_What does "training a model" mean? What are a loss function, a gradient and a learning rate? When would you use the closed form instead of gradient descent?_

## What broke

_The learning-rate and feature-scaling experiments, plus anything else. What I predicted, what actually happened, and why._

## Key plot

_The loss curves, or the descent path on the contour plot._

## Interview questions

1. Why do we square the errors instead of just adding them up?
2. What happens if the learning rate is too high? Too low?
3. Why scale features before gradient descent?
4. Linear regression has an exact solution. Why would anyone use gradient descent for it?
5. How would you check that your gradient code is correct?
6. You estimated a beta of 1.3 from last year's data. Why might it be wrong next year?

## Questions I still have
