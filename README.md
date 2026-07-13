# ml-from-scratch

Learning machine learning from first principles. Each lesson starts from zero (why the idea exists, then the maths, then the code), and every algorithm gets built in NumPy before I touch the library version. The experiments that went wrong stay in.

## How each lesson works

- **`lesson.ipynb`**: the explanation, worked examples and my experiments. Before running anything interesting, I write down what I expect to happen.
- **A small module** (e.g. `linreg.py`): my from-scratch implementation, checked by tests.
- **The lesson's README**: what I learned in my own words, what broke, and questions I still have.
- New terms go in [GLOSSARY.md](GLOSSARY.md); each study session gets a line in [LOG.md](LOG.md).

## Curriculum

### Part 1: Classic machine learning

| # | Lesson | Question | Status |
|---|---|---|---|
| 01 | [Linear regression and gradient descent](lessons/01-linear-regression/) | How does a machine learn a straight line? | Next up |
| 02 | Logistic regression | How does a model predict yes or no? | Planned |
| 03 | Overfitting and regularisation | Why does a perfect fit on training data fail on new data? | Planned |
| 04 | Evaluating classifiers | Why can a 99% accurate model be useless? | Planned |
| 05 | Decision trees and random forests | How does a model learn if-then rules? | Planned |
| 06 | Clustering and PCA | How do you find structure in data with no labels? | Planned |

### Part 2: Neural networks

| # | Lesson | Question | Status |
|---|---|---|---|
| 07 | From a neuron to a network | Why can't a straight line solve XOR, and what can? | Planned |
| 08 | Backpropagation | How does a network work out which weights to blame? | Planned |
| 09 | Training deep networks | Why do deep networks get stuck, and what helps? | Planned |

### Part 3: Language models

| # | Lesson | Question | Status |
|---|---|---|---|
| 10 | Tokenisation | How does text become numbers? | Planned |
| 11 | Embeddings | How can a word be a point in space? | Planned |
| 12 | Attention and a tiny transformer | How does a model decide which earlier words matter? | Planned |
| 13 | Retrieval and evaluation | How do you make an LLM answer from your own documents, and check that it did? | Planned |

Where there's a natural finance example, the lesson uses it: a stock's beta, credit default, yield curves, fraud.

## Running it

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                       # install everything
uv run jupyter lab                            # open the lessons
uv run pytest lessons/01-linear-regression    # check one lesson's implementation
```

Applied finance experiments live in [fintech-lab](https://github.com/KevST14/fintech-lab).
