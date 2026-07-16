"""The maths toolkit, written as plain Python.

Rule for this lesson: loops and arithmetic only, no NumPy. The point is to see
that every formula is a loop and some arithmetic. The notebook compares your
answers with NumPy's afterwards.
"""


def mean(xs: list[float]) -> float:
    """Section 2. The average: add everything up, divide by how many there are."""
    raise NotImplementedError


def variance(xs: list[float]) -> float:
    """Section 2. The average squared distance from the mean (divide by n)."""
    raise NotImplementedError


def std(xs: list[float]) -> float:
    """Section 2. Standard deviation: the square root of the variance."""
    raise NotImplementedError


def numerical_derivative(f, x: float, h: float = 1e-5) -> float:
    """Section 4. The slope of f at x, measured by nudging x by h either way."""
    raise NotImplementedError


def numerical_gradient(f, params: list[float], h: float = 1e-5) -> list[float]:
    """Section 5. One slope per parameter: nudge each in turn, holding the rest still."""
    raise NotImplementedError


def dot(a: list[float], b: list[float]) -> float:
    """Section 7. Multiply matching pairs and add them up."""
    raise NotImplementedError


def matvec(A: list[list[float]], x: list[float]) -> list[float]:
    """Section 8. A matrix times a vector: one dot product per row of A."""
    raise NotImplementedError
