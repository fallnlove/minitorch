"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Multiplies two numbers.
    
    Computes the mathematical operation $f(x, y) = x * y$.
    
    Args:
        x: First number to multiply.
        y: Second number to multiply.
        
    Returns:
        The product of x and y.
    """
    return x * y


def id(x: float) -> float:
    """Identity function.
    
    Returns the input value unchanged. Computes $f(x) = x$.
    
    Args:
        x: Input value.
        
    Returns:
        The same value as input.
    """
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.
    
    Computes the mathematical operation $f(x, y) = x + y$.
    
    Args:
        x: First number to add.
        y: Second number to add.
        
    Returns:
        The sum of x and y.
    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.
    
    Computes the mathematical operation $f(x) = -x$.
    
    Args:
        x: Number to negate.
        
    Returns:
        The negative of x.
    """
    return -float(x)


def lt(x: float, y: float) -> float:
    """Less than comparison.
    
    Computes $f(x, y) = 1.0$ if x is less than y, else $0.0$.
    
    Args:
        x: First number to compare.
        y: Second number to compare.
        
    Returns:
        1.0 if x < y, otherwise 0.0.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Equality comparison.
    
    Computes $f(x, y) = 1.0$ if x is equal to y, else $0.0$.
    
    Args:
        x: First number to compare.
        y: Second number to compare.
        
    Returns:
        1.0 if x == y, otherwise 0.0.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Maximum of two numbers.
    
    Computes $f(x, y) = x$ if x is greater than y, else $y$.
    
    Args:
        x: First number to compare.
        y: Second number to compare.
        
    Returns:
        The larger of x and y.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if two numbers are approximately equal.
    
    Computes $f(x, y) = |x - y| < 1e-2$.
    
    Args:
        x: First number to compare.
        y: Second number to compare.
        
    Returns:
        1.0 if the absolute difference is less than 1e-2, otherwise 0.0.
    """
    return abs(add(x, neg(y))) < 1e-2


def sigmoid(x: float) -> float:
    """Sigmoid activation function.
    
    Computes the sigmoid function $f(x) = \frac{1.0}{(1.0 + e^{-x})}$.
    
    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Uses a numerically stable implementation:
    - For x >= 0: $f(x) = \frac{1.0}{(1.0 + e^{-x})}$
    - For x < 0: $f(x) = \frac{e^x}{(1.0 + e^{x})}$

    Args:
        x: Input value.
        
    Returns:
        Sigmoid of x, a value between 0 and 1.
    """
    return inv(add(1.0, exp(neg(x)))) if x >= 0.0 else mul(exp(x), inv(add(1.0, exp(x))))


def relu(x: float) -> float:
    """Rectified Linear Unit (ReLU) activation function.
    
    Computes $f(x) = x$ if x is greater than 0, else $0$.

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    
    Args:
        x: Input value.
        
    Returns:
        x if x > 0, otherwise 0.0.
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Backward pass for the logarithm function.
    
    If $f = log$ as above, computes $d \times f'(x)$.
    The derivative of log(x) is 1/x.
    
    Args:
        x: Input value to log function.
        d: Gradient from the output.
        
    Returns:
        The gradient with respect to x: d * (1/x).
    """
    return mul(d, inv(x))


def inv(x: float) -> float:
    """Inverse function.
    
    Computes $f(x) = 1/x$.
    
    Args:
        x: Input value (should not be zero).
        
    Returns:
        The reciprocal of x.
    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Backward pass for the inverse function.
    
    If $f(x) = 1/x$, computes $d \times f'(x)$.
    The derivative of 1/x is -1/x^2.
    
    Args:
        x: Input value to inverse function.
        d: Gradient from the output.
        
    Returns:
        The gradient with respect to x: d * (-1/x^2).
    """
    return mul(d, neg(inv(x)**2))


def relu_back(x: float, d: float) -> float:
    """Backward pass for the ReLU function.
    
    If $f = relu$, computes $d \times f'(x)$.
    The derivative of ReLU is 1 if x > 0, else 0.
    
    Args:
        x: Input value to ReLU function.
        d: Gradient from the output.
        
    Returns:
        The gradient with respect to x: d if x > 0, otherwise 0.0.
    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map function.

    Creates a function that applies a given function to each element of a list.
    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list with the transformed elements.
    """
    def apply(list: Iterable[float]) -> Iterable[float]:
        """Apply the function to each element in the list.
        
        Args:
            list: Iterable of float values.
            
        Returns:
            New list with fn applied to each element.
        """
        return [fn(x) for x in list]

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negates each element in a list.
    
    Uses `map` and `neg` to negate each element in `ls`.
    
    Args:
        ls: List of numbers to negate.
        
    Returns:
        New list with each element negated.
    """
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith (or map2) function.

    Creates a function that applies a binary function element-wise to two lists.
    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function that combines two values.

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produces a new list by
         applying fn(x, y) on each pair of elements.

    """
    def apply(list1: Iterable[float], list2: Iterable[float]) -> Iterable[float]:
        """Apply the binary function to corresponding elements of two lists.
        
        Args:
            list1: First list of values.
            list2: Second list of values (must be same length as list1).
            
        Returns:
            New list with fn applied to each pair of corresponding elements.
            
        Raises:
            AssertionError: If the lists have different lengths.
        """
        assert len(list1) == len(list2)
        return [fn(x, y) for x, y in zip(list1, list2)]

    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements of two lists.
    
    Uses `zipWith` and `add` to add the elements of `ls1` and `ls2`.
    
    Args:
        ls1: First list of numbers.
        ls2: Second list of numbers (must be same length as ls1).
        
    Returns:
        New list where each element is the sum of corresponding elements from ls1 and ls2.
    """
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce function.

    Creates a function that reduces a list to a single value by repeatedly applying
    a binary function, starting from an initial value.

    Args:
        fn: Binary function that combines two values.
        start: Start value $x_0$.

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`.
    """
    def apply(list: Iterable[float]) -> float:
        """Apply the reduction operation to a list.
        
        Args:
            list: List of values to reduce.
            
        Returns:
            Single value obtained by reducing the list with the binary function.
        """
        r = start
        for x in list:
            r = fn(r, x)
        return r
    return apply


def sum(ls: Iterable[float]) -> float:
    """Sums all elements in a list.
    
    Uses `reduce` and `add` to sum up a list.
    
    Args:
        ls: List of numbers to sum.
        
    Returns:
        The sum of all elements in the list.
    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Computes the product of all elements in a list.
    
    Uses `reduce` and `mul` to compute the product of a list.
    
    Args:
        ls: List of numbers to multiply.
        
    Returns:
        The product of all elements in the list.
    """
    return reduce(mul, 1.0)(ls)
