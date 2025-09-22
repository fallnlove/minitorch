from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """Multiplication function for scalars with automatic differentiation support."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.
        
        Args:
            ctx: Context to save values for backward pass.
            a: First input value.
            b: Second input value.
            
        Returns:
            The product a * b.
        """
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication.
        
        Computes gradients using the product rule:
        ∂/∂a (a * b) = b, ∂/∂b (a * b) = a
        
        Args:
            ctx: Context with saved values from forward pass.
            d_output: Gradient of the output.
            
        Returns:
            Tuple of gradients (d_a, d_b).
        """
        (a, b) = ctx.saved_values
        return operators.mul(d_output, b), operators.mul(d_output, a)


class Inv(ScalarFunction):
    """Inverse (reciprocal) function for scalars with automatic differentiation support."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse function.
        
        Args:
            ctx: Context to save values for backward pass.
            a: Input value.
            
        Returns:
            The reciprocal 1/a.
        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse function.
        
        Computes gradient using the derivative of 1/x which is -1/x^2.
        
        Args:
            ctx: Context with saved values from forward pass.
            d_output: Gradient of the output.
            
        Returns:
            Gradient with respect to input.
        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function for scalars with automatic differentiation support."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.
        
        Args:
            ctx: Context (not used for negation).
            a: Input value.
            
        Returns:
            The negation -a.
        """
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation.
        
        The derivative of -x is -1, so gradient is simply negated.
        
        Args:
            ctx: Context (not used).
            d_output: Gradient of the output.
            
        Returns:
            Negated gradient.
        """
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid activation function for scalars with automatic differentiation support."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid function.
        
        Args:
            ctx: Context to save values for backward pass.
            a: Input value.
            
        Returns:
            Sigmoid of a: 1/(1 + exp(-a)).
        """
        ctx.save_for_backward(a)
        return operators.sigmoid(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid function.
        
        The derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x)).
        
        Args:
            ctx: Context with saved values from forward pass.
            d_output: Gradient of the output.
            
        Returns:
            Gradient with respect to input.
        """
        (a,) = ctx.saved_values
        return operators.sigmoid(a) * (1 - operators.sigmoid(a)) * d_output


class ReLU(ScalarFunction):
    """Rectified Linear Unit (ReLU) activation function for scalars with automatic differentiation support."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU function.
        
        Args:
            ctx: Context to save values for backward pass.
            a: Input value.
            
        Returns:
            ReLU of a: max(0, a).
        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU function.
        
        The derivative of ReLU(x) is 1 if x > 0, else 0.
        
        Args:
            ctx: Context with saved values from forward pass.
            d_output: Gradient of the output.
            
        Returns:
            Gradient with respect to input.
        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function for scalars with automatic differentiation support."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential function.
        
        Args:
            ctx: Context to save values for backward pass.
            a: Input value.
            
        Returns:
            Exponential of a: e^a.
        """
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential function.
        
        The derivative of exp(x) is exp(x).
        
        Args:
            ctx: Context with saved values from forward pass.
            d_output: Gradient of the output.
            
        Returns:
            Gradient with respect to input.
        """
        (a,) = ctx.saved_values
        return operators.exp(a) * d_output


class LT(ScalarFunction):
    """Less-than comparison function for scalars with automatic differentiation support."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less-than comparison.
        
        Computes $f(a, b) = 1.0$ if a < b, else $0.0$.
        
        Args:
            ctx: Context (not used for comparison).
            a: First value to compare.
            b: Second value to compare.
            
        Returns:
            1.0 if a < b, otherwise 0.0.
        """
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less-than comparison.
        
        Comparison operations have zero gradients as they are non-differentiable.
        
        Args:
            ctx: Context (not used).
            d_output: Gradient of the output.
            
        Returns:
            Zero gradients for both inputs.
        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equality comparison function for scalars with automatic differentiation support."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality comparison.
        
        Computes $f(a, b) = 1.0$ if a == b, else $0.0$.
        
        Args:
            ctx: Context (not used for comparison).
            a: First value to compare.
            b: Second value to compare.
            
        Returns:
            1.0 if a == b, otherwise 0.0.
        """
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality comparison.
        
        Comparison operations have zero gradients as they are non-differentiable.
        
        Args:
            ctx: Context (not used).
            d_output: Gradient of the output.
            
        Returns:
            Zero gradients for both inputs.
        """
        return 0.0, 0.0