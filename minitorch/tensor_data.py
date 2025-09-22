from __future__ import annotations

import random
from typing import Iterable, Optional, Sequence, Tuple, Union

import numba
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias

from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """Converts a multidimensional tensor index into a single-dimensional position in storage.

    Transforms a multidimensional index tuple into a linear position in the
    flattened storage array using the tensor's stride information.

    Args:
        index: Index tuple of ints representing position in each dimension.
        strides: Tensor strides indicating memory layout.

    Returns:
        Position in storage as an integer.
    """

    pos = 0
    for idx, stride in zip(index, strides):
        pos += idx * stride

    return int(pos)


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """Converts an ordinal position to a multidimensional index.

    Transforms a linear position in flattened storage back to a multidimensional
    index tuple based on the tensor shape. Ensures that enumerating positions
    0 ... size produces every index exactly once.

    Args:
        ordinal: Ordinal position to convert (linear index).
        shape: Tensor shape defining dimensions.
        out_index: Output array to store the resulting index.

    Returns:
        None. Modifies out_index in place.
    """

    for dim in range(len(shape) - 1, -1, -1):
        out_index[dim] = ordinal % shape[dim]
        ordinal //= shape[dim]
    
    assert ordinal == 0, "Ordinal not zero after processing all dimensions."


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Converts an index from a larger tensor to a smaller tensor following broadcasting rules.

    Maps an index from a larger (broadcasted) tensor shape to the corresponding
    index in a smaller tensor shape. Handles dimension alignment and size-1
    dimension mapping according to NumPy broadcasting semantics.

    Args:
        big_index: Multidimensional index of bigger tensor.
        big_shape: Tensor shape of bigger tensor.
        shape: Tensor shape of smaller tensor.
        out_index: Output array for multidimensional index of smaller tensor.

    Returns:
        None. Modifies out_index in place.
    """
    
    dim_offset = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + dim_offset]


def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcasts two shapes to create a new union shape.

    Implements NumPy-style broadcasting rules to determine the resulting shape
    when two tensors with different shapes are used in element-wise operations.
    Dimensions are aligned from the right, and size-1 dimensions are stretched.

    Args:
        shape1: First tensor shape.
        shape2: Second tensor shape.

    Returns:
        Broadcasted shape that both input shapes can be broadcast to.

    Raises:
        IndexingError: If shapes cannot be broadcast together.
    """

    larger, smaller = (shape1, shape2) if len(shape1) >= len(shape2) else (shape2, shape1)
    dim_offset = len(larger) - len(smaller)

    new_shape = []

    for i in range(len(larger)):
        if i < dim_offset:
            new_shape.append(larger[i])
            continue

        if larger[i] == smaller[i - dim_offset] or min(larger[i], smaller[i - dim_offset]) == 1:
            new_shape.append(max(larger[i], smaller[i - dim_offset]))
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")

    return tuple(new_shape)


def strides_from_shape(shape: UserShape) -> UserStrides:
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """Permutes the dimensions of the tensor.

        Reorders the dimensions of the tensor according to the specified order.
        This creates a new view of the same data with dimensions rearranged.

        Args:
            *order: A permutation of the dimensions (must include each dimension exactly once).

        Returns:
            New `TensorData` with the same storage and reordered dimensions.

        Raises:
            AssertionError: If order doesn't contain each dimension exactly once.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        new_shape = tuple(self.shape[i] for i in order)
        new_strides =  tuple(self.strides[i] for i in order)

        return TensorData(self._storage, new_shape, new_strides)

    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
