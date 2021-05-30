from __future__ import annotations
from enum import Enum, unique
from typing import NamedTuple, Optional, Sequence, Union


class uCType:
    """
    Class that represents a type in the uC language.  Basic
    Types are declared as singleton instances of this type.
    """

    __slots__ = "_typename", "binary_ops", "unary_ops", "rel_ops", "assign_ops"

    def __init__(
        self,
        name: Optional[str],
        unary_ops: set[str] = set(),
        binary_ops: set[str] = set(),
        rel_ops: set[str] = set(),
        assign_ops: set[str] = set(),
    ):
        """
        You must implement yourself and figure out what to store.
        """
        self._typename = name
        self.unary_ops = frozenset(unary_ops)
        self.binary_ops = frozenset(binary_ops)
        self.rel_ops = frozenset(rel_ops)
        self.assign_ops = frozenset(assign_ops)

    def __eq__(self, other: uCType) -> bool:
        # primary types are only equal to themselves
        return self is other

    def typename(self) -> str:
        """The name of the uCType."""
        return self._typename or "<unnamed>"

    def __str__(self) -> str:
        """Standard type formatting."""
        return f"type({self!r})"

    def __repr__(self) -> str:
        """Only show typename."""
        return self.typename()


# # # # # # # # #
# Primary Types #


@unique
class PrimaryType(uCType, Enum):
    def __init__(self, *op_sets: set[str]):
        # use enum name
        super().__init__(self.name, *op_sets)

    @classmethod
    def get(cls, typename: str) -> Optional[PrimaryType]:
        return getattr(cls, typename, None)

    int = (
        {"-", "+", "&"},
        {"+", "-", "*", "/", "%"},
        {"==", "!=", "<", ">", "<=", ">="},
        {"="},
    )
    float = (
        {"-", "+", "&"},
        {"+", "-", "*", "/"},
        {"==", "!=", "<", ">", "<=", ">="},
        {"="},
    )
    char = (
        {"&"},
        {},
        {"==", "!=", "<", ">", "<=", ">="},
        {"="},
    )
    bool = (
        {"!", "&"},
        {"&&", "||"},
        {"==", "!="},
        {"="},
    )
    void = ()  # no valid operation


IntType = PrimaryType.int
CharType = PrimaryType.char
BoolType = PrimaryType.bool
VoidType = PrimaryType.void
FloatType = PrimaryType.float


# # # # # # # #
# Array Type  #

# special type for empty lists that can be coerced to any type
_UndefinedType = uCType("<undefined>")


class ArrayType(uCType):
    __slots__ = "elem_type", "size"

    def __init__(self, element_type: uCType, size: Optional[int] = None):
        """
        element_type: Any of the uCTypes can be used as the array's type. This
            means that there's support for nested types, like matrices.
        size: Integer with the length of the array.
        """
        super().__init__(None, unary_ops={"&"}, rel_ops={"==", "!="})
        self.elem_type = element_type
        self.size = size

    @staticmethod
    def cmp_size(this: uCType, other: uCType) -> bool:
        """Compare array types matching sizes."""
        if isinstance(this, ArrayType) and isinstance(other, ArrayType):
            return this == other and this.size == other.size
        else:
            return this == other

    def __eq__(self, other: uCType) -> bool:
        """Array are equal to other arrays with same basic type and dimensions."""
        if not isinstance(other, ArrayType) or self is other:
            return self is other
        # coercion of undefined types
        if self.elem_type is _UndefinedType:
            self.elem_type = other.elem_type
        elif other.elem_type is _UndefinedType:
            other.elem_type = self.elem_type
        # inner types must have same size
        return ArrayType.cmp_size(self.elem_type, other.elem_type)

    def typename(self) -> str:
        return f"{self.elem_type!r}[{self.size or ''}]"

    @staticmethod
    def empty_list() -> ArrayType:
        """Special type for empty initialization lists: '{}'."""
        return ArrayType(_UndefinedType, 0)

    def out_of_bounds(self, value: Union[int, str]) -> bool:
        """Check if value is inside of bound for array type."""
        try:
            value = int(value)
        except ValueError:
            return True
        # must be nonnegative and less than size, if known
        return value < 0 or (self.size is not None and value >= self.size)


# # # # # # # # #
# Function Type #


class ParamSpec(NamedTuple):
    name: str
    type: uCType


class FunctionType(uCType):
    __slots__ = "funcname", "rettype", "params"

    def __init__(self, name: str, return_type: uCType, params: Sequence[tuple[str, uCType]] = ()):
        """
        name: Function definition name.
        return_type: Any uCType can be used here.
        params: Sequence of 'name, type' for each of the function parameters.
        """
        super().__init__(None, unary_ops={"&"})  # only call and get reference
        self.funcname = name
        self.rettype = return_type
        self.params = tuple(ParamSpec(n, t) for n, t in params)

    @property
    def param_types(self) -> tuple[uCType, ...]:
        return tuple(t for _, t in self.params)

    def __eq__(self, other: uCType) -> bool:
        return self is other or (
            isinstance(other, FunctionType)
            and self.rettype == other.rettype
            and self.param_types == other.param_types
        )

    def typename(self, *, show_names: bool = False) -> str:
        if show_names:
            params = ", ".join(f"{n}: {t!r}" for n, t in self.params)
            return f"{self.rettype!r} {self.funcname}({params})"
        else:
            # no space between parameters
            params = ",".join(f"{t!r}" for t in self.param_types)
            return f"{self.rettype!r}({params})"


# # # # # # # # #
# Pointer Type  #


class PointerType(uCType):
    __slots__ = ("inner",)

    def __init__(self, inner: uCType):
        relation = {"==", "!=", "<", ">", "<=", ">="}
        super().__init__(None, unary_ops={"*", "&"}, rel_ops=relation, assign_ops={"="})
        self.inner = inner

    def __eq__(self, other: uCType) -> bool:
        return isinstance(other, PointerType) and self.inner == other.inner

    def typename(self) -> str:
        return f"ptr<{self.inner!r}>"
