from __future__ import annotations

from ctypes import (
    CFUNCTYPE,
    c_bool,
    c_char,
    c_char_p,
    c_double,
    c_int32,
    c_void_p,
    pointer,
)
from enum import Enum, unique
from typing import Final, NamedTuple, Optional, Sequence, Union

from llvmlite.ir import types


class uCType:
    """
    Class that represents a type in the uC language.  Basic
    Types are declared as singleton instances of this type.
    """

    __slots__ = "binary_ops", "unary_ops", "rel_ops", "assign_ops"

    def __init__(
        self,
        unary_ops: set[str] = set(),
        binary_ops: set[str] = set(),
        rel_ops: set[str] = set(),
        assign_ops: set[str] = set(),
    ):
        self.unary_ops = frozenset(unary_ops)
        self.binary_ops = frozenset(binary_ops)
        self.rel_ops = frozenset(rel_ops)
        self.assign_ops = frozenset(assign_ops)

    def __eq__(self, other: uCType) -> bool:
        # primary types are only equal to themselves
        return self is other

    def typename(self) -> str:
        """The name of the uCType."""
        raise NotImplementedError()

    def ir(self) -> str:
        """Valid name for uCIR."""
        return self.typename()

    def __ucsize__(self) -> int:
        """Size of type in memory."""
        raise NotImplementedError()

    def __int__(self) -> int:
        return self.__ucsize__()

    def __str__(self) -> str:
        """Standard type formatting."""
        return f"type({self.typename()})"

    def __repr__(self) -> str:
        """Only show typename."""
        return self.typename()

    def __hash__(self) -> int:
        return hash(self.typename())

    def as_llvm(self) -> types.Type:
        raise NotImplementedError()

    def as_ctype(self) -> Optional[type]:
        raise NotImplementedError()


# # # # # # # # #
# Primary Types #

Int = int


@unique
class PrimaryType(uCType, Enum):
    def __new__(cls, *op_sets: set[str]) -> PrimaryType:
        uc_type = uCType.__new__(cls)
        uc_type._value_ = uc_type
        return uc_type

    @classmethod
    def get(cls, typename: str) -> Optional[PrimaryType]:
        return getattr(cls, typename, None)

    def typename(self) -> str:
        return self.name

    def __ucsize__(self) -> Int:
        return 0 if self == VoidType else 1

    def as_llvm(self) -> types.Type:
        if self is IntType:
            return types.IntType(32)
        elif self is CharType:
            return types.IntType(8)
        elif self is BoolType:
            return types.IntType(1)
        elif self is FloatType:
            return types.DoubleType()
        else:
            assert self is VoidType
            return types.VoidType()

    def as_ctype(self) -> Optional[type]:
        if self is IntType:
            return c_int32
        elif self is CharType:
            return c_char
        elif self is BoolType:
            return c_bool
        elif self is FloatType:
            return c_double
        else:
            assert self is VoidType
            return None

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


IntType: Final = PrimaryType.int
CharType: Final = PrimaryType.char
BoolType: Final = PrimaryType.bool
VoidType: Final = PrimaryType.void
FloatType: Final = PrimaryType.float


# # # # # # # #
# Array Type  #

# special type for empty lists that can be coerced to any type
_UndefinedType = uCType("<undefined>")


class ArrayType(uCType):
    __slots__ = "elem_type", "size", "_basic_type"

    def __init__(self, element_type: uCType, size: Optional[int] = None):
        """
        element_type: Any of the uCTypes can be used as the array's type. This
            means that there's support for nested types, like matrices.
        size: Integer with the length of the array.
        """
        super().__init__(unary_ops={"&"}, rel_ops={"==", "!="})
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

    def ir(self) -> str:
        qualifier = "*" if self.size is None else str(self.size)
        return self.elem_type.ir() + "_" + qualifier

    def __hash__(self) -> int:
        return super().__hash__()

    def __ucsize__(self) -> int:
        if self.size is None:
            return PointerType.__ucsize__()
        else:
            return self.size * self.elem_type.__ucsize__()

    def as_llvm(self) -> Union[types.ArrayType, types.PointerType]:
        if self.size is None:
            return self.as_pointer().as_llvm()
        else:
            return types.ArrayType(self.elem_type.as_llvm(), self.size)

    def as_ctype(self) -> type:
        if self.size is None:
            return self.as_pointer().as_ctype()
        else:
            return self.elem_type.as_ctype() * self.size

    @staticmethod
    def empty_list() -> ArrayType:
        """Special type for empty initialization lists: '{}'."""
        return ArrayType(_UndefinedType, 0)

    def out_of_bounds(self, value: Union[int, str]) -> bool:
        """Check if value is inside of bounds for array type."""
        try:
            value = int(value)
        except ValueError:
            return True
        # must be nonnegative and less than size, if known
        return value < 0 or (self.size is not None and value >= self.size)

    def basic_type(self) -> uCType:
        """Extract the innermost element type (i.e. int[2][3] -> int)"""
        # cache value
        if not hasattr(self, "_basic_type"):
            if isinstance(self.elem_type, ArrayType):
                self._basic_type = self.elem_type.basic_type()
            else:
                self._basic_type = self.elem_type
        return self._basic_type

    def as_pointer(self) -> PointerType:
        """Equivalent pointer type."""
        return PointerType(self.basic_type())


class StringType(ArrayType):
    "Type for string literals"

    def __init__(self, size: int):
        super().__init__(CharType, size)

    def typename(self) -> str:
        return "string_literal"

    def ir(self) -> str:
        return "string"


# # # # # # # # #
# Pointer Type  #


class PointerType(uCType):
    __slots__ = ("inner",)

    def __init__(self, inner: uCType):
        relation = {"==", "!=", "<", ">", "<=", ">="}
        super().__init__(unary_ops={"*", "&"}, rel_ops=relation, assign_ops={"="})
        self.inner = inner

    def __eq__(self, other: uCType) -> bool:
        return isinstance(other, PointerType) and self.inner == other.inner

    def __hash__(self) -> int:
        return super().__hash__()

    def typename(self) -> str:
        return f"*{self.inner!r}"

    def ir(self) -> str:
        return self.inner.ir() + "_*"

    @classmethod
    def __ucsize__(cls) -> int:
        # pointer is same as an integer
        return IntType.__ucsize__()

    def as_llvm(self) -> types.PointerType:
        return types.PointerType(self.inner.as_llvm())

    def as_ctype(self) -> type:
        if self.inner is CharType:
            return c_char_p
        if (inner := self.inner.as_ctype()) is not None:
            return pointer(inner)
        else:
            return c_void_p


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
        super().__init__(unary_ops={"&"})  # only call and get reference
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

    def __hash__(self) -> int:
        return super().__hash__()

    def typename(self, *, show_names: bool = False) -> str:
        if show_names:
            params = ", ".join(f"{n}: {t!r}" for n, t in self.params)
            return f"{self.rettype!r} {self.funcname}({params})"
        else:
            # no space between parameters
            params = ",".join(f"{t!r}" for t in self.param_types)
            return f"{self.rettype!r}({params})"

    def ir(self) -> str:
        params = ",".join(ty.ir() for _, ty in self.params)
        return self.rettype.ir() + "_(" + params + ")"

    @classmethod
    def __ucsize__(self) -> int:
        # same as pointer
        return PointerType.__ucsize__()

    def as_llvm(self) -> types.FunctionType:
        params = (ty.as_llvm() for ty in self.param_types)
        return types.FunctionType(self.rettype.as_llvm(), params)

    def as_ctype(self) -> type:
        params = (ty.as_ctype() for ty in self.param_types)
        return CFUNCTYPE(self.rettype.as_ctype(), *params)
