from __future__ import annotations
from enum import Enum, unique
from typing import NamedTuple, Optional, Sequence, Set, Tuple


class uCType:
    """
    Class that represents a type in the uC language.  Basic
    Types are declared as singleton instances of this type.
    """

    __slots__ = "_typename", "binary_ops", "unary_ops", "rel_ops", "assign_ops"

    def __init__(
        self,
        name: Optional[str],
        unary_ops: Set[str] = set(),
        binary_ops: Set[str] = set(),
        rel_ops: Set[str] = set(),
        assign_ops: Set[str] = set(),
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
        """Primary types are only equal to themselves."""
        return self is other

    def typename(self) -> str:
        """The name of the uCType."""
        return self._typename or "<unnamed>"

    def __repr__(self) -> str:
        return self.typename()

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
    def __init__(self, *op_sets: Set[str]):
        # use enum name
        super().__init__(self.name, *op_sets)

    @classmethod
    def get(cls, typename: str) -> Optional[PrimaryType]:
        return getattr(cls, typename, None)

    int = (
        {"-", "+"},
        {"+", "-", "*", "/", "%"},
        {"==", "!=", "<", ">", "<=", ">="},
        {"="},
    )
    char = (
        {},
        {},
        {"==", "!=", "&&", "||"},
        {"="},
    )
    bool = (
        {"!"},
        {},
        {"==", "!=", "&&", "||"},
        {"="},
    )
    void = ()  # no valid operation


IntType = PrimaryType.int
CharType = PrimaryType.char
BoolType = PrimaryType.bool
VoidType = PrimaryType.void


# # # # # # # # # #
# Compound Types  #

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
        super().__init__(None, unary_ops={"*", "&"}, rel_ops={"==", "!="})
        self.elem_type = element_type
        self.size = size

    def __eq__(self, other: uCType) -> bool:
        """Array are equal to other arrays with same basic type and dimensions."""
        if not isinstance(other, ArrayType) or self is other:
            return self is other
        # coercion of undefined types
        if self.elem_type == _UndefinedType:
            self.elem_type = other.elem_type
        elif other.elem_type == _UndefinedType:
            other.elem_type = self.elem_type
        # type must match after coerced
        return self.elem_type == other.elem_type

    def typename(self) -> str:
        return f"{self.elem_type!r}[{self.size or ''}]"

    @staticmethod
    def empty_list() -> ArrayType:
        """Special type for empty initialization lists: '{}'."""
        return ArrayType(_UndefinedType, 0)


class ParamSpec(NamedTuple):
    name: str
    type: uCType


class FunctionType(uCType):
    __slots__ = "_funcname", "rettype", "params"

    def __init__(self, return_type: uCType, params: Sequence[Tuple[str, uCType]] = ()):
        """
        name: Function definition name.
        return_type: Any uCType can be used here.
        params: Sequence of 'name, type' for each of the function parameters.
        """
        super().__init__(None)  # only valid operation is call
        self._funcname: Optional[str] = None
        self.rettype = return_type
        self.params = tuple(ParamSpec(n, t) for n, t in params)

    @property
    def param_types(self) -> Tuple[uCType, ...]:
        return tuple(t for _, t in self.params)

    @property
    def funcname(self) -> str:
        return self._funcname or "<function>"

    @funcname.setter
    def funcname(self, name: str) -> None:
        # update name if missing
        if not self._funcname:
            self._funcname = name

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
