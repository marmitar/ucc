from __future__ import annotations
from dataclasses import dataclass
from typing import (
    ClassVar,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from uc.uc_type import StringType, VoidType, uCType

# # # # # # # # # #
# Variable Types  #


T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class Variable(Generic[T]):
    """ABC for variables."""

    value: T

    @property
    def name(self) -> str:
        raise NotImplementedError()

    def __eq__(self, other) -> bool:
        return self.__class__ is other.__class__ and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        return self.name


class LocalVariable(Generic[T], Variable[T]):
    """ABC for variables (numbered or named) with local scope."""

    __slots__ = ()

    @property
    def name(self) -> str:
        return f"%{self.value}"


class GlobalVariable(Variable[T]):
    """ABC for variables with global scope."""

    __slots__ = ()
    _format: ClassVar[str]

    @property
    def name(self) -> str:
        return "@" + self._format.format(self.value)


class NamedVariable(LocalVariable[str]):
    """Variable referenced by name."""

    __slots__ = ()


class TempVariable(LocalVariable[int]):
    """Variable referenced by a temporary number."""

    __slots__ = ()


class DataVariable(GlobalVariable[str]):
    """Variable that lives on the 'data' section."""

    __slots__ = ()
    _format = "{}"


class TextVariable(GlobalVariable[Tuple[str, int]]):
    """Variable that lives on the 'text' section."""

    __slots__ = ()
    _format = ".const_{0}.{1}"

    @property
    def version(self) -> int:
        return self.value[1]


class LabelName(Variable[str]):
    """Special variable for block labels."""

    __slots__ = ()

    def name(self) -> str:
        return f"label {self.name}"


# variables that lives on memory
MemoryVariable = Union[NamedVariable, GlobalVariable]

# # # # # # # # # # #
# INSTRUCTION TYPES #


Value = Union[int, float, str]


class Instruction:
    __slots__ = ()

    opname: str
    type: Optional[uCType] = None
    arguments: tuple[str, ...] = ()
    target_attr: Optional[str] = None
    indent: bool = True

    @property
    def operation(self) -> str:
        return self.opname

    def as_tuple(self) -> tuple[str, ...]:
        values = (getattr(self, attr) for attr in self.arguments)
        return (self.operation,) + tuple(values)

    def get(self, attr: str) -> Optional[str]:
        value = getattr(self, attr, None)
        if value is not None:
            return str(value)

    def values(self) -> Iterator[Union[Value, Variable]]:
        for attr in self.arguments:
            value = getattr(self, attr, None)
            if value is not None:
                yield value

    def format_args(self) -> Iterator[str]:
        if self.indent:
            yield " "

        if self.target is not None:
            yield self.get(self.target_attr)
            yield "="

        yield self.opname

        if self.type is not None:
            yield self.type.ir()

        for attr in self.arguments:
            if attr == self.target_attr:
                continue
            value = self.get(attr)
            if value is not None:
                yield value

    def format(self) -> str:
        return " ".join(self.format_args())


class TypedInstruction(Instruction):
    __slots__ = ("type",)

    type: uCType

    def __init__(self, type: uCType):
        super().__init__()
        self.type = type

    @property
    def operation(self) -> str:
        return f"{self.opname}_{self.type.ir()}"


class TargetInstruction(TypedInstruction):
    __slots__ = ("target",)

    target_attr = "target"

    def __init__(self, type: uCType, target: TempVariable):
        super().__init__(type)
        self.target = target


# # # # # # # # # # # #
# Variables & Values  #


class AllocInstr(TypedInstruction):
    """Allocate on stack (ref by register) a variable of a given type."""

    __slots__ = ("varname",)

    opename = "alloc"
    arguments = ("varname",)
    target_attr = "varname"

    def __init__(self, type: uCType, varname: NamedVariable):
        super().__init__(type)
        self.varname = varname


class GlobalInstr(AllocInstr):
    """Allocate on heap a global var of a given type. value is optional."""

    __slots__ = ("_value",)

    opname = "global"
    arguments = "varname", "value"
    indent = False

    def __init__(
        self, type: uCType, varname: GlobalVariable, value: Union[Value, list[Value], None] = None
    ):
        super().__init__(type, varname)
        self._value = value

    def as_tuple(self) -> tuple[str, ...]:
        return self.operation, self.varname, self.value

    @property
    def value(self) -> Union[Value, list[Value]]:
        # format string as expected
        if isinstance(self.type, StringType):
            return f"'{self._value}'"
        else:
            return self._value


class LoadInstr(TargetInstruction):
    """Load the value of a variable (stack/heap) into target (register)."""

    __slots__ = ("varname",)

    opname = "load"
    arguments = "varname", "target"

    def __init__(self, type: uCType, varname: MemoryVariable, target: TempVariable):
        super().__init__(type, target)
        self.varname = varname


class StoreInstr(TypedInstruction):
    """Store the source/register into target/varname."""

    __slots__ = "varname", "target"

    opname = "store"
    arguments = "source", "target"

    def __init__(self, type: uCType, source: TempVariable, target: MemoryVariable):
        super().__init__(type)
        self.source = source
        self.target = target


class LiteralInstr(TargetInstruction):
    """Load a literal value into target."""

    __slots__ = ("value",)

    opname = "literal"
    arguments = "value", "target"

    def __init__(self, type: uCType, value: Value, target: TempVariable):
        super().__init__(type, target)
        self.value = value


class ElemInstr(TargetInstruction):
    """Load into target the address of source (array) indexed by index."""

    __slots__ = ("source", "index")

    opname = "elem"
    arguments = "source", "index", "target"

    def __init__(
        self, type: uCType, source: TempVariable, index: TempVariable, target: TempVariable
    ):
        super().__init__(type, target)
        self.source = source
        self.index = index


class GetInstr(TargetInstruction):
    """Store into target the address of source."""

    __slots__ = ("source",)

    opname = "get"
    arguments = "source", "target"

    def __init__(self, type: uCType, source: MemoryVariable, target: TempVariable):
        super().__init__(type)
        self.source = source
        self.target = target


class CopyInstr(TypedInstruction):
    """Copy contents from a memory region."""

    __slots__ = ("source", "target")

    opname = "copy"
    arguments = "source", "target"

    def __init__(self, type: uCType, source: TempVariable, target: TempVariable):
        super().__init__(type)
        self.source = source
        self.target = target


# # # # # # # # # # #
# Binary Operations #


class BinaryOpInstruction(TargetInstruction):
    __slots__ = ("left", "right")

    arguments = "left", "right", "target"

    def __init__(
        self, type: uCType, left: TempVariable, right: TempVariable, target: TempVariable
    ):
        super().__init__(type, target)
        self.left = left
        self.right = right


class AddInstr(BinaryOpInstruction):
    """target = left + right"""

    opname = "add"


class SubInstr(BinaryOpInstruction):
    """target = left - right"""

    opname = "sub"


class MulInstr(BinaryOpInstruction):
    """target = left * right"""

    opname = "mul"


class DivInstr(BinaryOpInstruction):
    """target = left / right"""

    opname = "div"


class ModInstr(BinaryOpInstruction):
    """target = left % right"""

    opname = "mod"


# # # # # # # # # # #
# Unary Operations  #


class UnaryOpInstruction(TargetInstruction):

    __slots__ = ("expr",)

    arguments = "expr", "target"

    def __init__(self, type: uCType, expr: TempVariable, target: TempVariable):
        super().__init__(type, target)
        self.expr = expr


class NotInstr(UnaryOpInstruction):
    """target = !expr"""

    opname = "not"


# # # # # # # # # # # # # # # #
# Relational/Equality/Logical #


class LogicalInstruction(BinaryOpInstruction):
    __slots__ = ()


class LtInstr(LogicalInstruction):
    """target = left < right"""

    opname = "lt"


class LeInstr(LogicalInstruction):
    """target = left <= right"""

    opname = "le"


class GtInstr(LogicalInstruction):
    """target = left > right"""

    opname = "gt"


class GeInstr(LogicalInstruction):
    """target = left >= right"""

    opname = "ge"


class EqInstr(LogicalInstruction):
    """target = left == right"""

    opname = "eq"


class NeInstr(LogicalInstruction):
    """target = left != right"""

    opname = "ne"


class AndInstr(LogicalInstruction):
    """target = left && right"""

    opname = "and"


class OrInstr(LogicalInstruction):
    """target = left || right"""

    opname = "or"


# # # # # # # # # # #
# Labels & Branches #


class LabelInstr(Instruction):
    """Label definition"""

    __slots__ = ("label",)

    indent = False

    def __init__(self, label: str):
        super().__init__()
        self.label = label

    @property
    def opname(self) -> str:
        return f"{self.label}:"


class JumpInstr(Instruction):
    """Jump to a target label"""

    __slots__ = ("target",)

    opname = "jump"
    arguments = ("target",)

    def __init__(self, target: LabelName):
        super().__init__()
        self.target = target


class CBranchInstr(Instruction):
    """Conditional Branch"""

    __slots__ = ("expr_test", "true_target", "false_target")

    opname = "cbranch"
    arguments = "expr_test", "true_target", "false_target"

    def __init__(self, expr_test: TempVariable, true_target: LabelName, false_target: LabelName):
        super().__init__()
        self.expr_test = expr_test
        self.true_target = true_target
        self.false_target = false_target


# # # # # # # # # # # # #
# Functions & Builtins  #


class DefineParam(NamedTuple):
    """Parameters for the 'define' instruction"""

    type: uCType
    name: TempVariable

    def __str__(self) -> str:
        return f"{self.type.ir()} {self.name}"

    def __repr__(self) -> str:
        return f"({self.type.typename()}, {self.name})"


class DefineInstr(TypedInstruction):
    """
    Function definition. Source=function label, args=list of pairs
    (type, name) of formal arguments.
    """

    __slots__ = ("source", "args")

    opname = "define"
    arguments = "source", "args"
    indent = False

    def __init__(
        self,
        type: uCType,
        source: GlobalVariable,
        args: Iterable[tuple[uCType, TempVariable]] = (),
    ):
        super().__init__(type)
        self.source = source
        self.args = tuple(DefineParam(type, name) for type, name in args)

    def format(self) -> str:
        return "\n" + super().format()


class CallInstr(TypedInstruction):
    """Call a function. target is an optional return value"""

    __slots__ = ("source", "target")

    opname = "call"
    arguments = "source", "target"

    def __init__(
        self, type: uCType, source: GlobalVariable, target: Optional[TempVariable] = None
    ):
        super().__init__(type)
        self.source = source
        self.target = target

    @property
    def target_attr(self) -> Optional[str]:
        if self.target is None:
            return None
        else:
            return "target"


class ReturnInstr(TypedInstruction):
    """Return from function. target is an optional return value"""

    __slots__ = ("target",)

    opname = "return"
    arguments = ("target",)

    def __init__(self, type: uCType, target: Optional[TempVariable] = None):
        super().__init__(type)
        self.target = target


class ParamInstr(TypedInstruction):
    """source is an actual parameter"""

    __slots__ = ("source",)

    opname = "param"
    arguments = ("source",)

    def __init__(self, type: uCType, source: TempVariable):
        super().__init__(type)
        self.source = source


class ReadInstr(ParamInstr):
    """Read value to source"""

    __slots__ = ()
    opname = "read"

    def __init__(self, type: uCType, source: TempVariable):
        super().__init__(type, source)


class PrintInstr(ParamInstr):
    """Print value of source"""

    __slots__ = ()
    opname = "print"

    source: Optional[TempVariable]

    def __init__(self, type: uCType = VoidType, source: Optional[TempVariable] = None):
        super().__init__(type, source)
