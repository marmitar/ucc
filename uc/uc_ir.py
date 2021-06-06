from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, NamedTuple, Optional, Union
from uc.uc_type import StringType, VoidType, uCType

# # # # # # # # # #
# Variable Types  #


@dataclass(frozen=True)
class Variable:
    name: Union[str, int]

    def __str__(self) -> str:
        return f"%{self.name}"

    def __repr__(self) -> str:
        return str(self.name)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Variable) -> bool:
        return str(self) == str(other)


class NamedVariable(Variable):
    """Variable referenced by name."""

    __slots__ = ()

    name: str

    def __init__(self, name: str):
        super().__init__(name)


class TempVariable(Variable):
    """Variable referenced by a temporary number."""

    __slots__ = ()

    name: int

    def __init__(self, version: int):
        super().__init__(version)


class GlobalVariable(NamedVariable):
    """Variable that lives on the 'data' section."""

    __slots__ = ()

    def __str__(self) -> str:
        return f"@{self.name}"


class TextVariable(GlobalVariable):
    """Variable that lives on the 'text' section."""

    __slots__ = ("version",)

    def __init__(self, typename: str, version: int):
        super().__init__(typename)
        self.version = version

    def __str__(self) -> str:
        return f"@.const_{self.name}.{self.version}"

    def __repr__(self) -> str:
        return f"{self.name}.{self.version}"


class LabelName(NamedVariable):
    """Special variable for block labels."""

    __slots__ = ()

    def __str__(self) -> str:
        return f"label {self.name}"


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

    def values(self) -> Iterator[Value]:
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

    def __init__(self, type: uCType, varname: NamedVariable, target: TempVariable):
        super().__init__(type, target)
        self.varname = varname


class StoreInstr(TypedInstruction):
    """Store the source/register into target/varname."""

    __slots__ = "varname", "target"

    opname = "store"
    arguments = "source", "target"

    def __init__(self, type: uCType, source: TempVariable, target: NamedVariable):
        super().__init__(type)
        self.source = source
        self.target = target


class LiteralInstr(TargetInstruction):
    """Load a literal value into target."""

    __slots__ = ("value",)

    opname = "literal"
    arguments = "value", "target"

    def __init__(self, type: uCType, value: Union[int, str], target: TempVariable):
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

    def __init__(self, type: uCType, source: NamedVariable, target: TempVariable):
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

    source: NamedVariable

    def __init__(self, type: uCType, source: NamedVariable):
        super().__init__(type, source)


class PrintInstr(ParamInstr):
    """Print value of source"""

    __slots__ = ()
    opname = "print"

    source: Optional[Variable]

    def __init__(self, type: uCType = VoidType, source: Optional[Variable] = None):
        super().__init__(type, source)
