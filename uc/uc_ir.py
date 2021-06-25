from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Generic,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
    Union,
)
from uc.uc_type import VoidType, uCType

# # # # # # # # # #
# Variable Types  #


@dataclass(frozen=True)
class Variable:
    """ABC for variables."""

    name: str
    version: int

    def __eq__(self, other) -> bool:
        return (
            self.__class__ is other.__class__
            and self.name == other.name
            and self.version == other.version
        )

    def format(self) -> str:
        if self.version != 0:
            return f"{self.name}.{self.version}"
        else:
            return f"{self.name}"

    def __hash__(self) -> int:
        return hash((self.name, self.version))

    def __str__(self) -> str:
        return self.format()


class LocalVariable(Variable):
    """ABC for variables (numbered or named) with local scope."""

    __slots__ = ()

    def __str__(self) -> str:
        return f"%{self.format()}"


class GlobalVariable(Variable):
    """ABC for variables with global scope."""

    __slots__ = ()

    def __str__(self) -> str:
        return f"@{self.format()}"


class NamedVariable(LocalVariable):
    """Local variable referenced by name."""

    __slots__ = ()

    def __init__(self, name: str, version: int):
        super().__init__(name, version)


class TempVariable(LocalVariable):
    """Local variable referenced by a temporary number."""

    __slots__ = ()

    name: Literal[""]

    def __init__(self, version: int):
        super().__init__("", version)

    def format(self) -> str:
        return str(self.version)

    def __int__(self) -> int:
        return self.version


class DataVariable(GlobalVariable):
    """Global variable that lives on the 'data' section."""

    __slots__ = ()

    version: Literal[0]

    def __init__(self, name: str):
        super().__init__(name, 0)


class TextVariable(GlobalVariable):
    """Global variable that lives on the 'text' section."""

    __slots__ = ()

    def format(self) -> int:
        return f".const_{self.name}.{self.version}"


@dataclass(frozen=True)
class LabelName:
    """Special variable for block labels."""

    name: str

    def __str__(self) -> str:
        return f"label {self.name}"


# variables that lives on memory
MemoryVariable = Union[NamedVariable, GlobalVariable]

# # # # # # # # # # #
# INSTRUCTION TYPES #


Value = Union[int, float, str, MemoryVariable]


class Instruction:
    __slots__ = ()

    operations: dict[str, Type[Instruction]] = {}

    def __init_subclass__(cls) -> None:
        """Register instruction by opname"""
        opname = getattr(cls, "opname", None)
        if isinstance(opname, str):
            Instruction.operations[opname] = cls

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

        if self.target_attr is not None:
            if (attr := self.get(self.target_attr)) is not None:
                yield attr
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

    def __str__(self) -> str:
        return repr(self.format())

    def __repr__(self) -> str:
        params = ", ".join(f"{attr}={getattr(self, attr, None)}" for attr in self.arguments)
        return f"{self.__class__.__name__}({params})"

    def __hash__(self) -> int:
        return hash(id(self))


class TypedInstruction(Instruction):
    __slots__ = ("type",)

    type: uCType

    def __init__(self, type: uCType):
        super().__init__()
        self.type = type

    @property
    def operation(self) -> str:
        return f"{self.opname}_{self.type.ir()}"


V = TypeVar("V")


class TargetInstruction(Generic[V], TypedInstruction):
    __slots__ = ("target",)

    target_attr: Literal["target"] = "target"

    def __init__(self, type: uCType, target: V):
        super().__init__(type)
        self.target = target


class TempTargetInstruction(TargetInstruction[Optional[TempVariable]]):
    __slots__ = ()

    target: Optional[TempVariable]


# # # # # # # # # # # #
# Variables & Values  #


class AllocInstr(TargetInstruction[LocalVariable]):
    """Allocate on stack (ref by register) a variable of a given type."""

    __slots__ = ()

    opname = "alloc"
    arguments = ("target",)

    def __init__(self, type: uCType, varname: LocalVariable):
        super().__init__(type, varname)

    @property
    def varname(self) -> LocalVariable:
        return self.target


class GlobalInstr(TargetInstruction[GlobalVariable]):
    """Allocate on heap a global var of a given type. value is optional."""

    __slots__ = ("value",)

    opname = "global"
    arguments = "target", "_value"
    indent = False

    def __init__(
        self,
        type: uCType,
        varname: GlobalVariable,
        value: Union[Variable, Value, list[Value], None] = None,
    ):
        super().__init__(type, varname)
        self.value = value

    @property
    def varname(self) -> GlobalVariable:
        return self.target

    @property
    def _value(self) -> Union[Variable, Value, list[Value]]:
        # format string as expected
        if isinstance(self.value, str):
            return f"'{self.value}'"
        else:
            return self.value


class LoadInstr(TempTargetInstruction):
    """Load the value of a variable (stack/heap) into target (register)."""

    __slots__ = ("varname",)

    opname = "load"
    arguments = "varname", "target"
    target: TempVariable

    def __init__(self, type: uCType, varname: Variable, target: TempVariable):
        super().__init__(type, target)
        self.varname = varname


class StoreInstr(TypedInstruction):
    """Store the source/register into target/varname."""

    __slots__ = "source", "target"

    opname = "store"
    arguments = "source", "target"

    def __init__(self, type: uCType, source: Variable, target: Variable):
        super().__init__(type)
        self.source = source
        self.target = target


class LiteralInstr(TempTargetInstruction):
    """Load a literal value into target."""

    __slots__ = ("value",)

    opname = "literal"
    arguments = "_value", "target"
    target: TempVariable

    def __init__(self, type: uCType, value: Value, target: TempVariable):
        super().__init__(type, target)
        self.value = value

    @property
    def _value(self) -> Value:
        # format string as expected
        if isinstance(self.value, str):
            return f"'{self.value}'"
        else:
            return self.value


class ElemInstr(TargetInstruction):
    """Load into target the address of source (array) indexed by index."""

    __slots__ = ("source", "index")

    opname = "elem"
    arguments = "source", "index", "target"

    def __init__(self, type: uCType, source: Variable, index: Variable, target: TempVariable):
        super().__init__(type, target)
        self.source = source
        self.index = index


class GetInstr(TypedInstruction):
    """Store into target the address of source."""

    __slots__ = ("source", "target")

    opname = "get"
    arguments = "source", "target"

    def __init__(self, type: uCType, source: MemoryVariable, target: TempVariable):
        super().__init__(type)
        self.source = source
        self.target = target


# # # # # # # # # # #
# Binary Operations #


class BinaryOpInstruction(TempTargetInstruction):
    __slots__ = ("left", "right")

    arguments = "left", "right", "target"
    target: TempVariable

    def __init__(self, type: uCType, left: Variable, right: Variable, target: TempVariable):
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


class UnaryOpInstruction(TempTargetInstruction):

    __slots__ = ("expr",)

    arguments = "expr", "target"
    target: TempVariable

    def __init__(self, type: uCType, expr: Variable, target: TempVariable):
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

    @property
    def name(self) -> LabelName:
        return LabelName(self.label)


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

    def __init__(
        self,
        expr_test: Variable,
        true_target: LabelName,
        false_target: LabelName,
    ):
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
        source: DataVariable,
        args: Iterable[tuple[uCType, TempVariable]] = (),
    ):
        super().__init__(type)
        self.source = source
        self.args = tuple(DefineParam(type, name) for type, name in args)

    def format(self) -> str:
        return "\n" + super().format()


class CallInstr(TempTargetInstruction):
    """Call a function. target is an optional return value"""

    __slots__ = ("source", "target")

    opname = "call"
    arguments = "source", "target"

    def __init__(
        self, type: uCType, source: GlobalVariable, target: Optional[TempVariable] = None
    ):
        super().__init__(type, target)
        self.source = source


class ReturnInstr(TypedInstruction):
    """Return from function. target is an optional return value"""

    __slots__ = ("target",)

    opname = "return"
    arguments = ("target",)

    def __init__(self, type: uCType = VoidType, target: Optional[Variable] = None):
        super().__init__(type)
        self.target = target


class ParamInstr(TypedInstruction):
    """source is an actual parameter"""

    __slots__ = ("source",)

    opname = "param"
    arguments = ("source",)

    def __init__(self, type: uCType, source: Variable):
        super().__init__(type)
        self.source = source


class ReadInstr(ParamInstr):
    """Read value to source"""

    __slots__ = ()
    opname = "read"

    def __init__(self, type: uCType, source: Variable):
        super().__init__(type, source)


class PrintInstr(ParamInstr):
    """Print value of source"""

    __slots__ = ()
    opname = "print"

    source: Optional[Variable]

    def __init__(self, type: uCType = VoidType, source: Optional[Variable] = None):
        super().__init__(type, source)


class ExitInstr(Instruction):
    """Call exit syscall to terminate program"""

    # Added instruction to emulate exit syscall

    __slots__ = ("source",)

    opname = "exit"
    arguments = ("source",)

    def __init__(self, source: TempVariable):
        super().__init__()
        self.source = source
