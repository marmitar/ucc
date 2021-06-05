from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from typing import (
    Any,
    Callable,
    DefaultDict,
    Generic,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    TypeVar,
    Union,
)
from graphviz import Digraph
from uc.uc_type import FunctionType, StringType, uCType

# # # # # # # # # #
# Variable Types  #


@dataclass(frozen=True)
class Variable:
    name: Union[str, int]

    def __str__(self) -> str:
        return f"%{self.name}"

    def __repr__(self) -> str:
        return str(self.name)


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


class TextVariable(NamedVariable):
    """Variable that lives on the 'text' section."""

    __slots__ = ("version",)

    def __init__(self, typename: str, version: int):
        super().__init__(typename)
        self.version = version

    def __str__(self) -> str:
        return f"@.const_{self.name}.{self.version}"

    def __repr__(self) -> str:
        return f"{self.name}.{self.version}"


class GlobalVariable(NamedVariable):
    """Variable that lives on the 'data' section."""

    __slots__ = ()

    def __str__(self) -> str:
        return f"@{self.name}"


class LabelName(NamedVariable):
    """Special variable for block labels."""

    __slots__ = ()

    def __str__(self) -> str:
        return f"label {self.name}"


# # # # # # # # # # #
# INSTRUCTION TYPES #


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

    def values(self) -> Iterator[Any]:
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

    def __init__(self, type: uCType, target: Variable):
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

    def __init__(self, type: uCType, varname: NamedVariable, value: Optional[Any] = None):
        super().__init__(type, varname)
        self._value = value

    def as_tuple(self) -> tuple[str, ...]:
        return self.operation, self.varname, self.value

    @property
    def value(self) -> Optional[Any]:
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

    def __init__(self, type: uCType, value: Any, target: TempVariable):
        super().__init__(type, target)
        self.value = value


class ElemInstr(TargetInstruction):
    """Load into target the address of source (array) indexed by index."""

    __slots__ = ("source", "index")

    opname = "elem"
    arguments = "source", "index", "target"

    def __init__(self, type: uCType, source: Variable, index: TempVariable, target: TempVariable):
        super().__init__(type, target)
        self.source = source
        self.index = index


class GetInstr(Instruction):
    """Store into target the address of source."""

    __slots__ = ("source", "target")

    opname = "get"
    arguments = "source", "target"
    target_attr = "target"

    def __init__(self, source: NamedVariable, target: TempVariable):
        super().__init__()
        self.source = source
        self.target = target


# # # # # # # # # # #
# Binary Operations #


class BinaryOpInstruction(TargetInstruction):
    __slots__ = ("left", "right")

    arguments = "left", "right", "target"

    def __init__(self, type: uCType, left: Variable, right: Variable, target: Variable):
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

    def __init__(self, type: uCType, expr: Variable, target: Variable):
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

    def __init__(self, expr_test: Variable, true_target: LabelName, false_target: LabelName):
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
        self, type: uCType, source: NamedVariable, args: Iterable[tuple[uCType, TempVariable]] = ()
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

    def __init__(self, type: uCType, source: NamedVariable, target: Optional[TempVariable] = None):
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


# # # # # #
# BLOCKS  #


class Block:
    __slots__ = ("instr", "next")

    def __init__(self) -> None:
        self.next: Optional[Block] = None

    @property
    def classname(self) -> str:
        return self.__class__.__name__

    def instructions(self) -> Iterator[Instruction]:
        raise NotImplementedError()


class CountedBlock(Block):
    __slots__ = ("_count",)

    next: None

    def __init__(self):
        super().__init__()
        self._count = DefaultDict[str, int](int)

    def _new_version(self, key: str) -> int:
        value = self._count[key]
        self._count[key] += 1
        return value


class GlobalBlock(CountedBlock):
    """Main block, able to declare globals and constants."""

    def __init__(self):
        super().__init__()

        self.data: list[GlobalInstr] = []
        self.text: list[GlobalInstr] = []
        # cache of defined constants, to avoid repeated values
        self.consts: dict[tuple[str, str], TextVariable] = {}
        # all functions in the program
        self.functions: list[FunctionBlock] = []

    def new_literal(self, ty: uCType, value: Any) -> TextVariable:
        """Create a new literal constant on the 'text' section."""
        # avoid repeated constants
        varname = self.consts.get((ty, str(value)))
        if varname is not None:
            return varname

        # remove non alphanumeric character
        name = "".join(ch if ch.isalnum() else "_" for ch in ty)
        varname = TextVariable(name, self._new_version(name))
        # and insert into the text section
        self.text.append(GlobalInstr(ty, varname, value))
        self.consts[ty, str(value)] = varname
        return varname

    def new_global(self, uctype: uCType, varname: GlobalVariable, value: Optional[Any]) -> None:
        self.data.append(GlobalInstr(uctype, varname, value))

    def instructions(self) -> Iterator[Instruction]:
        # show text variables, then data
        return chain(self.text, self.data)


class FunctionBlock(CountedBlock):
    """Special block for function definition."""

    def __init__(self, program: GlobalBlock, function: FunctionType):
        super().__init__()
        self.program = program
        program.functions.append(self)
        # initialize register count on 1
        self._count["%temp%"] = 1

        # function data
        self.name = function.funcname
        self.params = [(name, ty, self.new_temp()) for name, ty in function.params]
        # function definition
        self.define = DefineInstr(
            function.rettype, GlobalVariable(self.name), ((ty, var) for _, ty, var in self.params)
        )
        self.entry = BasicBlock(self, "entry")

    def new_temp(self) -> TempVariable:
        """
        Create a new temporary variable for the function scope.
        """
        return TempVariable(self._new_version("%temp%"))

    def new_label(self) -> str:
        version = self._new_version("label")
        return f".L{version}"

    def instructions(self) -> Iterator[DefineInstr]:
        yield self.define


class BasicBlock(Block):
    """
    Class for a simple basic block.  Control flow unconditionally
    flows to the next block.
    """

    next: Optional[BasicBlock]

    def __init__(self, function: FunctionBlock, name: Optional[str] = None):
        super().__init__()
        self.function = function

        self.instr: list[Instruction] = []
        # label definition
        if name is None:
            self.name = function.new_label()
        else:
            self.name = name

    def new_temp(self) -> TempVariable:
        return self.function.new_temp()

    @property
    def label(self) -> LabelName:
        return LabelName(self.name)

    def append(self, instr: Instruction) -> None:
        self.instr.append(instr)

    def instructions(self) -> Iterator[Instruction]:
        init = (LabelInstr(self.name),)
        return chain(init, self.instr)


# class ConditionBlock(Block):
#     """
#     Class for a block representing an conditional statement.
#     There are two branches to handle each possibility.
#     """

#     def __init__(self, label: str):
#         super(self).__init__(label)
#         self.taken: Optional[Block] = None
#         self.fall_through: Optional[Block] = None


# # # # # # # # #
# BLOCK VISITOR #

# container
C = TypeVar("C")


class BlockVisitor(Generic[C]):
    """
    Class for visiting blocks.  Define a subclass and define
    methods such as visit_BasicBlock or visit_ConditionalBlock to
    implement custom processing (similar to ASTs).
    """

    def __init__(self, default: Callable[[], C]):
        self.visitor = lru_cache(maxsize=None)(self.visitor)
        self.default = default

    def generic_visit(self, _block: Block, _total: C) -> Optional[C]:
        raise NotImplementedError()

    def visitor(self, classname: str) -> Callable[[Block, C], Optional[C]]:
        return getattr(self, f"visit_{classname}", self.generic_visit)

    def visit(self, block: Block, total: Optional[C] = None) -> C:
        if total is None:
            total = self.default()

        value = self.visitor(block.classname)(block, total)
        if value is not None:
            return value
        else:
            return total


class EmitBlocks(BlockVisitor[List[Instruction]]):
    def __init__(self):
        super().__init__(list)

    def generic_visit(self, block: Block, total: list[Instruction]) -> None:
        total.extend(block.instructions())
        if block.next is not None:
            self.visit(block.next)

    def visit_GlobalBlock(self, block: GlobalBlock, total: list[Instruction]) -> None:
        total.extend(block.instructions())
        for subblock in block.functions:
            self.visit(subblock)


# # # # # # # # # # # #
# CONTROL FLOW GRAPH  #


@dataclass
class GraphData:
    """Wrapper for building the CFG graph."""

    def __init__(self, graph: Digraph):
        self.graph = graph
        self.nodes: dict[Block, str] = {}

    def build_label(self, name: str = "", instr: Iterable[Instruction] = ()) -> str:
        """Create node label from instructions."""
        if not name and not instr:
            raise ValueError()
        elif not instr:
            return "{" + name + "}"

        init = "{"
        if name:
            init += name + ":"
        body = (i.format() for i in instr)
        end = "}"

        return "\\l\t".join(chain((init,), body, (end,)))

    def add_node(
        self,
        block: Optional[Block],
        name: str,
        instr: Iterable[Instruction] = (),
        show_name: bool = True,
    ) -> None:
        label = self.build_label(name if show_name else "", instr)
        self.graph.node(name, label=label)

        if block:
            self.nodes[block] = name

    def add_edge(
        self, tail: Union[str, Block], head: Union[str, Block], label: Optional[str] = None
    ) -> None:
        if isinstance(tail, Block):
            tail = self.nodes[tail]
        if isinstance(head, Block):
            head = self.nodes[head]

        self.graph.edge(tail, head, label=label)


class CFG(BlockVisitor[GraphData]):
    def __init__(self, name: Optional[str] = None):
        # program name
        self.name = name or "program"

        def new_data():
            g = Digraph("g", filename=f"{self.name}.gv", node_attr={"shape": "record"})
            return GraphData(g)

        super().__init__(new_data)

    def generic_visit(self, block: Block, g: GraphData) -> None:
        g.add_node(block, "", block.instructions())

    def visit_GlobalBlock(self, block: GlobalBlock, g: GraphData) -> None:
        # special node for data and text sections
        g.add_node(None, ".text", block.text)
        g.add_node(block, ".data", block.instr)
        # and all functions as well
        for func in block.subblocks():
            self.visit(func, g)

    def visit_FuntionBlock(self, func: FunctionBlock, g: GraphData) -> None:
        g.add_node(func, func.name, func.instr)
        # connect to the first block
        self.visit(func.next, g)
        g.add_edge(func, func.head)

    def visit_BasicBlock(self, block: BasicBlock, g: GraphData) -> None:
        name = f"<{block.function.name}>{block.name}"
        g.add_node(block, name, block.instr)

        if block.next is not None:
            self.visit(block.next, g)
            g.add_edge(block, block.next)
        # TODO:
        # # Function definition. An empty block that connect to the Entry Block
        # self.g.node(self.fname, label=None, _attributes={"shape": "ellipse"})
        # self.g.edge(self.fname, block.next_block.label)

    # def visit_ConditionBlock(self, block: ConditionBlock) -> None:
    #     # Get the label as node name
    #     name = block.label
    #     # get the formatted instructions as node label
    #     label = "{" + name + ":\\l\t"
    #     for inst in block.instructions[1:]:
    #         label += format_instruction(inst) + "\\l\t"
    #     label += "|{<f0>T|<f1>F}}"
    #     self.g.node(name, label=label)
    #     self.g.edge(name + ":f0", block.taken.label)
    #     self.g.edge(name + ":f1", block.fall_through.label)

    def view(self, block: Block) -> None:
        graph = self.visit(block).graph
        # You can use the next stmt to see the dot file
        # print(graph.source)
        graph.view(quiet=True, quiet_view=True)
