from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from typing import (
    Callable,
    DefaultDict,
    Generic,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
from graphviz import Digraph
from uc.uc_type import FunctionType, uCType

Instr = Tuple[str, ...]


def format_instruction(t: Instr) -> str:
    operand = t[0].split("_")
    op = operand[0]
    ty = operand[1] if len(operand) > 1 else None
    if len(operand) >= 3:
        for qual in operand[2:]:
            if qual == "*":
                ty += "*"
            else:
                ty += f"[{qual}]"
    if len(t) > 1:
        if op == "define":
            return f"\n{op} {ty} {t[1]} (" + ", ".join(" ".join(el) for el in t[2]) + ")"
        else:
            _str = "" if op == "global" else "  "
            if op == "jump":
                _str += f"{op} label {t[1]}"
            elif op == "cbranch":
                _str += f"{op} {t[1]} label {t[2]} label {t[3]}"
            elif op == "global":
                if ty.startswith("string"):
                    _str += f"{t[1]} = {op} {ty} '{t[2]}'"
                elif len(t) > 2:
                    _str += f"{t[1]} = {op} {ty} {t[2]}"
                else:
                    _str += f"{t[1]} = {op} {ty}"
            elif op == "return" or op == "print":
                _str += f"{op} {ty} {t[1]}"
            elif op == "sitofp" or op == "fptosi":
                _str += f"{t[2]} = {op} {t[1]}"
            elif op == "store" or op == "param":
                _str += f"{op} {ty} "
                for el in t[1:]:
                    _str += f"{el} "
            else:
                _str += f"{t[-1]} = {op} {ty} "
                for el in t[1:-1]:
                    _str += f"{el} "
            return _str
    elif ty == "void":
        return f"  {op}"
    else:
        return f"{op}"


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
    type: Optional[str] = None
    arguments: tuple[str, ...] = ()
    target_attr: Optional[str] = None
    indent: bool = True

    @property
    def operation(self) -> str:
        if self.type is not None:
            return f"{self.opname}_{self.type}"
        else:
            return self.opname

    def get(self, attr: str) -> Optional[str]:
        value = getattr(self, attr, None)
        if value is not None:
            return str(value)

    def format_args(self) -> Iterator[str]:
        if self.indent:
            yield " "

        if self.target is not None:
            yield self.get(self.target_attr)
            yield "="

        yield self.opname

        if self.type is not None:
            yield self.type

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

    type: str

    def __init__(self, type: str):
        super().__init__()
        self.type = type


class TargetInstruction(TypedInstruction):
    __slots__ = ("target",)

    target_attr = "target"

    def __init__(self, type: str, target: Variable):
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

    def __init__(self, type: str, varname: Variable):
        super().__init__(type)
        self.varname = varname


class GlobalInstr(AllocInstr):
    """Allocate on heap a global var of a given type. value is optional."""

    __slots__ = ("value",)

    opname = "global"
    arguments = "varname", "value"
    indent = False

    def __init__(self, type: str, varname: Variable, value: Optional[str] = None):
        super().__init__(type, varname)
        # format string as expected
        if self.type.startswith("string") and value is not None:
            self.value = f"'{value}'"
        else:
            self.value = value


class LoadInstr(TargetInstruction):
    """Load the value of a variable (stack/heap) into target (register)."""

    __slots__ = ("varname",)

    opname = "load"
    arguments = "varname", "target"

    def __init__(self, type: str, varname: Variable, target: Variable):
        super().__init__(type, target)
        self.varname = varname


class StoreInstr(TargetInstruction):
    """Store the source/register into target/varname."""

    __slots__ = ("varname",)

    opname = "store"
    arguments = "source", "target"
    target_attr = None

    def __init__(self, type: str, varname: Variable, target: Variable):
        super().__init__(type, target)
        self.varname = varname


class LiteralInstr(TargetInstruction):
    """Load a literal value into target."""

    __slots__ = ("value",)

    opname = "literal"
    arguments = "value", "target"

    def __init__(self, type: str, value: str, target: Variable):
        super().__init__(type, target)
        self.value = value


class ElemInstr(TargetInstruction):
    """Load into target the address of source (array) indexed by index."""

    __slots__ = ("source", "index")

    opname = "elem"
    arguments = "source", "index", "target"

    def __init__(self, type: str, source: Variable, index: Variable, target: Variable):
        super().__init__(type, target)
        self.source = source
        self.index = index


class GetInstr(TargetInstruction):
    """Store into target the address of source."""

    __slots__ = ("source",)

    opname = "get"
    arguments = "source", "target"

    def __init__(self, type: str, source: Variable, target: Variable):
        super().__init__(type, target)
        self.source = source


# # # # # # # # # # #
# Binary Operations #


class BinaryOpInstruction(TargetInstruction):
    __slots__ = ("left", "right")

    arguments = "left", "right", "target"

    def __init__(self, type: str, left: Variable, right: Variable, target: Variable):
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

    def __init__(self, type: str, expr: Variable, target: Variable):
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

    type: str
    name: Variable

    def __str__(self) -> str:
        return f"{self.type} {self.name}"

    def __repr__(self) -> str:
        return f"({self.type}, {self.name})"


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
        self, type: str, source: NamedVariable, args: Iterable[tuple[str, Variable]] = ()
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

    def __init__(self, type: str, source: Variable, target: Optional[Variable] = None):
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

    def __init__(self, type: str, target: Optional[Variable] = None):
        super().__init__(type)
        self.target = target


class ParamInstr(TypedInstruction):
    """source is an actual parameter"""

    __slots__ = ("source",)

    opname = "param"
    arguments = ("source",)

    def __init__(self, type: str, source: Variable):
        super().__init__(type)
        self.source = source


class ReadInstr(ParamInstr):
    """Read value to source"""

    __slots__ = ()
    opname = "read"


class PrintInstr(ParamInstr):
    """Print value of source"""

    __slots__ = ()
    opname = "print"


# # # # # #
# BLOCKS  #


class Block:
    __slots__ = ()

    @property
    def classname(self) -> str:
        return self.__class__.__name__

    def instructions(self) -> Iterator[Instruction]:
        raise NotImplementedError()

    def subblocks(self) -> Iterator[Block]:
        raise NotImplementedError()


class CountedBlock(Block):
    __slots__ = ("_count",)

    def __init__(self, initial: int = 0):
        super().__init__()
        self._count = DefaultDict[str, int](lambda: initial)

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
        # list o function blocks
        self.functions: list[FunctionBlock] = []

    def new_function(self, uctype: FunctionType) -> FunctionBlock:
        """Create a new function block."""
        # types and variable names
        rettype = uctype.rettype.typename()
        varname = GlobalVariable(uctype.funcname)
        params = (ty.typename() for ty in uctype.param_types)
        # create function block
        block = FunctionBlock(self, rettype, varname, params)
        self.functions.append(block)
        return block

    def new_global(self, name: str, ty: uCType, init: Optional[str] = None) -> GlobalVariable:
        """Create a new global variable on the 'data' section."""
        varname = GlobalVariable(name)
        self.data.append(GlobalInstr(ty.typename(), varname, init))
        return varname

    def new_literal(self, typename: str, value: str) -> TextVariable:
        """Create a new literal constant on the 'text' section."""
        # avoid repeated constants
        varname = self.consts.get((typename, value))
        if varname is not None:
            return varname

        # remove non alphanumeric character
        name = "".join(ch if ch.isalnum() else "_" for ch in typename)
        varname = TextVariable(name, self._new_version(name))
        # and insert into the text section
        self.text.append(GlobalInstr(typename, varname, value))
        self.consts[typename, value] = varname
        return varname

    def instructions(self) -> Iterator[GlobalInstr]:
        # show text variables, then data
        return chain(self.text, self.data)

    def subblocks(self) -> Iterator[FunctionBlock]:
        return iter(self.functions)


class FunctionBlock(CountedBlock):
    """Special block for function definition."""

    def __init__(
        self,
        parent: GlobalBlock,
        rettype: str,
        funcname: NamedVariable,
        param_types: Iterable[str] = (),
    ):
        super().__init__()
        self.parent = parent

        params = ((ty, self.new_temp()) for ty in param_types)
        self.define = DefineInstr(rettype, funcname, params)
        # function body
        self.blocks: list[BasicBlock] = []

    def new_temp(self) -> TempVariable:
        return TempVariable(self._new_version("temp"))

    def named_var(self, name: str) -> NamedVariable:
        return NamedVariable(name)

    def new_block(self, name: Optional[str] = None) -> BasicBlock:
        if name is None:
            # generate generic name
            version = self._new_version("label")
            name = f".L{version}"

        block = BasicBlock(self, name)
        self.blocks.append(block)
        return block

    def instructions(self) -> Iterator[DefineInstr]:
        yield self.define

    def subblocks(self) -> Iterator[BasicBlock]:
        return iter(self.blocks)


class BasicBlock(Block):
    """
    Class for a simple basic block.  Control flow unconditionally
    flows to the next block.
    """

    def __init__(self, function: FunctionBlock, name: str):
        super().__init__()
        self.function = function

        self.instr: list[Instruction] = []
        # label definition
        self.label_def = LabelInstr(name)

    @property
    def name(self) -> str:
        return self.label_def.label

    @property
    def label(self) -> LabelName:
        return LabelName(self.name)

    def append(self, instr: Instruction) -> None:
        self.instr.append(instr)

    def new_temp(self) -> TempVariable:
        return self.function.new_temp()

    def named_var(self, name: str) -> NamedVariable:
        return self.function.named_var(name)

    def new_literal(self, typename: str, value: str) -> TextVariable:
        return self.function.parent.new_literal(typename, value)

    def instructions(self) -> Iterator[Instruction]:
        return chain((self.label_def,), self.instr)

    def subblocks(self) -> Iterator[Block]:
        return iter(())


# class ConditionBlock(Block):
#     """
#     Class for a block representing an conditional statement.
#     There are two branches to handle each possibility.
#     """

#     def __init__(self, label: str):
#         super(self).__init__(label)
#         self.taken: Optional[Block] = None
#         self.fall_through: Optional[Block] = None

# container and value
C = TypeVar("C")
V = TypeVar("V")


class BlockVisitor(Generic[C, V]):
    """
    Class for visiting blocks.  Define a subclass and define
    methods such as visit_BasicBlock or visit_ConditionalBlock to
    implement custom processing (similar to ASTs).
    """

    def __init__(self, combine: Callable[[C, V], Optional[C]], default: Callable[[], C]):
        self.visitor = lru_cache(maxsize=None)(self.visitor)
        self.default = default
        # insert new value in total
        def append(total: C, data: Optional[V]) -> C:
            # nothing ot insert
            if data is None:
                return total
            new_total = combine(total, data)
            # insertion may be in-place
            if new_total is None:
                return total
            else:
                return new_total

        self.combine = append

    def generic_visit(self, _block: Block) -> Optional[V]:
        raise NotImplementedError()

    def visitor(self, classname: str) -> Callable[[Block], Optional[V]]:
        return getattr(self, f"visit_{classname}", self.generic_visit)

    def visit(self, block: Block) -> C:
        result = self.default()

        value = self.visitor(block.classname)(block)
        result = self.combine(result, value)

        for subblock in block.subblocks():
            result = self.combine(result, self.visit(subblock))

        return result


class EmitBlocks(BlockVisitor[List[Instruction], Iterator[Instruction]]):
    def __init__(self):
        super().__init__(list.extend, default=list)

    def generic_visit(self, block: Block) -> Iterator[Instruction]:
        return block.instructions()


@dataclass
class NodeData:
    instr: tuple[Instruction, ...]
    name: Optional[str]

    def __init__(self, instr: Iterable[Instruction] = (), name: Optional[str] = None):
        self.instr = tuple(instr)
        self.name = name

    def as_label(self) -> str:
        """Create node label from data."""
        if self.name:
            init = ("{" + self.name + ":",)
        else:
            init = "{"
        instr = (i.format() for i in self.instr)
        end = "}"

        return "\\l\t".join(chain(init, instr, end))


class CFG(BlockVisitor[Digraph, NodeData]):
    def __init__(self, name: str):
        # initializer for base BlockVisitor
        def new_graph() -> Digraph:
            return Digraph("g", filename=f"{name}.gv", node_attr={"shape": "record"})

        def add_node(g: Digraph, node: NodeData):
            # TODO: edges
            g.node(node.name, node.as_label())

        super().__init__(add_node, new_graph)

    def generic_visit(self, block: Block) -> NodeData:
        return NodeData(block.instructions())

    def visit_BasicBlock(self, block: BasicBlock) -> NodeData:
        return NodeData(block.instr, block.name)
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
        graph = self.visit(block)
        # You can use the next stmt to see the dot file
        # print(graph.source)
        graph.view(quiet=True, quiet_view=True)
