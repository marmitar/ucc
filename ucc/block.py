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
    Optional,
    TypeVar,
    Union,
)
from graphviz import Digraph
from .ast import ID, FuncDef, Program
from .ir import (
    AllocInstr,
    CallInstr,
    CBranchInstr,
    DataVariable,
    DefineInstr,
    ExitInstr,
    GlobalInstr,
    Instruction,
    JumpInstr,
    LabelInstr,
    LabelName,
    LiteralInstr,
    LocalVariable,
    NamedVariable,
    ReturnInstr,
    TargetInstruction,
    TempVariable,
    TextVariable,
    Value,
    Variable,
)
from .type import FunctionType, IntType, VoidType, uCType

# # # # # #
# BLOCKS  #


class Block:
    __slots__ = ("instr", "next")

    name: str

    def __init__(self):
        self.next: Optional[Block] = None

    @property
    def classname(self) -> str:
        return self.__class__.__name__

    def instructions(self) -> Iterator[Instruction]:
        raise NotImplementedError()

    def subblocks(self) -> Iterator[Block]:
        if self.next is not None:
            yield self.next

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return self is other

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.classname}({self.name})"


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

    def __init__(self, program: Program):
        super().__init__()
        self.name = program.name or "<program>"
        program.cfg = self

        self.data: list[GlobalInstr] = []
        self.cdata: list[GlobalInstr] = []
        # cache of defined constants, to avoid repeated values
        self.consts: dict[tuple[str, str], TextVariable] = {}
        # all functions in the program
        self.start: Optional[StartFunction] = None
        self.functions: list[FunctionBlock] = []

    def new_text(self, ty: uCType, value: Any) -> TextVariable:
        """Create a new literal constant on the 'text' section."""
        # avoid repeated constants
        varname = self.consts.get((ty, str(value)))
        if varname is not None:
            return varname

        varname = TextVariable(ty.ir(), self._new_version(ty.ir()))
        # and insert into the text section
        self.cdata.append(GlobalInstr(ty, varname, value))
        self.consts[ty, str(value)] = varname
        return varname

    def new_global(self, uctype: uCType, varname: DataVariable, value: Optional[Any]) -> None:
        self.data.append(GlobalInstr(uctype, varname, value))

    def new_function(self, function: FuncDef) -> FunctionBlock:
        block = FunctionBlock(self, function)
        self.functions.append(block)
        return block

    def add_start(self, main: FunctionType) -> StartFunction:
        self.start = StartFunction(main)
        return self.start

    def instructions(self) -> Iterator[GlobalInstr]:
        # show text variables, then data
        return chain(self.cdata, self.data)

    def subblocks(self) -> Iterator[Block]:
        yield from self.functions


# # # # # # # # # #
# FUNCTION BLOCKS #


class FunctionBlock(CountedBlock):
    """Special block for function definition."""

    def __init__(self, program: GlobalBlock, function: Union[FuncDef, FunctionType]):
        # link node to code gen block
        if isinstance(function, FuncDef):
            self.definition = function
            self.definition.cfg = self
            function = function.func_type
        # for '.start', there is nothing to link
        else:
            self.definition = None

        super().__init__()
        self.name = function.funcname
        self.program = program
        self.fntype = function
        # initialize register count on 1
        self._count["%temp%"] = 1

        # function data
        self.params = [(name, ty, self.new_temp()) for name, ty in function.params]
        # function definition
        self.define = DefineInstr(
            function.rettype,
            DataVariable(self.name),
            ((ty, var) for _, ty, var in self.params),
        )
        self.entry = EntryBlock(self)

    def local_variables(self) -> Iterator[LocalVariable]:
        """Iterator over local (named and temp) variables."""
        # list named variable
        for instr in self.entry.instr:
            if isinstance(instr.target, NamedVariable):
                yield instr.target
        # and temporaries
        for i in range(1, self._count["%temp%"]):
            yield TempVariable(i)

    @property
    def label(self) -> DataVariable:
        return DataVariable(self.name)

    def new_temp(self) -> TempVariable:
        """
        Create a new temporary variable for the function scope.
        """
        return TempVariable(self._new_version("%temp%"))

    def new_label(self) -> str:
        version = self._new_version("label")
        return f".L{version}"

    def alloc(self, uctype: uCType, name: LocalVariable) -> None:
        self.entry.instr.append(AllocInstr(uctype, name))

    def instructions(self) -> Iterator[DefineInstr]:
        yield self.define

    def subblocks(self) -> Iterator[EntryBlock]:
        yield self.entry

    def all_blocks(self) -> Iterator[CodeBlock]:
        block = self.entry
        while block is not None:
            yield block
            block = block.next


class StartFunction(Block):
    """Entry function for a program (may not be needed on llvm)"""

    name = ".start"

    def __init__(self, main: FunctionType):
        super().__init__()
        # '.start' is a function without arguments, that never returns
        self.instr: list[Instruction] = [DefineInstr(VoidType, self.label)]
        self.maintype = main

        temp = TempVariable(1)
        # main returns void, exit with zero
        if main.rettype is VoidType:
            self.instr.append(CallInstr(VoidType, self.main))
            self.instr.append(LiteralInstr(IntType, 0, temp))
        # main returns number, exit with return value
        else:
            self.instr.append(CallInstr(main.rettype, self.main, temp))
        self.instr.append(ExitInstr(temp))

    @property
    def label(self) -> DataVariable:
        return DataVariable(self.name)

    @property
    def main(self) -> DataVariable:
        return DataVariable(self.maintype.funcname)

    def instructions(self) -> Iterator[Instruction]:
        return iter(self.instr)


# # # # # # # #
# CODE BLOCKS #


C = TypeVar("C")


class CodeBlock(Block):
    next: Optional[CodeBlock]

    def __init__(self, function: FunctionBlock, name: Optional[str] = None):
        super().__init__()
        # label definition
        if name is None:
            name = function.new_label()
        self.label = LabelName(name)
        self.label_instr = LabelInstr(self.name)
        self.function = function

        self.instr: list[Instruction] = []

    @property
    def name(self) -> str:
        return self.label.name

    def alloc(self, uctype: uCType, name: ID) -> NamedVariable:
        varname = NamedVariable(name.name, name.version)
        self.function.alloc(uctype, varname)
        return varname

    def new_temp(self) -> TempVariable:
        """Generate new temp variable."""
        return self.function.new_temp()

    def append_instr(self, instr: Instruction) -> None:
        self.instr.append(instr)

    def target_instr(self, instr: type[TargetInstruction], *args) -> TempVariable:
        """Generate instruction and temp variable for output"""
        target = self.new_temp()
        self.instr.append(instr(*args, target))
        return target

    def new_literal(self, value: Value, uctype: uCType = IntType) -> TempVariable:
        """Generate new temp var with literal value"""
        return self.target_instr(LiteralInstr, uctype, value)

    def instructions(self) -> Iterator[Instruction]:
        yield self.label_instr
        yield from self.instr

    def insert(self, block: C) -> C:
        # insert new block in the linked list
        block.next = self.next
        self.next = block
        return block

    def insert_new(self, block: type[C], *args) -> C:
        # create and insert
        return self.insert(block(self.function, *args))


class BasicBlock(CodeBlock):
    """
    Class for a simple basic block.  Control flow unconditionally
    flows to the next block.
    """

    def __init__(self, function: FunctionBlock, name: Optional[str] = None):
        super().__init__(function, name=name)
        self.jumps: list[CodeBlock] = []
        self.jump_instr: list[JumpInstr] = []

    def jump_to(self, block: CodeBlock) -> None:
        self.jumps.append(block)
        self.jump_instr.append(JumpInstr(block.label))

    def clear_jumps(self) -> None:
        self.jumps = []
        self.jump_instr = []

    def instructions(self) -> Iterator[Instruction]:
        yield from super().instructions()
        yield from self.jump_instr


class BranchBlock(CodeBlock):
    def branch(self, condition: Variable, true: CodeBlock, false: CodeBlock) -> None:
        self.condition = condition
        self.taken = true
        self.fallthrough = false
        self.cbranch = CBranchInstr(condition, true.label, false.label)

    def instructions(self) -> Iterator[Instruction]:
        yield from super().instructions()
        yield self.cbranch


class EntryBlock(CodeBlock):
    """Initial block in function, used for stack allocations"""

    next: BasicBlock
    instr: list[AllocInstr]

    def __init__(self, function: FunctionBlock, next_block: Optional[str] = None):
        super().__init__(function, name="entry")
        self.next = BasicBlock(function, name=next_block)

    def new_temp(self) -> TempVariable:
        raise ValueError()


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

    def __init__(self, default: Callable[[Block], C]):
        self.visitor = lru_cache(maxsize=None)(self.visitor)
        self.default = default

    def generic_visit(self, _block: Block, _total: C) -> Optional[C]:
        raise NotImplementedError()

    def visitor(self, classname: str) -> Callable[[Block, C], Optional[C]]:
        return getattr(self, f"visit_{classname}", self.generic_visit)

    def visit(self, block: Block, total: Optional[C] = None) -> C:
        if total is None:
            total = self.default(block)

        value = self.visitor(block.classname)(block, total)
        if value is not None:
            return value
        else:
            return total


class CodeList(List[Instruction]):
    def __init__(self, program: GlobalBlock) -> None:
        self.program = program

    def with_start(self) -> Iterator[Instruction]:
        yield from self
        if self.program.start is not None:
            yield from self.program.start.instructions()


class EmitBlocks(BlockVisitor[CodeList]):
    def __init__(self):
        super().__init__(CodeList)

    def generic_visit(self, block: Block, total: CodeList) -> None:
        total.extend(block.instructions())
        for subblock in block.subblocks():
            self.visit(subblock, total)


# # # # # # # # # # # #
# CONTROL FLOW GRAPH  #


@dataclass
class GraphData:
    """Wrapper for building the CFG graph."""

    def __init__(self, name: str, func_graph: bool = False):
        self.graph = Digraph(name, filename=name + ".gv", node_attr={"shape": "record"})
        self.nodes: dict[Block, str] = {}
        self.func_graph = func_graph

    def build_label(self, name: str = "", instr: Iterable[Instruction] = ()) -> str:
        """Create node label from instructions."""
        if not name and not instr:
            raise ValueError()
        elif not instr:
            return "{" + name + "}"

        body = (i.format() for i in instr)
        return "\\l\t".join(chain(body, [""]))

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
    def __init__(self):
        def new_data(block: Union[FunctionBlock, StartFunction, GlobalBlock]):
            return GraphData(block.name, not isinstance(block, GlobalBlock))

        super().__init__(new_data)

    def visit_CodeBlock(self, block: CodeBlock, g: GraphData) -> None:
        if not g.func_graph:
            name = f"<{block.function.name}>{block.name}"
        else:
            name = block.name
        g.add_node(block, name, block.instructions())

        if block.next is not None:
            self.visit(block.next, g)
            connect = True
        else:
            connect = False

        for instr in block.instructions():
            if isinstance(instr, (JumpInstr, CBranchInstr, ExitInstr, ReturnInstr)):
                connect = False
            elif not g.func_graph and isinstance(instr, CallInstr):
                g.add_edge(block, instr.source.name)

        if connect:
            g.add_edge(block, block.next)

    visit_EntryBlock = visit_CodeBlock

    def visit_BasicBlock(self, block: BasicBlock, g: GraphData) -> None:
        self.visit_CodeBlock(block, g)

        for next in block.jumps:
            g.add_edge(block, next)

    def visit_BranchBlock(self, block: BranchBlock, g: GraphData) -> None:
        self.visit_CodeBlock(block, g)
        g.add_edge(block, block.taken)
        g.add_edge(block, block.fallthrough)

    def visit_GlobalBlock(self, block: GlobalBlock, g: GraphData) -> None:
        # special node for data and text sections
        g.add_node(None, ".cdata", block.cdata)
        g.add_node(block, ".data", block.data)
        # and all functions as well
        for func in block.subblocks():
            self.visit(func, g)

    def visit_FunctionBlock(self, func: FunctionBlock, g: GraphData) -> None:
        g.add_node(func, func.name, func.instructions())
        # connect to the first block
        self.visit(func.entry, g)
        g.add_edge(func, func.entry)

    def visit_StartFunction(self, func: StartFunction, g: GraphData) -> None:
        g.add_node(func, func.name, func.instructions())
        g.add_edge(func, func.main)

    def view(self, block: Block) -> None:
        graph = self.visit(block).graph
        # You can use the next stmt to see the dot file
        # print(graph.source)
        graph.view(quiet=True, quiet_view=True)
