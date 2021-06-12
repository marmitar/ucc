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
from uc.uc_ast import ID
from uc.uc_interpreter import Value
from uc.uc_ir import (
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
    Variable,
)
from uc.uc_type import FunctionType, IntType, VoidType, uCType

# # # # # #
# BLOCKS  #


class Block:
    __slots__ = ("instr", "next")

    def __init__(self, name: str):
        self.next: Optional[Block] = None
        self.name = name

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


class CountedBlock(Block):
    __slots__ = ("_count",)

    next: None

    def __init__(self, name: str):
        super().__init__(name)
        self._count = DefaultDict[str, int](int)

    def _new_version(self, key: str) -> int:
        value = self._count[key]
        self._count[key] += 1
        return value


class GlobalBlock(CountedBlock):
    """Main block, able to declare globals and constants."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "program")

        self.data: list[GlobalInstr] = []
        self.text: list[GlobalInstr] = []
        # cache of defined constants, to avoid repeated values
        self.consts: dict[tuple[str, str], TextVariable] = {}
        # all functions in the program
        self.functions: list[FunctionBlock] = []

    def new_text(self, ty: uCType, value: Any) -> TextVariable:
        """Create a new literal constant on the 'text' section."""
        # avoid repeated constants
        varname = self.consts.get((ty, str(value)))
        if varname is not None:
            return varname

        varname = TextVariable(ty.ir(), self._new_version(ty.ir()))
        # and insert into the text section
        self.text.append(GlobalInstr(ty, varname, value))
        self.consts[ty, str(value)] = varname
        return varname

    def new_global(self, uctype: uCType, varname: DataVariable, value: Optional[Any]) -> None:
        self.data.append(GlobalInstr(uctype, varname, value))

    def new_function(self, function: FunctionType) -> FunctionBlock:
        block = FunctionBlock(self, function)
        self.functions.append(block)
        return block

    def add_start(self, rettype: uCType) -> StartFunction:
        start = StartFunction(self, rettype)
        self.functions.append(start)
        return start

    def instructions(self) -> Iterator[Instruction]:
        # show text variables, then data
        return chain(self.text, self.data)

    def subblocks(self) -> Iterator[Block]:
        for function in self.functions:
            yield function


# # # # # # # # # #
# FUNCTION BLOCKS #


class FunctionBlock(CountedBlock):
    """Special block for function definition."""

    def __init__(self, program: GlobalBlock, function: FunctionType):
        super().__init__(function.funcname)
        self.program = program
        # initialize register count on 1
        self._count["%temp%"] = 1

        # function data
        self.params = [(name, ty, self.new_temp()) for name, ty in function.params]
        # function definition
        self.define = DefineInstr(
            function.rettype, DataVariable(self.name), ((ty, var) for _, ty, var in self.params)
        )
        self.entry = EntryBlock(self)

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

    def subblocks(self) -> Iterator[Block]:
        yield self.entry


class StartFunction(FunctionBlock):
    """Function that calls main"""

    name = ".start"

    def __init__(self, program: GlobalBlock, rettype: uCType = VoidType):
        self.uctype = FunctionType(self.name, VoidType)
        super().__init__(program, self.uctype)
        self.entry = self.entry.next

        # main returns void, exit with zero
        if rettype is VoidType:
            self.entry.append_instr(CallInstr(VoidType, DataVariable("main")))
            zero = self.entry.new_literal(0)
            self.entry.append_instr(ExitInstr(zero))
        # main returns number, exit with return value
        else:
            retval = self.entry.new_temp()
            self.entry.append_instr(CallInstr(rettype, DataVariable("main"), retval))
            self.entry.append_instr(ExitInstr(retval))


# # # # # # # #
# CODE BLOCKS #


class BasicBlock(Block):
    """
    Class for a simple basic block.  Control flow unconditionally
    flows to the next block.
    """

    def __init__(self, function: FunctionBlock, name: Optional[str] = None):
        # label definition
        if name is None:
            name = function.new_label()
        super().__init__(name)
        self.function = function

        self.instr: list[Instruction] = []

    @property
    def label(self) -> LabelName:
        return LabelName(self.name)

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

    def jump_to(self, block: BasicBlock) -> None:
        self.instr.append(JumpInstr(block.label))

    def branch(self, condition: Variable, true: BasicBlock, false: BasicBlock) -> None:
        self.instr.append(CBranchInstr(condition, true.label, false.label))

    def instructions(self) -> Iterator[Instruction]:
        yield LabelInstr(self.name)
        for instr in self.instr:
            yield instr

    def insert(self, block: BasicBlock) -> BasicBlock:
        # insert new block in the linked list
        block.next = self.next
        self.next = block
        return block

    def insert_new(self, name: Optional[str] = None) -> BasicBlock:
        # create and insert
        return self.insert(BasicBlock(self.function, name=name))


class EntryBlock(BasicBlock):
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


class EmitBlocks(BlockVisitor[List[Instruction]]):
    def __init__(self):
        super().__init__(lambda _: [])

    def generic_visit(self, block: Block, total: list[Instruction]) -> None:
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
        def new_data(block: Union[FunctionBlock, GlobalBlock]):
            return GraphData(block.name, isinstance(block, FunctionBlock))

        super().__init__(new_data)

    def visit_BasicBlock(self, block: BasicBlock, g: GraphData) -> None:
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
            if isinstance(instr, JumpInstr):
                g.add_edge(block, instr.target.name)
                connect = False
            elif isinstance(instr, CBranchInstr):
                g.add_edge(block, instr.true_target.name)
                g.add_edge(block, instr.false_target.name)
                connect = False
            elif isinstance(instr, (ExitInstr, ReturnInstr)):
                connect = False
            elif not g.func_graph and isinstance(instr, CallInstr):
                g.add_edge(block, instr.source.value)

        if connect:
            g.add_edge(block, block.next)

    visit_EntryBlock = visit_BasicBlock

    def visit_GlobalBlock(self, block: GlobalBlock, g: GraphData) -> None:
        # special node for data and text sections
        g.add_node(None, ".text", block.text)
        g.add_node(block, ".data", block.data)
        # and all functions as well
        for func in block.subblocks():
            self.visit(func, g)

    def visit_FunctionBlock(self, func: FunctionBlock, g: GraphData) -> None:
        g.add_node(func, func.name, func.instructions())
        # connect to the first block
        self.visit(func.entry, g)
        g.add_edge(func, func.entry)

    visit_StartFunction = visit_FunctionBlock

    def view(self, block: Block) -> None:
        graph = self.visit(block).graph
        # You can use the next stmt to see the dot file
        # print(graph.source)
        graph.view(quiet=True, quiet_view=True)
