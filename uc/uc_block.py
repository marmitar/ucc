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
    AddInstr,
    AllocInstr,
    ArrayDataVaraible,
    CBranchInstr,
    DataVariable,
    DefineInstr,
    ElemInstr,
    GeInstr,
    GlobalInstr,
    Instruction,
    JumpInstr,
    LabelInstr,
    LabelName,
    LiteralInstr,
    NamedVariable,
    PrintInstr,
    ReturnInstr,
    StoreInstr,
    TempVariable,
    TextVariable,
)
from uc.uc_type import (
    ArrayType,
    CharType,
    FunctionType,
    IntType,
    PointerType,
    VoidType,
    uCType,
)

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

    def subblocks(self) -> Iterator[Block]:
        if self.next is not None:
            yield self.next


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
        self._memcpy: Optional[MemCopy] = None
        self._puts: Optional[PutsBlock] = None
        self.functions: list[FunctionBlock] = []

    def new_text(self, ty: uCType, value: Any) -> TextVariable:
        """Create a new literal constant on the 'text' section."""
        # avoid repeated constants
        varname = self.consts.get((ty, str(value)))
        if varname is not None:
            return varname

        varname = TextVariable((ty.ir(), self._new_version(ty.ir())))
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

    @property
    def memcpy(self) -> DataVariable:
        if self._memcpy is None:
            self._memcpy = MemCopy(self)
        return self._memcpy.label

    @property
    def puts(self) -> DataVariable:
        if self._puts is None:
            self._puts = PutsBlock(self)
        return self._puts.label

    def instructions(self) -> Iterator[Instruction]:
        # show text variables, then data
        return chain(self.text, self.data)

    def subblocks(self) -> Iterator[Block]:
        if self._memcpy:
            yield self._memcpy
        if self._puts:
            yield self._puts
        for function in self.functions:
            yield function


# # # # # # # # # #
# FUNCTION BLOCKS #


class FunctionBlock(CountedBlock):
    """Special block for function definition."""

    def __init__(self, program: GlobalBlock, function: FunctionType):
        super().__init__()
        self.program = program
        # initialize register count on 1
        self._count["%temp%"] = 1

        # function data
        self.name = function.funcname
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

    def alloc(self, uctype: uCType, name: NamedVariable) -> None:
        self.entry.append(AllocInstr(uctype, name))

    def instructions(self) -> Iterator[DefineInstr]:
        yield self.define

    def subblocks(self) -> Iterator[Block]:
        yield self.entry


class PutsBlock(FunctionBlock):
    def __init__(self, program: GlobalBlock):
        # build definition
        self.uctype = FunctionType(
            ".puts", VoidType, [("str", ArrayType(CharType, None)), ("len", IntType)]
        )
        super().__init__(program, self.uctype)
        self.entry = self.entry.next

        # create loop index and constant 1
        index = self.entry.new_literal(0)
        one = self.entry.new_literal(1)

        self.entry.next = loop = BasicBlock(self)  # TODO: loop block
        string, length = (var for _, _, var in self.params)
        result = loop.new_temp()
        # iterate over caracters
        loop.append(
            GeInstr(IntType, index, length, result),
            CBranchInstr(result, true_target=LabelName("exit")),
            ElemInstr(CharType, string, index, result),
            PrintInstr(CharType, result),
            AddInstr(IntType, index, one, index),
            JumpInstr(loop.label),
        )
        # then exit
        loop.next = BasicBlock(self, "exit")
        loop.next.append(ReturnInstr(VoidType))


class MemCopy(FunctionBlock):
    def __init__(self, program: GlobalBlock):
        # build definition
        ptr = PointerType(VoidType)
        self.uctype = FunctionType(".memcpy", ptr, [("src", ptr), ("dest", ptr), ("len", IntType)])
        super().__init__(program, self.uctype)
        self.entry = self.entry.next

        # create loop index and constant 1
        index = self.entry.new_literal(0)
        one = self.entry.new_literal(1)

        self.entry.next = loop = BasicBlock(self)  # TODO: loop block
        src, dest, length = (var for _, _, var in self.params)
        result = loop.new_temp()
        # iterate over caracters
        loop.append(
            GeInstr(IntType, index, length, result),
            CBranchInstr(result, true_target=LabelName("exit")),
            ElemInstr(IntType, src, index, result),
            StoreInstr(IntType, result, dest),
            AddInstr(IntType, index, one, index),
            AddInstr(ptr, dest, one, dest),
            JumpInstr(loop.label),
        )
        # then exit
        loop.next = BasicBlock(self, "exit")
        loop.next.append(ReturnInstr(VoidType))


# # # # # # # #
# CODE BLOCKS #


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

    def new_literal(self, value: Value, uctype: uCType = IntType) -> TempVariable:
        target = self.new_temp()
        self.append(LiteralInstr(uctype, value, target))
        return target

    def alloc(self, uctype: uCType, name: ID, array_data: bool = False) -> NamedVariable:
        if array_data:
            varname = ArrayDataVaraible((name.name, name.version))
        else:
            varname = NamedVariable((name.name, name.version))

        self.function.alloc(uctype, varname)
        return varname

    @property
    def label(self) -> LabelName:
        return LabelName(self.name)

    def append(self, *instr: Instruction) -> None:
        self.instr.extend(instr)

    def instructions(self) -> Iterator[Instruction]:
        init = (LabelInstr(self.name),)
        return chain(init, self.instr)

    def new_conditional(self) -> ConditionBlock:
        self.next = ConditionBlock(self.function)
        return self.next


class EntryBlock(BasicBlock):
    "Block specialized for stack allocations"

    next: BasicBlock

    def __init__(self, function: FunctionBlock, next_block: Optional[str] = None):
        super().__init__(function, name="entry")
        self.next = BasicBlock(function, next_block)

    def new_temp(self) -> TempVariable:
        raise ValueError()

    def new_literal(self, value: Value, uctype: uCType) -> TempVariable:
        raise ValueError()


class ConditionBlock(BasicBlock):
    """
    Class for a block representing an conditional statement.
    There are two branches to handle each possibility.
    """

    next: BasicBlock

    def __init__(self, function: FunctionBlock, name: Optional[str] = None):
        super().__init__(function, name)
        self.taken: Optional[BasicBlock] = None
        self.next = BasicBlock(self.function, f"{self.name}.end")

    def taken_block(self) -> BasicBlock:
        self.taken = BasicBlock(self.function, f"{self.name}.false")
        self.taken.next = self.next
        return self.taken

    def instructions(self) -> Iterator[Instruction]:
        for instr in super().instructions():
            yield instr
        if self.taken is not None:
            yield JumpInstr(self.taken.label)

    def subblocks(self) -> Iterator[Block]:
        if self.taken is not None:
            yield self.taken
        else:
            yield self.next


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
        for subblock in block.subblocks():
            self.visit(subblock, total)


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
