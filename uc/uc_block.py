from __future__ import annotations
from typing import Iterator, Optional, Tuple
from graphviz import Digraph

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


class Block:
    def __init__(self, label: str):
        self.label = label  # Label that identifies the block
        self.instructions: list[Instr] = []  # Instructions in the block
        self.predecessors: list[Block] = []  # List of predecessors
        self.next_block: Optional[Block] = None  # Link to the next block

    @property
    def classname(self) -> str:
        return self.__class__.__name__

    def append(self, instr: Instr) -> None:
        self.instructions.append(instr)

    def __iter__(self) -> Iterator[Instr]:
        return iter(self.instructions)


class BasicBlock(Block):
    """
    Class for a simple basic block.  Control flow unconditionally
    flows to the next block.
    """

    def __init__(self, label: str):
        super(self).__init__(label)
        # Not necessary the same as next_block in the linked list
        self.branch: Optional[Block] = None


class ConditionBlock(Block):
    """
    Class for a block representing an conditional statement.
    There are two branches to handle each possibility.
    """

    def __init__(self, label: str):
        super(self).__init__(label)
        self.taken: Optional[Block] = None
        self.fall_through: Optional[Block] = None


class BlockVisitor:
    """
    Class for visiting blocks.  Define a subclass and define
    methods such as visit_BasicBlock or visit_ConditionalBlock to
    implement custom processing (similar to ASTs).
    """

    def visit(self, block: Optional[Block]) -> None:
        while isinstance(block, Block):
            name = f"visit_{block.classname}"
            getattr(self, name, lambda _: None)(block)
            block = block.next_block


class EmitBlocks(BlockVisitor):
    def __init__(self):
        super().__init__()
        self.code: list[Instr] = []

    def visit_BasicBlock(self, block: BasicBlock) -> None:
        for inst in block.instructions:
            self.code.append(inst)

    def visit_ConditionBlock(self, block: ConditionBlock) -> None:
        for inst in block.instructions:
            self.code.append(inst)


class CFG(BlockVisitor):
    def __init__(self, fname: str):
        super().__init__()
        self.fname = fname
        self.g = Digraph("g", filename=fname + ".gv", node_attr={"shape": "record"})

    def visit_BasicBlock(self, block: BasicBlock) -> None:
        # Get the label as node name
        name = block.label
        if name:
            # get the formatted instructions as node label
            label = "{" + name + ":\\l\t"
            for inst in block.instructions[1:]:
                label += format_instruction(inst) + "\\l\t"
            label += "}"
            self.g.node(name, label=label)
            if block.branch:
                self.g.edge(name, block.branch.label)
        else:
            # Function definition. An empty block that connect to the Entry Block
            self.g.node(self.fname, label=None, _attributes={"shape": "ellipse"})
            self.g.edge(self.fname, block.next_block.label)

    def visit_ConditionBlock(self, block: ConditionBlock) -> None:
        # Get the label as node name
        name = block.label
        # get the formatted instructions as node label
        label = "{" + name + ":\\l\t"
        for inst in block.instructions[1:]:
            label += format_instruction(inst) + "\\l\t"
        label += "|{<f0>T|<f1>F}}"
        self.g.node(name, label=label)
        self.g.edge(name + ":f0", block.taken.label)
        self.g.edge(name + ":f1", block.fall_through.label)

    def view(self, block: Optional[Block] = None) -> None:
        self.visit(block)
        # You can use the next stmt to see the dot file
        # print(self.g.source)
        self.g.view(quiet=True, quiet_view=True)
