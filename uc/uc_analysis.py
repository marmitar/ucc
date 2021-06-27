from __future__ import annotations
import argparse
import pathlib
import sys
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, unique
from typing import (
    Callable,
    Dict,
    Literal,
    Optional,
    Set,
    TextIO,
    Tuple,
    Union,
    overload,
)
from uc.uc_ast import FuncDef, Program
from uc.uc_block import CFG, CodeBlock, EmitBlocks, FunctionBlock, GlobalBlock
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_ir import (
    AllocInstr,
    BinaryOpInstruction,
    CallInstr,
    CBranchInstr,
    DivInstr,
    ElemInstr,
    ExitInstr,
    GetInstr,
    GlobalInstr,
    Instruction,
    JumpInstr,
    LabelInstr,
    LiteralInstr,
    LoadInstr,
    NotInstr,
    ParamInstr,
    ReadInstr,
    ReturnInstr,
    StoreInstr,
    TempTargetInstruction,
    TextVariable,
    UnaryOpInstruction,
    Variable,
)
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor
from uc.uc_type import ArrayType, BoolType, FloatType, PointerType

FlowBlock = Union[CodeBlock, GlobalBlock]
VarData = Dict[Variable, Tuple[Instruction, ...]]
Locations = Set[Tuple[FlowBlock, Instruction]]
InOut = Dict[Variable, Locations]
GenKill = Tuple[Tuple[Variable, ...], Tuple[Variable, ...]]


# # # # # # # # # # #
# Generic Data Flow #


class BlockData:
    """Info for a basic block in data flow analysis"""

    def __init__(self, block: Union[GlobalBlock, CodeBlock]) -> None:
        self.block = block
        # successors for this block
        self.succ: set[FlowBlock] = set()
        # and predecessors
        self.pred: set[FlowBlock] = set()

        # input in data flow equations
        self.inp: InOut = {}
        # and output
        self.out: InOut = {}
        # GEN for all instructions in this block
        self.gen: VarData = {}
        # KILL entire block
        self.kill: dict[Variable, Instruction] = {}

        # IN and OUT for each instruction
        self.iinp: dict[Instruction, InOut] = {}
        self.iout: dict[Instruction, InOut] = {}
        # GEN and KILL for each instruction in this block
        self.igen: dict[Instruction, tuple[Variable, ...]] = {}
        self.ikill: dict[Instruction, tuple[Variable, ...]] = {}

    def update(
        self, instr: Instruction, gen: tuple[Variable, ...], kill: tuple[Variable, ...]
    ) -> None:
        """Update GEN and KILL for this block"""

        self.igen[instr] = gen
        self.ikill[instr] = kill

        for var in kill:
            if var in self.gen:
                del self.gen[var]
            self.kill[var] = instr
        for var in gen:
            vgen = self.gen.get(var, ())
            self.gen[var] = vgen + (instr,)

    def build_forward_in_out(self) -> None:
        """Build IN and OUT for each instruction in forward analysis"""
        inout = self.inp
        for instr in self.block.instructions():
            self.iinp[instr] = inout

            new: InOut = {}
            for var, data in inout.items():
                if var not in self.ikill[instr]:
                    new[var] = data
            for var in self.igen[instr]:
                data = new.get(var, set())
                new[var] = data | {(self.block, instr)}

            inout = self.iout[instr] = new
        assert inout == self.out

    def build_backward_in_out(self) -> None:
        """Build IN and OUT for each instruction in backward analysis"""
        inout = self.out
        for instr in self.block.instructions():
            self.iout[instr] = inout

            new: InOut = {}
            for var, data in inout.items():
                if var not in self.ikill[instr]:
                    new[var] = data
            for var in self.igen[instr]:
                data = new.get(var, set())
                new[var] = data | {(self.block, instr)}

            inout = self.iinp[instr] = new
        assert inout == self.inp

    def apply(self, data: InOut) -> InOut:
        """Apply GEN and KILL for input/output sets"""

        new: InOut = {}
        for var, instr in data.items():
            if var not in self.kill:
                new[var] = instr
        for var, instr in self.gen.items():
            current = new.get(var, set())
            new[var] = current | {(self.block, i) for i in instr}
        return new


class DataFlowAnalysis:
    """ABC for data flow analysis (forward and backward)"""

    def __init__(self, function: FunctionBlock) -> None:
        self.function = function

        # mapping for labels to blocks
        self.blocks = {block.label: block for block in function.all_blocks()}
        # data flow info
        self.data: dict[FlowBlock, BlockData] = {
            block: self.block_transfer(block) for block in function.all_blocks()
        }
        self.data[function.program] = self.global_transfer(function.program, function.entry)

        # build predecessors, IN and OUT
        self._update_preds()
        self._build_in_out()

    def _update_preds(self) -> None:
        """Set predecessors for all block (run only once)"""
        for tail, data in self.data.items():
            for head in data.succ:
                self.data[head].pred.add(tail)

    # # # # # # # # # # # #
    # Data Flow Equations #

    def _forward_eq(self, block: FlowBlock) -> bool:
        """Data flow equations for forward analysis"""
        changed = False

        # generate IN as union(OUT[b] for pred b)
        inp: InOut = {}
        for pred in self.data[block].pred:
            for var, dat in self.data[pred].out.items():
                instr = inp.get(var, set())
                inp[var] = instr | dat
        # mark if changed
        if inp != self.data[block].inp:
            changed = True
        self.data[block].inp = inp

        # generate OUT as fB(IN) and mark if changed
        out = self.data[block].apply(inp)
        if out != self.data[block].out:
            changed = True
        self.data[block].out = out

        return changed

    def _backward_eq(self, block: FlowBlock) -> bool:
        """Data flow equations for backward analysis"""
        changed = False

        # generate OUT as union(IN[b] for succ b)
        out: InOut = {}
        for succ in self.data[block].succ:
            for var, dat in self.data[succ].inp.items():
                instr = out.get(var, set())
                out[var] = instr | dat
        # mark if changed
        if out != self.data[block].out:
            changed = True
        self.data[block].out = out

        # generate IN as fB(OUT) and mark if changed
        inp = self.data[block].apply(out)
        if inp != self.data[block].inp:
            changed = True

        return changed

    def _equation(self, _: FlowBlock) -> bool:
        """Data flow equations (forward and backward)"""
        raise NotImplementedError

    def _build_instr_in_out(_: BlockData) -> None:
        """Build IN and OUT for each instruction"""
        raise NotImplementedError

    def _build_in_out(self) -> None:
        """Build IN and OUT sets"""
        changed = True
        # until stable
        while changed:
            changed = False

            for block in self.data.keys():
                # apply for/backward equations
                changed |= self._equation(block)

        # generate IN and OUT for each instruction
        for data in self.data.values():
            self._build_instr_in_out(data)

    Position = Union[FlowBlock, Tuple[FlowBlock, Instruction]]

    # fmt: off
    @overload
    def in_set(self, block: Position, var: None = None) -> InOut:
        ...
    @overload
    def in_set(self, block: Position, var: Variable) -> Locations:
        ...
    # fmt: on
    def in_set(self, block: Position, var: Optional[Variable] = None) -> Union[InOut, Locations]:
        """IN set for a block or instruction"""
        if isinstance(block, tuple):
            block, instr = block
            data = self.data[block].iinp[instr]
        else:
            data = self.data[block].inp

        if var is not None:
            return data[var]
        else:
            return data

    # fmt: off
    @overload
    def out_set(self, block: Position, var: None = None) -> InOut:
        ...
    @overload
    def out_set(self, block: Position, var: Variable) -> Locations:
        ...
    # fmt: on
    def out_set(self, block: Position, var: Optional[Variable] = None) -> Union[InOut, Locations]:
        """OUT set for a block or instruction"""
        if isinstance(block, tuple):
            block, instr = block
            data = self.data[block].iout[instr]
        else:
            data = self.data[block].out

        if var is not None:
            return data[var]
        else:
            return data

    # # # # # # # # # # # #
    # Transfer functions  #

    def block_transfer(self, block: CodeBlock) -> BlockData:
        """Generate transfer functions and CFG connections"""
        data = BlockData(block)

        connect = True
        for instr in block.instructions():
            # update block GEN and KILL
            gen, kill = self.defs(instr)
            data.update(instr, gen, kill)

            # mark CFG successors
            if isinstance(instr, CBranchInstr):
                data.succ.add(self.blocks[instr.true_target])
                data.succ.add(self.blocks[instr.false_target])
                connect = False
            elif isinstance(instr, JumpInstr):
                data.succ.add(self.blocks[instr.target])
                connect = False
            elif isinstance(instr, (ExitInstr, ReturnInstr)):
                connect = False

        # mark unconditional successor, if reachable
        if connect and block.next is not None:
            data.succ.add(block.next)
        return data

    def global_transfer(self, block: GlobalBlock, entry: CodeBlock) -> BlockData:
        """Generate transfer functions for .text and .data sections"""
        data = BlockData(block)
        # global variables for this funciton
        self.text = tuple(instr.target for instr in block.text)
        self.globals = tuple(instr.target for instr in block.data)

        for instr in block.instructions():
            # update block GEN and KILL
            gen, kill = self.defs(instr)
            data.update(instr, gen, kill)

        data.succ.add(entry)
        return data

    # GEN and KILL generators for each instruction type
    _defs: dict[str, Callable[[Instruction], GenKill]]

    def defs(self, instr: Instruction) -> GenKill:
        """Get GEN and KILL for 'instr'"""
        get_defs = self._defs.get(instr.opname, None)
        if get_defs is not None:
            return get_defs(self, instr)
        else:
            return (), ()

    def __init_subclass__(cls, flow: Literal["forward", "backward"]) -> None:
        # choose forward or backward flow equations
        if flow == "forward":
            cls._equation = cls._forward_eq
            cls._build_instr_in_out = BlockData.build_forward_in_out
        else:
            cls._equation = cls._backward_eq
            cls._build_instr_in_out = BlockData.build_backward_in_out
        cls._build_instr_in_out = staticmethod(cls._build_instr_in_out)

        # find GEN and KILL generators
        cls._defs = {}
        for name, attr in cls.__dict__.items():
            if name.startswith("defs_"):
                _, op = name.split("_", maxsplit=1)
                cls._defs[op] = attr


# # # # # # # # # # #
# Reaching Analysis #


class ReachingDefinitions(DataFlowAnalysis, flow="forward"):
    """Find reachable definitions for each instruction and variable"""

    def defs_binary_op(self, instr: BinaryOpInstruction) -> GenKill:
        return (instr.target,), (instr.target,)

    defs_add = defs_sub = defs_binary_op
    defs_mul = defs_div = defs_mod = defs_binary_op
    defs_lt = defs_le = defs_eq = defs_binary_op
    defs_gt = defs_ge = defs_ne = defs_binary_op
    defs_and = defs_or = defs_binary_op

    def defs_unary_op(self, instr: UnaryOpInstruction) -> GenKill:
        return (instr.target,), (instr.target,)

    defs_not = defs_unary_op

    def defs_alloc(self, instr: AllocInstr) -> GenKill:
        return (instr.target,), (instr.target,)

    def defs_store(self, instr: StoreInstr) -> GenKill:
        if isinstance(instr.type, (PointerType, ArrayType)):
            return (instr.source, instr.target), (instr.source, instr.target)
        else:
            return (instr.target,), (instr.target,)

    def defs_load(self, instr: LoadInstr) -> GenKill:
        return (instr.target,), (instr.target,)

    def defs_global(self, instr: GlobalInstr) -> GenKill:
        return (instr.target,), (instr.target,)

    def defs_literal(self, instr: LiteralInstr) -> GenKill:
        return (instr.target,), (instr.target,)

    def defs_elem(self, instr: ElemInstr) -> GenKill:
        return (instr.target,), (instr.target,)

    def defs_get(self, instr: GetInstr) -> GenKill:
        return (instr.target, instr.source), (instr.target,)

    def defs_call(self, instr: CallInstr) -> GenKill:
        if instr.target is not None:
            ret = (instr.target,)
        else:
            ret = ()
        return self.globals + ret, ret

    def defs_param(self, instr: ParamInstr) -> GenKill:
        if isinstance(instr.type, (ArrayType, PointerType)):
            return (instr.source,), ()
        else:
            return (), ()

    def defs_read(self, instr: ReadInstr) -> GenKill:
        return (instr.source,), (instr.source,)


# # # # # # # # # # # # #
# Constant Propagation  #


class Const:
    """The semilattice for constant analysis"""

    __slots__ = ()

    @abstractmethod
    def __xor__(self, other: Const) -> Const:
        raise NotImplementedError


@unique
class ConstSpecial(Const, Enum):
    NAC = object()
    Undef = object()

    def __xor__(self, other: Const) -> Const:
        if self is NAC:
            return self
        else:
            return other

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Const[{self.name}]"


NAC = ConstSpecial.NAC
Undef = ConstSpecial.Undef


Value = Union[int, float, str, bool]


@dataclass(frozen=True)
class ConstValue(Const):

    value: Union[Value, list[Value], tuple[Const]]

    def __xor__(self, other: Const) -> Const:
        if not isinstance(other, ConstValue):
            return other ^ self
        elif self.value == other.value:
            return self
        else:
            return NAC

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"Const({self.value})"


Constants = Dict[Variable, Const]


class ConstantAnalysis:
    def __init__(self, rdefs: ReachingDefinitions) -> None:
        self.rdefs = rdefs
        self.values: dict[FlowBlock, dict[Instruction, Constants]] = {}

        for block in rdefs.data.keys():
            self.visit(block)

    def get(self, block: CodeBlock, instr: Instruction, var: Variable) -> Const:
        """Get constant value for variable after given instruction"""
        # visit, if needed
        self.visit(block)
        # if reaching here again, then variable cannot be constant
        value = self.values[block].get(instr, None)
        if value is None:
            return NAC  # TODO: is this right?
        # otherwise return current value
        return value.get(var, Undef)

    def get_const(self, defs: InOut, var: Variable) -> Union[ConstSpecial, ConstValue]:
        result = Undef
        for block, instr in defs.get(var, set()):
            result ^= self.get(block, instr, var)
        return result

    def visit(self, block: FlowBlock) -> None:
        """Analyse constants for a block"""
        if block in self.values:
            return

        self.values[block] = {}
        for instr in block.instructions():
            self.values[block][instr] = self.gen_const(block, instr)

    def gen_const(self, block: FlowBlock, instr: Instruction) -> Constants:  # TODO: a set?
        """Return the set of constants generated by this instruction"""
        generator = self._gen.get(instr.opname, None)
        if generator is not None:
            return generator(self, instr, self.rdefs.in_set((block, instr)))
        else:
            return {}

    def _gen_global(self, instr: GlobalInstr, _: InOut) -> Constants:
        if isinstance(instr.target, TextVariable) and instr.value is not None:
            return {instr.target: ConstValue(instr.value)}
        else:
            return {instr.target: NAC}

    def _gen_literal(self, instr: LiteralInstr, _: InOut) -> Constants:
        return {instr.target: ConstValue(instr.value)}

    def _gen_alloc(self, instr: AllocInstr, _: InOut) -> Constants:
        value = (Undef,)
        return {instr.target: ConstValue(value)}

    def _gen_load(self, instr: LoadInstr, defs: InOut) -> Constants:
        value = self.get_const(defs, instr.varname)
        if isinstance(value, ConstSpecial):
            return {instr.target: value}
        else:
            return {instr.target: value.value[0]}

    def _gen_store(self, instr: StoreInstr, defs: InOut) -> Constants:
        value = self.get_const(defs, instr.source)
        if isinstance(instr.type, ArrayType):
            return {instr.target: value}
        else:
            return {instr.target: ConstValue((value,))}

    def _generic_bin_op(
        self,
        instr: TempTargetInstruction,
        defs: InOut,
        left: str,
        right: str,
        op: Callable[[Value, Value], Value],
    ) -> Constants:
        left = self.get_const(defs, getattr(instr, left))
        right = self.get_const(defs, getattr(instr, right))

        if left is NAC or right is NAC:
            return {instr.target: NAC}
        elif left is Undef or right is Undef:
            return {instr.target: Undef}
        else:
            value = op(left.value, right.value)
            return {instr.target: ConstValue(value)}

    def _gen_elem(self, instr: ElemInstr, defs: InOut) -> Constants:
        return self._generic_bin_op(instr, defs, "source", "index", lambda s, i: s[i])

    def _gen_get(self, instr: GetInstr, defs: InOut) -> Constants:
        value = self.get_const(defs, instr.source)
        return {instr.source: ConstValue((NAC,)), instr.target: ConstValue((value,))}

    _binary_ops: dict[str, Callable[[Value, Value], Value]] = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "mod": lambda x, y: x % y,
        "le": lambda x, y: x < y,
        "lt": lambda x, y: x <= y,
        "ge": lambda x, y: x >= y,
        "gt": lambda x, y: x > y,
        "eq": lambda x, y: x == y,
        "ne": lambda x, y: x != y,
        "and": lambda x, y: x and y,
        "or": lambda x, y: x or y,
    }

    def _gen_binary_op(self, instr: BinaryOpInstruction, defs: InOut) -> Constants:
        return self._generic_bin_op(instr, defs, "left", "right", self._binary_ops[instr.opname])

    def _gen_div(self, instr: DivInstr, defs: InOut) -> Constants:
        if instr.type is FloatType:
            return self._generic_bin_op(instr, defs, "left", "right", lambda a, b: a / b)
        else:
            return self._generic_bin_op(instr, defs, "left", "right", lambda a, b: a // b)

    def _gen_not(self, instr: NotInstr, defs: InOut) -> Constants:
        value = self.get_const(defs, instr.expr)
        if isinstance(value, ConstSpecial):
            return {instr.target: value}
        elif instr.type is BoolType:
            return {instr.target: ConstValue(not value.value)}
        else:
            return {instr.target: ConstValue(~value.value)}

    def _gen_call(self, instr: CallInstr, _: InOut) -> Constants:
        glob: Constants = {var: NAC for var in self.rdefs.globals}
        if instr.target is not None:
            glob[instr.target] = NAC
        return glob

    def _gen_read(self, instr: ReadInstr, _: InOut) -> Constants:
        return {instr.source: ConstValue((NAC,))}

    _gen: dict[str, Callable[[ConstantAnalysis, Instruction, InOut], Constants]] = {
        "literal": _gen_literal,
        "alloc": _gen_alloc,
        "load": _gen_load,
        "store": _gen_store,
        "global": _gen_global,
        "elem": _gen_elem,
        "get": _gen_get,
        "add": _gen_binary_op,
        "sub": _gen_binary_op,
        "mul": _gen_binary_op,
        "div": _gen_div,
        "mod": _gen_binary_op,
        "not": _gen_not,
        "le": _gen_binary_op,
        "lt": _gen_binary_op,
        "gt": _gen_binary_op,
        "ge": _gen_binary_op,
        "eq": _gen_binary_op,
        "ne": _gen_binary_op,
        "and": _gen_binary_op,
        "or": _gen_binary_op,
        "call": _gen_call,
        "read": _gen_read,
    }


# # # # # # # # # # # # # #
# Data Flow Optimizations #


class DataFlow(NodeVisitor[None]):
    def __init__(self, viewcfg: bool):
        self.viewcfg = viewcfg

    def show(self, buf: TextIO = sys.stdout) -> None:
        for code in self.code:
            print(code.format(), file=buf)

    @property
    def code(self) -> list[Instruction]:
        """
        The generated code (can be mapped to a list of tuples)
        """
        if not hasattr(self, "_code"):
            bb = EmitBlocks()
            self._code = bb.visit(self.glob)
        return self._code

    # TODO: add analyses

    def visit_Program(self, node: Program) -> None:
        # First, save the global instructions on code member
        self.glob = node.cfg

        # then rebuild functions
        self.glob.functions = []
        for decl in node.gdecls:
            self.visit(decl)

        if self.viewcfg:
            dot = CFG()
            for function in self.glob.functions:
                dot.view(function)

    def constant_propagation(self, block: FunctionBlock) -> FunctionBlock:
        # # start with Reach Definitions Analysis
        # self.buildRD_blocks(decl.cfg)
        # self.computeRD_gen_kill()
        # self.computeRD_in_out()
        # # and do constant propagation optimization
        # self.constant_propagation()
        rdefs = ReachingDefinitions(block)
        ctan = ConstantAnalysis(rdefs)

        # TODO
        return block

    def dead_code_elimination(self, block: FunctionBlock) -> FunctionBlock:
        # # after do live variable analysis
        # self.buildLV_blocks(decl.cfg)
        # self.computeLV_use_def()
        # self.computeLV_in_out()
        # # and do dead code elimination
        # self.deadcode_elimination()
        # # finally save optimized instructions in self.code
        # self.appendOptimizedCode(decl.cfg)

        # TODO
        return block

    def visit_FuncDef(self, node: FuncDef) -> None:
        cfg = self.constant_propagation(node.cfg)
        cfg = self.dead_code_elimination(cfg)
        node.cfg = cfg
        self.glob.functions.append(cfg)


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script runs the interpreter on the optimized uCIR \
              and shows the speedup obtained from comparing original uCIR with its optimized version.",
        type=str,
    )
    parser.add_argument(
        "--opt",
        help="Print optimized uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--speedup",
        help="Show speedup from comparing original uCIR with its optimized version.",
        action="store_true",
        default=True,
    )
    parser.add_argument("--debug", help="Run interpreter in debug mode.", action="store_true")
    parser.add_argument(
        "-c",
        "--cfg",
        help="show the CFG of the optimized uCIR for each function in pdf format",
        action="store_true",
    )
    args = parser.parse_args()

    speedup = args.speedup
    print_opt_ir = args.opt
    create_cfg = args.cfg
    interpreter_debug = args.debug

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f)

    sema = Visitor()
    sema.visit(ast)

    gen = CodeGenerator(False)
    gen.visit(ast)
    gencode = gen.code

    opt = DataFlow(create_cfg)
    opt.visit(ast)
    optcode = opt.code
    if print_opt_ir:
        print("Optimized uCIR: --------")
        opt.show()
        print("------------------------\n")

    speedup = len(gencode) / len(optcode)
    sys.stderr.write(
        "[SPEEDUP] Default: %d Optimized: %d Speedup: %.2f\n\n"
        % (len(gencode), len(optcode), speedup)
    )

    vm = Interpreter(interpreter_debug)
    vm.run(optcode)
