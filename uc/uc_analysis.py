from __future__ import annotations
import argparse
import pathlib
import sys
from typing import Callable, Dict, Literal, Set, TextIO, Tuple
from uc.uc_ast import Program
from uc.uc_block import CFG, CodeBlock, EmitBlocks, FunctionBlock
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_ir import (
    AllocInstr,
    BinaryOpInstruction,
    CallInstr,
    CBranchInstr,
    ElemInstr,
    ExitInstr,
    GetInstr,
    Instruction,
    JumpInstr,
    LiteralInstr,
    LoadInstr,
    ParamInstr,
    PrintInstr,
    ReadInstr,
    ReturnInstr,
    StoreInstr,
    UnaryOpInstruction,
    Variable,
)
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor
from uc.uc_type import ArrayType, PointerType

VarData = Dict[Variable, Tuple[Instruction, ...]]
InOut = Dict[Variable, Set[Tuple[CodeBlock, Instruction]]]
GenKill = Tuple[Tuple[Variable, ...], Tuple[Variable, ...]]


class BlockData:
    """Info for a basic block in data flow analysis"""

    def __init__(self, block: CodeBlock) -> None:
        self.block = block
        # successors for this block
        self.succ: set[CodeBlock] = set()
        # and predecessors
        self.pred: set[CodeBlock] = set()

        # input in data flow equations
        self.inp: InOut = {}
        # and output
        self.out: InOut = {}

        # GEN for all instructions in this block
        self.gen: VarData = {}
        # KILL entire block
        self.kill: VarData = {}

        # GEN for each instruction in this block
        self.igen: dict[Instruction, tuple[Variable, ...]] = {}
        # KILL for each instruction
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

    def apply(self, data: InOut) -> InOut:
        """Apply GEN and KILL for input/output sets"""

        new: InOut = {}
        for var, instr in data.items():
            if var not in self.kill:
                new[var] = instr
        for var, instr in self.gen.items():
            current = new.get(var, set())
            new[var] = current.union((self.block, i) for i in instr)
        return new


class DataFlowAnalysis:
    """ABC for data flow analysis (forward and backward)"""

    def __init__(self, function: FunctionBlock) -> None:
        self.function = function
        # global variables for this funciton
        self.globals = tuple(instr.target for instr in function.program.instructions())

        # mapping for labels to blocks
        self.blocks = {block.label: block for block in function.all_blocks()}
        # data flow info
        self.data = {block: self.analyze_block(block) for block in function.all_blocks()}

        # build predecessors, IN and OUT
        self.update_preds()
        self.build_in_out()

    def analyze_block(self, block: CodeBlock) -> BlockData:
        """Generate basic info for a given block"""
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
        if not connect and block.next is not None:
            data.succ.add(block.next)
        return data

    def update_preds(self) -> None:
        """Set predecessors for all block (run only once)"""
        for tail in self.function.all_blocks():
            for head in self.data[tail].succ:
                self.data[head].pred.add(tail)

    def forward_eq(self, block: CodeBlock) -> bool:
        """Data flow equations for forward analysis"""
        changed = False

        # generate IN as union(OUT[b] for pred b)
        inp: InOut = {}
        for next in self.data[block].pred:
            for var, dat in self.data[next].out.items():
                instr = inp.get(var, set())
                inp[var] = instr | dat
        # mark if changed
        changed |= inp != self.data[block].inp
        self.data[block].inp = inp

        # generate OUT as fB(IN) and mark if changed
        out = self.data[block].apply(inp)
        changed |= out != self.data[block].out
        self.data[block].out = out

        return changed

    def backward_eq(self, block: CodeBlock) -> bool:
        """Data flow equations for backward analysis"""
        changed = False

        # generate OUT as union(IN[b] for succ b)
        out: InOut = {}
        for next in self.data[block].succ:
            for var, dat in self.data[next].inp.items():
                instr = out.get(var, set())
                out[var] = instr | dat
        # mark if changed
        changed |= out != self.data[block].out
        self.data[block].out = out

        # generate IN as fB(OUT) and mark if changed
        inp = self.data[block].apply(out)
        changed |= inp != self.data[block].inp
        self.data[block].inp = inp

        return changed

    def equation(self, _: CodeBlock) -> bool:
        """Data flow equations (forward and backward)"""
        raise NotImplementedError

    def build_in_out(self) -> None:
        """Build IN and OUT sets"""
        changed = True
        # until stable
        while changed:
            changed = False

            for block in self.function.all_blocks():
                # apply for/backward equations
                changed |= self.equation(block)

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
        cls.equation = cls.forward_eq if flow == "forward" else cls.backward_eq

        # find GEN and KILL generators
        cls._defs = {}
        for name, attr in cls.__dict__.items():
            if name.startswith("defs_"):
                op = name.split("_")[1]
                cls._defs[op] = attr


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
        # for function in self.glob.functions:
        #     # start with Reach Definitions Analysis
        #     self.buildRD_blocks(decl.cfg)
        #     self.computeRD_gen_kill()
        #     self.computeRD_in_out()
        #     # and do constant propagation optimization
        #     self.constant_propagation()
        #     # after do live variable analysis
        #     self.buildLV_blocks(decl.cfg)
        #     self.computeLV_use_def()
        #     self.computeLV_in_out()
        #     # and do dead code elimination
        #     self.deadcode_elimination()

        #     # finally save optimized instructions in self.code
        #     self.appendOptimizedCode(decl.cfg)

        if self.viewcfg:
            dot = CFG()
            for function in self.glob.functions:
                dot.view(function)


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
