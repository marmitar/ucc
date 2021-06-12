from __future__ import annotations
import argparse
import pathlib
import sys
from typing import TextIO
from uc.uc_ast import Program
from uc.uc_block import CFG, EmitBlocks
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_ir import Instruction
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor


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
        # for decl in node.gdecls:
        #     if isinstance(decl, FuncDef):
        #         # start with Reach Definitions Analysis
        #         self.buildRD_blocks(decl.cfg)
        #         self.computeRD_gen_kill()
        #         self.computeRD_in_out()
        #         # and do constant propagation optimization
        #         self.constant_propagation()

        #         # after do live variable analysis
        #         self.buildLV_blocks(decl.cfg)
        #         self.computeLV_use_def()
        #         self.computeLV_in_out()
        #         # and do dead code elimination
        #         self.deadcode_elimination()

        #         # finally save optimized instructions in self.code
        #         self.appendOptimizedCode(decl.cfg)

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
