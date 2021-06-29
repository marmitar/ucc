import argparse
import pathlib
import sys
from uc.uc_ast import FuncDef
from uc.uc_block import CFG, format_instruction
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor


class DataFlow(NodeVisitor):
    def __init__(self, viewcfg):
        # flag to show the optimized control flow graph
        self.viewcfg = viewcfg
        # list of code instructions after optimizations
        self.code = []
        # TODO

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    # TODO: add analyses

    def visit_Program(self, node):
        # First, save the global instructions on code member
        self.code = node.text[:]  # [:] to do a copy
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # start with Reach Definitions Analysis
                self.buildRD_blocks(_decl.cfg)
                self.computeRD_gen_kill()
                self.computeRD_in_out()
                # and do constant propagation optimization
                self.constant_propagation()

                # after do live variable analysis
                self.buildLV_blocks(_decl.cfg)
                self.computeLV_use_def()
                self.computeLV_in_out()
                # and do dead code elimination
                self.deadcode_elimination()

                # after that do cfg simplify (optional)
                self.short_circuit_jumps(_decl.cfg)
                self.merge_blocks(_decl.cfg)
                self.discard_unused_allocs(_decl.cfg)

                # finally save optimized instructions in self.code
                self.appendOptimizedCode(_decl.cfg)

        if self.viewcfg:
            for _decl in node.gdecls:
                if isinstance(_decl, FuncDef):
                    dot = CFG(_decl.decl.name.name + ".opt")
                    dot.view(_decl.cfg)


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
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
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
        ast = p.parse(f.read())

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
