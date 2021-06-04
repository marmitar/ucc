import argparse
import pathlib
import sys
from typing import Optional, TextIO
from uc.uc_ast import (
    BinaryOp,
    Constant,
    FuncDef,
    Print,
    Program,
    StringConstant,
    VarDecl,
)
from uc.uc_block import (
    CFG,
    BasicBlock,
    Block,
    ConditionBlock,
    EmitBlocks,
    FunctionBlock,
    GlobalBlock,
    GlobalInstr,
    Instruction,
)
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor


class CodeGenerator(NodeVisitor[None]):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg: bool):
        super().__init__(None)
        self.viewcfg = viewcfg

        self.glob = GlobalBlock()
        self._function: Optional[FunctionBlock] = None
        self._block: Optional[BasicBlock] = None

        # The generated code (list of tuples)
        # At the end of visit_program, we call each function definition to emit
        # the instructions inside basic blocks. The global instructions that
        # are stored in self.text are appended at beginning of the code
        self.code: list[Instruction] = []

        # TODO: Complete if needed.

    def show(self, buf: TextIO = sys.stdout) -> None:
        text = ""
        for code in self.code:
            text += code.format() + "\n"
        buf.write(text)

    @property
    def current_function(self) -> FunctionBlock:
        if self._function is None:
            raise ValueError()
        return self._function

    @property
    def current_block(self) -> BasicBlock:
        if self._block is None:
            raise ValueError()
        return self._block

    # You must implement visit_Nodename methods for all of the other
    # AST nodes.  In your code, you will need to make instructions
    # and append them to the current block code list.
    #
    # A few sample methods follow. Do not hesitate to complete or change
    # them if needed.

    # # # # # # # # #
    # DECLARATIONS  #

    def visit_Program(self, node: Program) -> None:
        # Visit all of the global declarations
        for decl in node.gdecls:
            self.visit(decl)
        # At the end of codegen, first init the self.code with
        # the list of global instructions allocated in self.text
        self.code = self.text.copy()
        # Also, copy the global instructions into the Program node
        node.text = self.text.copy()
        # After, visit all the function definitions and emit the
        # code stored inside basic blocks.
        for decl in node.gdecls:
            if isinstance(decl, FuncDef):
                # _decl.cfg contains the Control Flow Graph for the function
                # cfg points to start basic block
                bb = EmitBlocks()
                bb.visit(decl.cfg)
                for _code in bb.code:
                    self.code.append(_code)

        if self.viewcfg:  # evaluate to True if -cfg flag is present in command line
            for decl in node.gdecls:
                if isinstance(decl, FuncDef):
                    dot = CFG(decl.decl.name.name)
                    dot.view(decl.cfg)  # _decl.cfg contains the CFG for the function

    def visit_VarDecl(self, node: VarDecl) -> None:
        # Allocate on stack memory
        varname = f"%{node.declname.name}"
        inst = (f"alloc_{node.type.name}", varname)
        self.current_block.append(inst)

        # Store optional init val
        init = node.decl.init
        if init is not None:
            self.visit(init)
            inst = (
                f"store_{node.type.name}",
                init.gen_location,
                node.declname.gen_location,
            )
            self.current_block.append(inst)

    # # # # # # # #
    # STATEMENTS  #

    def visit_Print(self, node: Print) -> None:
        # Visit the expression
        self.visit(node.param)

        # TODO: Load the location containing the expression

        # Create the opcode and append to list
        inst = (f"print_{node.param.uc_type!r}", node.param.gen_location)
        self.current_block.append(inst)

        # TODO: Handle the cases when node.expr is None or ExprList

    # # # # # # # #
    # EXPRESSIONS #

    def visit_BinaryOp(self, node: BinaryOp) -> None:
        # Visit the left and right expressions
        self.visit(node.left)
        self.visit(node.right)

        # TODO:
        # - Load the location containing the left expression
        # - Load the location containing the right expression

        # Make a new temporary for storing the result
        target = self.new_temp()

        # Create the opcode and append to list
        opcode = ""  # TODO: binary_ops[node.op] + "_" + node.left.type.name
        inst = (opcode, node.left.gen_location, node.right.gen_location, target)
        self.current_block.append(inst)

        # Store location of the result on the node
        node.gen_location = target

    # # # # # # # # #
    # BASIC SYMBOLS #

    def visit_Constant(self, node: Constant) -> None:
        if node.rawtype == "string":
            target = self.new_text("str")
            inst = ("global_string", target, node.value)
            self.text.append(inst)
        else:
            # Create a new temporary variable name
            target = self.new_temp()
            # Make the SSA opcode and append to list of generated instructions
            inst = (f"literal_{node.uc_type!r}", node.value, target)
            self.current_block.append(inst)
        # Save the name of the temporary variable where the value was placed
        node.gen_location = target

    def visit_StringConstant(self, node: StringConstant) -> None:
        target = self.new_text("str")
        # inst = GlobalInstr("string", )
        # self.text.append(inst)

    # TODO: Complete.


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script only runs the interpreter on the uCIR. \
              Use the other options for printing the uCIR, generating the CFG or for the debug mode.",
        type=str,
    )
    parser.add_argument(
        "--ir",
        help="Print uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument("--cfg", help="Show the cfg of the input_file.", action="store_true")
    parser.add_argument("--debug", help="Run interpreter in debug mode.", action="store_true")
    args = parser.parse_args()

    print_ir: bool = args.ir
    create_cfg: bool = args.cfg
    interpreter_debug: bool = args.debug

    # get input path
    input_path = pathlib.Path(args.input_file)

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

    gen = CodeGenerator(create_cfg)
    gen.visit(ast)
    gencode = gen.code

    if print_ir:
        print("Generated uCIR: --------")
        gen.show()
        print("------------------------\n")

    vm = Interpreter(interpreter_debug)
    vm.run(gencode)
