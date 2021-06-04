from __future__ import annotations
import argparse
import pathlib
import sys
from typing import Any, Optional, TextIO
from uc.uc_ast import (
    ID,
    BinaryOp,
    BoolConstant,
    CharConstant,
    Constant,
    Decl,
    FloatConstant,
    FuncDef,
    IntConstant,
    Node,
    ParamList,
    Print,
    Program,
    StringConstant,
)
from uc.uc_block import (
    CFG,
    AllocInstr,
    BasicBlock,
    EmitBlocks,
    FunctionBlock,
    GlobalBlock,
    GlobalVariable,
    Instruction,
    LiteralInstr,
    LoadInstr,
    NamedVariable,
    StoreInstr,
    TempVariable,
    TextVariable,
    Variable,
)
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor


class CodeGenerator(NodeVisitor[Optional[Variable]]):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg: bool):
        self.viewcfg = viewcfg

        self.glob = GlobalBlock()
        self.current: Optional[BasicBlock] = None

        # TODO: Complete if needed.

    def show(self, buf: TextIO = sys.stdout) -> None:
        text = ""
        for code in self.code:
            text += code.format() + "\n"
        buf.write(text)

    @property
    def code(self) -> list[Instruction]:
        """
        The generated code (can be mapped to a list of tuples)
        """
        return EmitBlocks().visit(self.glob)

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

        if self.viewcfg:  # evaluate to True if -cfg flag is present in command line
            dot = CFG(node.name)
            dot.view(node)

    def _evaluate_init(self, node: Optional[Node]) -> Any:
        raise NotImplementedError("# TODO")

    def visit_Decl(self, node: Decl, source: Optional[TempVariable] = None) -> None:
        uctype = node.type.uc_type
        varname = self.visit_ID(node.name)
        # insert globals on '.data' section
        if isinstance(varname, GlobalVariable):
            self.glob.new_global(uctype, varname, self._evaluate_init(node.init))
            return
        # local variables are allocated on the function stack
        instr = AllocInstr(uctype.typename(), varname)
        self.current.append(instr)
        # if a value is given, initialize it
        if source is None and node.init is not None:
            source = self.visit(node.init)
        if source is not None:
            instr = StoreInstr(uctype.typename(), source, varname)
            self.current.append(instr)

    def visit_FuncDef(self, node: FuncDef) -> None:
        decl = node.declaration.type
        # create function block
        block = FunctionBlock(self.glob, decl.uc_type)
        # create entry block and populate it
        self.current = block.entry
        self.visit(decl.param_list)
        # visit body
        self.visit(node.decl_list)
        self.visit(node.implementation)
        # remove from block list
        self.current = None

    def visit_ParamList(self, node: ParamList) -> None:
        tempname = [var for _, _, var in self.current.function.params]
        for decl, varname in zip(node.params, tempname):
            self.visit_Decl(decl, varname)

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

    def visit_Constant(self, node: Constant) -> TempVariable:
        # Create a new temporary variable name
        target = self.current.new_temp()
        # Make the SSA opcode and append to list of generated instructions
        instr = LiteralInstr(node.uc_type.typename(), node.value, target)
        self.current.append(instr)
        return target

    def visit_IntConstant(self, node: IntConstant) -> TempVariable:
        return self.visit_Constant(node)

    def visit_FloatConstant(self, node: FloatConstant) -> TempVariable:
        return self.visit_Constant(node)

    def visit_BoolConstant(self, node: BoolConstant) -> TempVariable:
        return self.visit_Constant(node)

    def visit_CharConstant(self, node: CharConstant) -> TempVariable:
        return self.visit_Constant(node)

    def visit_StringConstant(self, node: StringConstant) -> TextVariable:
        return self.glob.new_literal("string", node.value)

    def visit_ID(self, node: ID) -> NamedVariable:
        if node.is_global:
            return GlobalVariable(node.name)
        else:
            return NamedVariable(node.name)


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
        ast = p.parse(f)

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
