from __future__ import annotations
import argparse
import pathlib
import sys
from typing import Any, Callable, Literal, Optional, TextIO, Tuple, Type, Union
from uc.uc_ast import (
    ID,
    AddressOp,
    ArrayDecl,
    ArrayRef,
    Assignment,
    BinaryOp,
    Constant,
    Decl,
    ExprList,
    FuncCall,
    FuncDef,
    IntConstant,
    Node,
    ParamList,
    Print,
    Program,
    Return,
    StringConstant,
    UnaryOp,
    VarDecl,
    sizeof,
)
from uc.uc_block import (
    CFG,
    BasicBlock,
    EmitBlocks,
    FunctionBlock,
    GlobalBlock,
    PutsBlock,
)
from uc.uc_interpreter import Interpreter
from uc.uc_ir import (
    AddInstr,
    AllocInstr,
    AndInstr,
    BinaryOpInstruction,
    CallInstr,
    CopyInstr,
    DataVariable,
    DivInstr,
    ElemInstr,
    EqInstr,
    GeInstr,
    GetInstr,
    GtInstr,
    Instruction,
    LeInstr,
    LiteralInstr,
    LoadInstr,
    LtInstr,
    MemoryVariable,
    ModInstr,
    MulInstr,
    NamedVariable,
    NeInstr,
    NotInstr,
    OrInstr,
    ParamInstr,
    PrintInstr,
    ReturnInstr,
    StoreInstr,
    SubInstr,
    TempVariable,
    TextVariable,
    UnaryOpInstruction,
)
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor
from uc.uc_type import IntType, PrimaryType, VoidType, uCType


class CodeGenerator(NodeVisitor[Optional[TempVariable]]):
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

    def visit_Decl(self, node: Decl) -> None:
        self.visit(node.type, node.init)

    def visit_ArrayDecl(self, node: ArrayDecl, init: Optional[Node]) -> None:
        varname = self._varname(node.declname)
        data = NamedVariable(f".{node.declname.name}.content")
        pointer = self.current.new_temp()

        pointer_ty = node.uc_type.as_pointer()
        self.current.append(
            # space for pointer / reference
            AllocInstr(pointer_ty, varname),
            # space for data
            AllocInstr(node.uc_type, data),
            # store pointer in variable
            GetInstr(pointer_ty, data, pointer),
            StoreInstr(pointer_ty, pointer, varname),
        )
        # copy initialization data
        if init is not None:
            value = self.visit(init)
            self.current.append(CopyInstr(node.uc_type, value, pointer))

    def visit_VarDecl(self, node: VarDecl, init: Optional[Node]) -> None:
        varname = self._varname(node.declname)
        self.current.append(AllocInstr(node.uc_type, varname))

        if init is not None:
            loc = self.visit_ID(node.declname, ref=True)
            value = self.visit(init)
            self.current.append(StoreInstr(node.uc_type, value, loc))

    visit_PointerDecl = visit_VarDecl

    def visit_FuncDef(self, node: FuncDef) -> None:
        decl = node.declaration.type
        # create function block
        block = FunctionBlock(self.glob, decl.uc_type)
        # create entry block and populate it
        self.current = block.entry
        self.visit(decl.param_list)
        self.visit(node.decl_list)
        # visit body
        self.visit(node.implementation)
        # remove from block list
        self.current = None

    def visit_ParamList(self, node: ParamList) -> None:
        for decl, (_, _, tempvar) in zip(node.params, self.current.function.params):
            varname = self._varname(decl.name)
            self.current.append(
                AllocInstr(decl.type.uc_type, varname),
                StoreInstr(decl.type.uc_type, tempvar, varname),
            )

    # def visit_InitList(self, node: InitList) -> TempVariable:

    # # # # # # # #
    # STATEMENTS  #

    def _print_string(self, node: Node) -> None:
        # create 'puts' function
        if not isinstance(self.glob.functions[0], PutsBlock):
            PutsBlock(self.glob)
        puts = self.glob.functions[0]

        # and call it
        pointer = self.visit(node)
        size = self._new_constant(IntType, sizeof(node))
        self.current.append(
            ParamInstr(node.uc_type, pointer),
            ParamInstr(IntType, size),
            CallInstr(VoidType, puts.label),
        )

    def visit_Print(self, node: Print) -> None:
        # empty print, terminate line
        if node.param is None:
            instr = PrintInstr()
            self.current.append(instr)
            return
        # show data
        for param in node.param.expr:
            if isinstance(param.uc_type, PrimaryType):
                value = self.visit(param)
                self.current.append(PrintInstr(param.uc_type, value))
            else:
                self._print_string(param)

    def visit_Return(self, node: Return) -> None:
        if node.result is not None:
            result = self.visit(node.result)
            instr = ReturnInstr(node.result.uc_type, result)
        else:
            instr = ReturnInstr(VoidType)
        self.current.append(instr)

    def visit_ExprList(self, node: ExprList) -> TempVariable:
        for expr in node.expr:
            result = self.visit(expr)
        return result

    # # # # # # # #
    # EXPRESSIONS #

    def visit_Assignment(self, node: Assignment) -> TempVariable:
        value = self.visit(node.right)
        target = self.visit(node.left, ref=True)
        self.current.append(StoreInstr(node.uc_type, value, target))
        return value

    # instructions for basic operations
    binary_op: dict[str, Type[BinaryOpInstruction]] = {
        "+": AddInstr,
        "-": SubInstr,
        "*": MulInstr,
        "/": DivInstr,
        "%": ModInstr,
        "<": LtInstr,
        "<=": LeInstr,
        ">": GtInstr,
        ">=": GeInstr,
        "==": EqInstr,
        "!=": NeInstr,
        "&&": AndInstr,
        "||": OrInstr,
    }

    def visit_BinaryOp(self, node: BinaryOp) -> TempVariable:
        # Visit the left and right expressions
        left = self.visit(node.left)
        right = self.visit(node.right)
        # Make a new temporary for storing the result
        target = self.current.new_temp()

        # Create the opcode and append to list
        self.current.append(self.binary_op[node.op](node.uc_type, left, right, target))
        return target

    visit_RelationOp = visit_BinaryOp

    def _unary_plus(self, uctype: uCType, node: Node) -> TempVariable:
        return self.visit(node)

    # implementations for unary operators
    def _unary_minus(self, uctype: uCType, node: Node) -> TempVariable:
        source = self.visit(node)
        zero = self._new_constant(uctype, 0)
        out = self.current.new_temp()
        # negation is: 0 - value
        self.current.append(SubInstr(uctype, zero, source, out))
        return out

    def _unary_not(self, uctype: uCType, node: Node) -> TempVariable:
        source = self.visit(node)
        out = self.current.new_temp()
        self.current.append(NotInstr(uctype, source, out))
        return out

    def _unary_star(self, uctype: uCType, node: Node) -> TempVariable:
        source = self.visit(node)
        index = self._new_constant(IntType, 0)
        target = self.current.new_temp()
        self.current.append(ElemInstr(uctype, source, index, target))
        return target

    def _unary_star(self, uctype: uCType, node: Node, ref: bool = False) -> TempVariable:
        # get reference to inner value
        source = self.visit(node)
        if ref:
            return source
        # and get value using reference
        index = self._new_constant(IntType, 0)
        target = self.current.new_temp()
        self.current.append(ElemInstr(uctype, source, index, target))
        return target

    def _unary_ref(self, uctype: uCType, node: Node, ref: bool = False) -> TempVariable:
        return self.visit(node, ref=True)

    unary_op: dict[str, Callable[[CodeGenerator, uCType, Node], TempVariable]] = {
        "!": _unary_not,
        "-": _unary_minus,
        "+": _unary_plus,
        "&": _unary_ref,
        "*": _unary_star,
    }

    def visit_UnaryOp(self, node: UnaryOp) -> TempVariable:
        return self.unary_op[node.op](self, node.uc_type, node.expr)

    def visit_AddressOp(self, node: AddressOp, ref: bool = False) -> TempVariable:
        return self.unary_op[node.op](self, node.uc_type, node.expr, ref=ref)

    def visit_ArrayRef(self, node: ArrayRef, ref: bool = False) -> TempVariable:
        # TODO
        source = self.visit(node.array)
        index = self.visit(node.index)
        # calculate offset, if needed
        if sizeof(node) != 1:
            offset = self.current.new_temp()
            size = self._new_constant(IntType, sizeof(node))
            self.current.append(MulInstr(IntType, size, index, offset))
        else:
            offset = index
        # return reference for compound types
        if ref or not isinstance(node.uc_type, PrimaryType):
            address = self.current.new_temp()
            self.current.append(AddInstr(node.uc_type, source, offset, address))
            return address
        # and value for primaries
        else:
            value = self.current.new_temp()
            self.current.append(ElemInstr(node.uc_type, source, offset, value))
            return value

    def visit_FuncCall(self, node: FuncCall) -> Optional[TempVariable]:
        # get function address
        source = self._varname(node.callable)
        # load parameters
        for param in node.parameters():
            varname = self.visit(param)
            self.current.append(ParamInstr(param.uc_type, varname))
        # then call the function
        if node.uc_type is VoidType:
            target = None
        else:
            target = self.current.new_temp()
        self.current.append(CallInstr(node.uc_type, source, target))
        return target

    # # # # # # # # #
    # BASIC SYMBOLS #

    def _new_constant(self, uctype: PrimaryType, value: Any) -> TempVariable:
        # Create a new temporary variable name
        target = self.current.new_temp()
        # Make the SSA opcode and append to list of generated instructions
        self.current.append(LiteralInstr(uctype, value, target))
        return target

    def visit_Constant(self, node: Constant) -> TempVariable:
        return self._new_constant(node.uc_type, node.value)

    visit_IntConstant = visit_Constant
    visit_FloatConstant = visit_Constant
    visit_CharConstant = visit_Constant
    visit_BoolConstant = visit_Constant

    def visit_StringConstant(self, node: StringConstant) -> TextVariable:  # TODO
        literal = self.glob.new_literal(node.uc_type, node.value)
        pointer = self.current.new_temp()
        self.current.append(GetInstr(node.uc_type, literal, pointer))
        return pointer

    def _varname(self, ident: ID) -> MemoryVariable:
        """Get variable name for identifier"""
        if ident.is_global:
            return DataVariable(ident.name)
        else:
            return NamedVariable(ident.name)

    def visit_ID(self, node: ID, ref: bool = False) -> TempVariable:
        stackvar = self._varname(node)
        register = self.current.new_temp()
        # load value into a register
        if not ref:
            instr = LoadInstr(node.uc_type, stackvar, register)
        # or load address
        else:
            instr = GetInstr(node.uc_type, stackvar, register)
        self.current.append(instr)
        return register


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
