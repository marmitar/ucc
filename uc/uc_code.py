from __future__ import annotations
import argparse
import pathlib
import sys
from typing import Any, Callable, Optional, TextIO, Type, Union
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
    GlobalDecl,
    InitList,
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
    DataVariable,
    DivInstr,
    ElemInstr,
    EqInstr,
    GeInstr,
    GetInstr,
    GlobalInstr,
    GlobalVariable,
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
    Variable,
)
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor
from uc.uc_type import ArrayType, IntType, PointerType, PrimaryType, VoidType, uCType


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

    def _extract_value(self, node: Node) -> Union[Any, list[Any]]:
        if isinstance(node, InitList):
            return [self._extract_value(val) for val in node.init]
        elif isinstance(node, Constant):
            return node.value
        elif isinstance(node, AddressOp) and node.op == "&":
            return self._varname(node.expr)
        else:
            raise ValueError()

    def visit_Decl(self, node: Decl) -> None:
        self.visit(node.type, node.init)

    def visit_ArrayDecl(self, node: ArrayDecl, init: Optional[Node]) -> None:
        varname = self._varname(node.declname)
        data = self.current.new_temp()
        # alloc space for data and pointer
        self.current.append(
            AllocInstr(node.uc_type.as_pointer(), varname),
            AllocInstr(node.uc_type, data),
            StoreInstr(node.uc_type, data, varname),
        )
        if init is not None:
            value = self.visit(init)
            size = self.current.new_literal(sizeof(node))
            # copy initialization data
            self.current.append(
                ParamInstr(PointerType(VoidType), value),
                ParamInstr(PointerType(VoidType), data),
                ParamInstr(IntType, size),
                CallInstr(VoidType, self.glob.memcpy),
            )

    def visit_VarDecl(self, node: VarDecl, init: Optional[Node]) -> None:
        varname = self._varname(node.declname)
        self.current.append(AllocInstr(node.uc_type, varname))

        if init is not None:
            value = self.visit(init)
            self.current.append(StoreInstr(node.uc_type, value, varname))

    visit_PointerDecl = visit_VarDecl

    def visit_GlobalDecl(self, node: GlobalDecl) -> None:
        for decl in node.decls:
            varname = self._varname(decl.name)
            if decl.init is not None:
                value = self._extract_value(decl.init)
            else:
                value = None
            self.glob.new_global(decl.name.uc_type, varname, value)

    def visit_FuncDef(self, node: FuncDef) -> None:
        decl = node.declaration.type
        # create function block
        block = self.glob.new_function(decl.uc_type)
        self.current = block.entry
        # populate entry
        self.visit(decl.param_list)
        self.visit(node.decl_list)
        # visit body
        self.visit(node.implementation)
        # remove from block list
        self.current = None

    def visit_ParamList(self, node: ParamList) -> None:
        for decl, (_, _, tempvar) in zip(node.params, self.current.function.params):
            varname = self._varname(decl.name)
            # use arrays as pointer
            if isinstance(decl.type, ArrayDecl):
                uctype = decl.type.uc_type.as_pointer()
            else:
                uctype = decl.type

            self.current.append(
                AllocInstr(uctype, varname),
                StoreInstr(uctype, tempvar, varname),
            )

    def visit_InitList(self, node: InitList) -> TextVariable:
        data = self._extract_value(node)
        return self.glob.new_text(node.uc_type, data)

    # # # # # # # #
    # STATEMENTS  #

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
                # call puts function
                pointer = self.visit(param)
                size = self.current.new_literal(sizeof(param))
                self.current.append(
                    ParamInstr(param.uc_type, pointer),
                    ParamInstr(IntType, size),
                    CallInstr(VoidType, self.glob.puts),
                )

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

    def visit_Assignment(self, node: Assignment) -> Variable:
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
        zero = self.current.new_literal(0)
        out = self.current.new_temp()
        # negation is: 0 - value
        self.current.append(SubInstr(uctype, zero, source, out))
        return out

    def _unary_not(self, uctype: uCType, node: Node) -> TempVariable:
        source = self.visit(node)
        out = self.current.new_temp()
        self.current.append(NotInstr(uctype, source, out))
        return out

    def _unary_star(self, uctype: uCType, node: Node, ref: bool = False) -> Variable:
        # get reference to inner value
        source = self.visit(node)
        if ref:
            return source
        # and get value using reference
        target = self.current.new_temp()
        self.current.append(LoadInstr(uctype, source, target))
        return target

    def _unary_ref(self, uctype: uCType, node: Node, ref: bool = False) -> Variable:
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
        offset = self.visit(node.index)
        # calculate offset for first index
        if sizeof(node) != 1:
            size = self.current.new_literal(sizeof(node))
            self.current.append(MulInstr(IntType, offset, size, offset))
        else:
            size = None
        array = node.array
        # and all other indexes
        while isinstance(array, ArrayRef):
            index = self.visit(array.index)
            if size is None:
                size = self.current.new_temp()
            self.current.append(
                LiteralInstr(IntType, sizeof(array), size),
                MulInstr(IntType, index, size, index),
                AddInstr(IntType, index, offset, offset),
            )

        pointer = self.visit(array)
        value = self.current.new_temp()
        # return reference for compound types
        if ref or isinstance(node.uc_type, ArrayType):
            instr = AddInstr(node.uc_type, pointer, offset, value)
        # and value for primaries
        else:
            instr = ElemInstr(node.uc_type, pointer, offset, value)
        self.current.append(instr)
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
            instr = CallInstr(node.uc_type, source)
        else:
            target = self.current.new_temp()
            instr = CallInstr(node.uc_type, source, target)
        self.current.append(instr)
        return target

    # # # # # # # # #
    # BASIC SYMBOLS #

    def visit_Constant(self, node: Constant) -> TempVariable:
        return self.current.new_literal(node.value, node.uc_type)

    visit_IntConstant = visit_Constant
    visit_FloatConstant = visit_Constant
    visit_CharConstant = visit_Constant
    visit_BoolConstant = visit_Constant

    def visit_StringConstant(self, node: StringConstant) -> TextVariable:
        return self.glob.new_text(node.uc_type, node.value)

    def _varname(self, ident: ID) -> MemoryVariable:
        """Get variable name for identifier"""
        if ident.is_global:
            return DataVariable(ident.name)
        else:
            return NamedVariable(ident.name)

    def visit_ID(self, node: ID, ref: bool = False) -> Variable:
        stackvar = self._varname(node)
        # load address
        if ref or (node.is_global and isinstance(node.uc_type, ArrayType)):
            return stackvar
        # or value
        register = self.current.new_temp()
        self.current.append(LoadInstr(node.uc_type, stackvar, register))
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
