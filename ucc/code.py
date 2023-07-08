from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Callable, Optional, TextIO, Union

from .ast import (
    ID,
    AddressOp,
    ArrayDecl,
    ArrayRef,
    Assert,
    Assignment,
    BinaryOp,
    Break,
    Constant,
    Decl,
    ExprList,
    FuncCall,
    FuncDecl,
    FuncDef,
    GlobalDecl,
    If,
    InitList,
    IterationStmt,
    Node,
    ParamList,
    Print,
    Program,
    Return,
    StringConstant,
    UnaryOp,
    VarDecl,
)
from .block import CFG, BasicBlock, BranchBlock, CodeList, EmitBlocks, GlobalBlock
from .interpreter import Interpreter
from .ir import (
    AddInstr,
    AndInstr,
    BinaryOpInstruction,
    CallInstr,
    DataVariable,
    DivInstr,
    ElemInstr,
    EqInstr,
    ExitInstr,
    GeInstr,
    GtInstr,
    LeInstr,
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
from .parser import UCParser
from .sema import NodeVisitor, Visitor
from .type import (
    ArrayType,
    BoolType,
    CharType,
    FunctionType,
    IntType,
    StringType,
    VoidType,
    uCType,
)


class CodeGenerator(NodeVisitor[Optional[Variable]]):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg: bool = False):
        self.viewcfg = viewcfg

        self.glob: GlobalBlock = None
        self.current: Optional[BasicBlock] = None

    def show(self, buf: TextIO = sys.stdout) -> None:
        for code in self.code:
            print(code.format(), file=buf)

    @property
    def code(self) -> CodeList:
        """
        The generated code (can be mapped to a list of tuples)
        """
        if not hasattr(self, "_code"):
            bb = EmitBlocks()
            self._code = bb.visit(self.glob)
        return self._code

    # # # # # # # # #
    # DECLARATIONS  #

    def visit_Program(self, node: Program) -> None:
        self.glob = GlobalBlock(node)
        # Visit all of the global declarations
        for decl in node.gdecls:
            self.visit(decl)
        # define start point
        if isinstance(node.uc_type, FunctionType):
            self.glob.add_start(node.uc_type)

        if self.viewcfg:  # evaluate to True if -cfg flag is present in command line
            dot = CFG()
            for function in self.glob.functions:
                dot.view(function)

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

    def visit_VarDecl(self, node: VarDecl, init: Optional[Node]) -> None:
        varname = self.current.alloc(node.uc_type, node.declname)

        if init is not None:
            value = self.visit(init)
            self.current.append_instr(StoreInstr(node.uc_type, value, varname))

    visit_PointerDecl = visit_VarDecl
    visit_ArrayDecl = visit_VarDecl

    def visit_GlobalDecl(self, node: GlobalDecl) -> None:
        for decl in node.decls:
            if isinstance(decl.type.uc_type, FunctionType):
                continue
            varname = self._varname(decl.name)
            if decl.init is not None:
                value = self._extract_value(decl.init)
            else:
                value = None
            self.glob.new_global(decl.name.uc_type, varname, value)

    def visit_FuncDecl(self, node: FuncDecl) -> None:
        pass

    def visit_FuncDef(self, node: FuncDef) -> None:
        decl = node.declaration.type
        # create function block
        self.cfg = block = self.glob.new_function(node)
        self.current = block.entry.next
        # populate entry and build body
        self.visit(decl.param_list)
        self.visit(node.implementation)
        # end block list
        self.current = None

    def visit_ParamList(self, node: ParamList) -> None:
        for decl, (_, _, tempvar) in zip(node.params, self.current.function.params):
            # use arrays as pointer
            if isinstance(decl.type, ArrayDecl):
                uctype = decl.type.uc_type.as_pointer()
            else:
                uctype = decl.type.uc_type

            varname = self.current.alloc(uctype, decl.name)
            self.current.append_instr(
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
            self.current.append_instr(PrintInstr())
            return
        # show data
        for param in node.param.expr:
            value = self.visit(param)
            self.current.append_instr(PrintInstr(param.uc_type, value))

    def visit_If(self, node: If) -> None:
        cond_block = self.current.insert_new(BranchBlock)
        # evaluate condition (might have side effects)
        self.current = cond_block
        condition = self.visit(node.condition)
        # generate end block
        end_block = self.current.insert_new(BasicBlock)
        # evaluate true case
        if node.true_stmt is not None:
            true_target = self.current.insert_new(BasicBlock, f"{cond_block.name}.true")
            self.current = true_target
            self.visit(node.true_stmt)
            self.current.jump_to(end_block)
        else:
            true_target = end_block
        # and false case
        if node.false_stmt is not None:
            false_target = self.current.insert_new(BasicBlock, f"{cond_block.name}.false")
            self.current = false_target
            self.visit(node.false_stmt)
            self.current.jump_to(end_block)
        else:
            false_target = end_block
        # then build branch instruction
        cond_block.branch(condition, true_target, false_target)
        self.current = end_block

    def visit_Return(self, node: Return) -> None:
        if node.result is not None:
            result = self.visit(node.result)
            instr = ReturnInstr(node.result.uc_type, result)
        else:
            instr = ReturnInstr(VoidType)
        self.current.append_instr(instr)

    def visit_ExprList(self, node: ExprList) -> Variable:
        for expr in node.expr:
            result = self.visit(expr)
        return result

    def visit_IterationStmt(self, node: IterationStmt) -> None:
        # declare variables
        if node.declaration is not None:
            self.visit(node.declaration)
        # initialize loop block and end block
        cond_block = self.current.insert_new(BranchBlock)
        loop_body = cond_block.insert_new(BasicBlock, f"{cond_block.name}.body")
        node.end_block = loop_body.insert_new(BasicBlock)

        # test condition
        self.current = cond_block
        if node.condition is not None:
            condition = self.visit(node.condition)
        else:
            condition = self.current.new_literal(True, BoolType)
        cond_block.branch(condition, loop_body, node.end_block)
        # run body
        self.current = loop_body
        if node.body is not None:
            self.visit(node.body)
        # update vars
        if node.update is not None:
            self.visit(node.update)
        self.current.jump_to(cond_block)

        # analyze remaining nodes
        self.current = node.end_block

    visit_For = visit_IterationStmt
    visit_While = visit_IterationStmt

    def visit_Break(self, node: Break) -> None:
        self.current.jump_to(node.iteration.end_block)

    def visit_Assert(self, node: Assert) -> None:
        assert_test = self.current.insert_new(BranchBlock)
        assert_fail = assert_test.insert_new(BasicBlock, f"{assert_test.name}.fail")
        next_block = assert_fail.insert_new(BasicBlock)
        # if condition is true, jump to next block
        self.current = assert_test
        condition = self.visit(node.param)
        assert_test.branch(condition, next_block, assert_fail)
        # else, show fail message
        msg = "assertion_fail on \0".encode("utf8")
        msg_type = StringType(len(msg))
        message = self.glob.new_text(msg_type, msg)
        assert_fail.append_instr(PrintInstr(msg_type, message))
        # and coordinates
        coord = node.param.coord or node.coord
        line = assert_fail.new_literal(coord.line)
        assert_fail.append_instr(PrintInstr(IntType, line))
        sep = assert_fail.new_literal(":", CharType)
        assert_fail.append_instr(PrintInstr(CharType, sep))
        column = assert_fail.new_literal(coord.column)
        assert_fail.append_instr(PrintInstr(IntType, column))
        # then, exit
        zero = assert_fail.new_literal(0)
        assert_fail.append_instr(ExitInstr(zero))
        # otherwise, keep running in new block
        self.current = next_block

    # # # # # # # #
    # EXPRESSIONS #

    def visit_Assignment(self, node: Assignment) -> Variable:
        value = self.visit(node.right)
        target = self.visit(node.left, ref=True)
        self.current.append_instr(StoreInstr(node.left.uc_type, value, target))
        return value

    # instructions for basic operations
    binary_op: dict[str, type[BinaryOpInstruction]] = {
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
        # Create the opcode and append to list
        return self.current.target_instr(self.binary_op[node.op], node.uc_type, left, right)

    visit_RelationOp = visit_BinaryOp

    def _unary_plus(self, uctype: uCType, node: Node) -> TempVariable:
        # nothing to generate
        return self.visit(node)

    # implementations for unary operators
    def _unary_minus(self, uctype: uCType, node: Node) -> TempVariable:
        source = self.visit(node)
        zero = self.current.new_literal(0)
        # negation is: 0 - value
        return self.current.target_instr(SubInstr, uctype, zero, source)

    def _unary_not(self, uctype: uCType, node: Node) -> TempVariable:
        source = self.visit(node)
        return self.current.target_instr(NotInstr, uctype, source)

    def _unary_star(self, uctype: uCType, node: Node, ref: bool = False) -> Variable:
        # get reference to inner value
        source = self.visit(node)
        if ref:
            return source
        # and get value using reference
        return self.current.target_instr(LoadInstr, uctype, source)

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
        array = self.visit(node.array)
        offset = self.visit(node.index)

        ptr = self.current.target_instr(ElemInstr, node.uc_type, array, offset)
        # return reference for compound types
        if ref or isinstance(node.uc_type, ArrayType):
            return ptr
        # and value for primaries
        else:
            return self.current.target_instr(LoadInstr, node.uc_type, ptr)

    def visit_FuncCall(self, node: FuncCall) -> Optional[TempVariable]:
        # get function address
        source = self._varname(node.callable)
        # analyze parameters
        varnames: list[tuple[Variable, uCType]] = []
        for param in node.parameters():
            varname = self.visit(param)
            varnames.append((varname, param.uc_type))
        # and load them
        for varname, uctype in varnames:
            self.current.append_instr(ParamInstr(uctype, varname))
        # then call the function
        if node.uc_type is VoidType:
            self.current.append_instr(CallInstr(node.uc_type, source))
        else:
            target = self.current.target_instr(CallInstr, node.uc_type, source)
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
        return self.glob.new_text(node.uc_type, node.value.encode("utf8") + b"\0")

    def _varname(self, ident: ID) -> MemoryVariable:
        """Get variable name for identifier"""
        if ident.version == "global":
            return DataVariable(ident.name)
        else:
            return NamedVariable(ident.name, ident.version)

    def visit_ID(self, node: ID, ref: bool = False) -> Variable:
        stackvar = self._varname(node)
        # load address
        if ref or isinstance(node.uc_type, ArrayType):
            return stackvar
        # or value
        return self.current.target_instr(LoadInstr, node.uc_type, stackvar)


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
