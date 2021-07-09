from __future__ import annotations
import argparse
import sys
from ctypes import CFUNCTYPE, c_int
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Mapping, Optional, TextIO, Union
from graphviz import Source
from llvmlite import binding, ir
from llvmlite.binding import ExecutionEngine, ModuleRef
from llvmlite.ir import Constant, Function, Module
from llvmlite.ir.builder import IRBuilder
from uc.uc_analysis import DataFlow
from uc.uc_ast import FuncDef, Program
from uc.uc_block import (
    BasicBlock,
    Block,
    BlockVisitor,
    BranchBlock,
    CodeBlock,
    EntryBlock,
    FunctionBlock,
    GlobalBlock,
)
from uc.uc_code import CodeGenerator
from uc.uc_ir import (
    AddInstr,
    AllocInstr,
    AndInstr,
    CallInstr,
    CBranchInstr,
    DivInstr,
    ElemInstr,
    ExitInstr,
    GetInstr,
    GlobalInstr,
    GlobalVariable,
    Instruction,
    JumpInstr,
    LabelName,
    LiteralInstr,
    LoadInstr,
    LocalVariable,
    LogicalInstruction,
    ModInstr,
    MulInstr,
    NotInstr,
    OrInstr,
    ParamInstr,
    ReturnInstr,
    StoreInstr,
    SubInstr,
    Variable,
)
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor
from uc.uc_type import BoolType, CharType, FloatType, IntType, StringType


def make_bytearray(buf: bytes) -> Constant:
    # Make a byte array constant from *buf*.
    lltype = ir.ArrayType(ir.IntType(8), len(buf))
    return Constant(lltype, bytearray(buf))


def make_constant(value: int | bool | str | float | list[int | bool | str | float]) -> ir.Constant:
    if isinstance(value, bool):
        return ir.Constant(BoolType.as_llvm(), value)
    elif isinstance(value, int):
        return ir.Constant(IntType.as_llvm(), value)
    elif isinstance(value, float):
        return ir.Constant(FloatType.as_llvm(), value)
    elif isinstance(value, str):
        return ir.Constant(CharType.as_llvm(), ord(value))
    else:
        elements = [make_constant(item) for item in value]
        return ir.Constant.literal_array(elements)


class LLVMModuleVisitor(BlockVisitor[Module]):
    def __init__(self):
        super().__init__(self.build_module)

    def build_module(self, program: Block) -> Module:
        assert isinstance(program, GlobalBlock)
        module = ir.Module(program.name)
        module.triple = binding.get_default_triple()
        return module

    def declare_printf(self, mod: Module) -> ir.types.Function:
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        return ir.Function(mod, printf_ty, name="printf")

    def declare_scanf(self, mod: Module) -> ir.types.Function:
        voidptr_ty = ir.IntType(8).as_pointer()
        scanf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        return ir.Function(mod, scanf_ty, name="scanf")

    def declare_global(
        self, mod: Module, instr: GlobalInstr, *, constant: bool = False
    ) -> ir.GlobalVariable:
        var = ir.GlobalVariable(mod, instr.type.as_llvm(), instr.varname.format())
        var.global_constant = constant
        if instr.value is None:
            pass
        elif isinstance(instr.value, Variable):
            var.initializer = mod.get_global(instr.value.format())
        elif isinstance(instr.type, StringType):
            var.initializer = make_bytearray(instr.value)
        else:
            var.initializer = make_constant(instr.value)
        return var

    def visit_GlobalBlock(self, program: GlobalBlock, module: Module) -> None:
        # declare external functions
        self.declare_printf(module)
        self.declare_scanf(module)
        # and global variables
        for instr in program.cdata:
            self.declare_global(module, instr, constant=True)
        for instr in program.data:
            self.declare_global(module, instr)


class FunctionVariables(Dict[LocalVariable, ir.NamedValue]):
    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module
        self.blocks: dict[LabelName, ir.Block] = {}

    def get_global(self, var: GlobalVariable) -> ir.NamedValue:
        return self.module.get_global(var.format())

    def get_local(self, var: LocalVariable) -> ir.NamedValue:
        return super().__getitem__(var)

    def get_block(self, label: LabelName) -> ir.Block:
        return self.blocks[label]

    def __getitem__(self, var: Union[Variable, LabelName]) -> ir.NamedValue:
        if isinstance(var, GlobalVariable):
            return self.get_global(var)
        elif isinstance(var, LabelName):
            return self.get_block(var)
        else:
            return self.get_local(var)


class LLVMInstructionBuilder:
    def __init__(self, block: ir.Block, vars: FunctionVariables) -> None:
        self.vars = vars
        self.builder = IRBuilder(block)
        self.params: list[ir.NamedValue] = []

    def builder_for(self, opname: str) -> Callable[[Instruction], None]:
        return getattr(self, f"build_{opname}", self.no_builder_found)

    def no_builder_found(self, instr: Instruction) -> None:
        # raise NotImplementedError(f"no builder for: {instr}")
        return None

    def build(self, instr: Instruction) -> None:
        self.builder_for(instr.opname)(instr)

    # # # # # # # # # # # #
    # Variables & Values  #

    def build_alloc(self, instr: AllocInstr) -> None:
        var = self.builder.alloca(instr.type.as_llvm(), name=instr.target.format())
        self.vars[instr.target] = var

    def build_load(self, instr: LoadInstr) -> None:
        source = self.vars[instr.varname]
        target = self.builder.load(source, name=instr.target.format())
        self.vars[instr.target] = target

    def build_store(self, instr: StoreInstr) -> None:
        source, target = self.vars[instr.source], self.vars[instr.target]
        self.builder.store(source, target)

    def build_literal(self, instr: LiteralInstr) -> None:
        self.vars[instr.target] = make_constant(instr.value)

    def build_elem(self, instr: ElemInstr) -> None:
        source, index = self.vars[instr.source], self.vars[instr.index]
        target = self.builder.gep(source, [index], name=instr.target.format())
        self.vars[instr.target] = target

    def build_get(self, instr: GetInstr) -> None:
        self.vars[instr.target] = self.vars[instr.source]

    # # # # # # # # # # #
    # Binary Operations #

    def build_add(self, instr: AddInstr) -> None:
        left, right = self.vars[instr.left], self.vars[instr.right]
        if instr.type is FloatType:
            target = self.builder.fadd(left, right, name=instr.target.format())
        else:
            target = self.builder.add(left, right, name=instr.target.format())
        self.vars[instr.target] = target

    def build_sub(self, instr: SubInstr) -> None:
        left, right = self.vars[instr.left], self.vars[instr.right]
        if instr.type is FloatType:
            target = self.builder.fsub(left, right, name=instr.target.format())
        else:
            target = self.builder.sub(left, right, name=instr.target.format())
        self.vars[instr.target] = target

    def build_mul(self, instr: MulInstr) -> None:
        left, right = self.vars[instr.left], self.vars[instr.right]
        if instr.type is FloatType:
            target = self.builder.fmul(left, right, name=instr.target.format())
        else:
            target = self.builder.mul(left, right, name=instr.target.format())
        self.vars[instr.target] = target

    def build_div(self, instr: DivInstr) -> None:
        left, right = self.vars[instr.left], self.vars[instr.right]
        if instr.type is FloatType:
            target = self.builder.fdiv(left, right, name=instr.target.format())
        else:
            target = self.builder.sdiv(left, right, name=instr.target.format())
        self.vars[instr.target] = target

    def build_mod(self, instr: ModInstr) -> None:
        left, right = self.vars[instr.left], self.vars[instr.right]
        target = self.builder.srem(left, right, name=instr.target.format())
        self.vars[instr.target] = target

    # # # # # # # # # # #
    # Unary Operations  #

    def build_not(self, instr: NotInstr) -> None:
        expr = self.vars[instr.expr]
        target = self.builder.not_(expr, name=instr.target.format())
        self.vars[instr.target] = target

    # # # # # # # # # # # # # # # #
    # Relational/Equality/Logical #

    logical_cmp_op = {
        "lt": "<",
        "le": "<=",
        "eq": "==",
        "gt": ">",
        "ge": ">=",
        "ne": "!=",
    }

    def build_logical_cmp(self, instr: LogicalInstruction) -> None:
        op = self.logical_cmp_op[instr.opname]
        left, right = self.vars[instr.left], self.vars[instr.right]
        if instr.type is FloatType:
            target = self.builder.fcmp_ordered(op, left, right, name=instr.target.format())
        else:
            target = self.builder.icmp_signed(op, left, right, name=instr.target.format())
        self.vars[instr.target] = target

    build_lt = build_logical_cmp
    build_le = build_logical_cmp
    build_gt = build_logical_cmp
    build_ge = build_logical_cmp
    build_eq = build_logical_cmp
    build_ne = build_logical_cmp

    def build_and(self, instr: AndInstr) -> None:
        left, right = self.vars[instr.left], self.vars[instr.right]
        target = self.builder.and_(left, right, name=instr.target.format())
        self.vars[instr.target] = target

    def build_or(self, instr: OrInstr) -> None:
        left, right = self.vars[instr.left], self.vars[instr.right]
        target = self.builder.or_(left, right, name=instr.target.format())
        self.vars[instr.target] = target

    # # # # # # # # # # #
    # Labels & Branches #

    def build_jump(self, instr: JumpInstr) -> None:
        self.builder.branch(self.vars[instr.target])

    def build_cbranch(self, instr: CBranchInstr) -> None:
        condition = self.vars[instr.expr_test]
        true, false = self.vars[instr.true_target], self.vars[instr.false_target]
        self.builder.cbranch(condition, true, false)

    # # # # # # # # # # # # #
    # Functions & Builtins  #

    def build_call(self, instr: CallInstr) -> None:
        fn = self.vars[instr.source]
        if instr.target is not None:
            result = self.builder.call(fn, self.params, name=instr.target.format())
            self.vars[instr.target] = result
        else:
            self.builder.call(fn, self.params)
        self.params.clear()

    def build_return(self, instr: ReturnInstr) -> None:
        if instr.target is not None:
            self.builder.ret(self.vars[instr.target])
        else:
            self.builder.ret_void()

    def build_param(self, instr: ParamInstr) -> None:
        self.params.append(self.vars[instr.source])


class LLVMFunctionVisitor(BlockVisitor[Function]):
    def __init__(self, module: Module) -> None:
        super().__init__(self.build_function)
        self.module = module

    def build_function(self, block: Block) -> Function:
        assert isinstance(block, FunctionBlock)
        function = Function(self.module, block.fntype.as_llvm(), block.name)

        self.vars = FunctionVariables(self.module)
        for (_, _, temp), arg in zip(block.params, function.args):
            arg.name = temp.format()
            self.vars[temp] = arg
        return function

    def visit_FunctionBlock(self, block: FunctionBlock, func: Function) -> None:
        self.visit(block.entry, func)

    def visit_CodeBlock(self, block: CodeBlock, func: Function) -> LLVMInstructionBuilder:
        llvmblock = func.append_basic_block(block.name)
        builder = LLVMInstructionBuilder(llvmblock, self.vars)
        self.vars.blocks[block.label] = llvmblock

        for instr in block.instr:
            builder.build(instr)

        if block.next is not None:
            self.visit(block.next, func)
        return builder

    def visit_EntryBlock(self, block: EntryBlock, func: Function) -> None:
        self.visit_CodeBlock(block, func)

    def visit_BasicBlock(self, block: BasicBlock, func: Function) -> None:
        builder = self.visit_CodeBlock(block, func)
        for jump in block.jump_instr:
            builder.build(jump)

    def visit_BranchBlock(self, block: BranchBlock, func: Function) -> None:
        builder = self.visit_CodeBlock(block, func)
        builder.build(block.cbranch)


class LLVMCodeGenerator(NodeVisitor[None]):
    def __init__(self, viewcfg: bool = False) -> None:
        self.viewcfg = viewcfg
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        self.engine = self.create_execution_engine()

    def create_execution_engine(self) -> ExecutionEngine:
        """
        Create an ExecutionEngine suitable for JIT code generation on
        the host CPU.  The engine is reusable for an arbitrary number of
        modules.
        """
        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        # And an execution engine with an empty backing module
        backing_mod = binding.parse_assembly("")
        return binding.create_mcjit_compiler(backing_mod, target_machine)

    def visit_Program(self, node: Program) -> None:
        bb = LLVMModuleVisitor()
        self.module = bb.visit(node.cfg)
        # Visit all the function definitions and emit the llvm code from the
        # uCIR code stored inside basic blocks.
        for decl in node.gdecls:
            self.visit(decl)

    def visit_FuncDef(self, node: FuncDef) -> None:
        # decl.cfg contains the Control Flow Graph for the function
        bb = LLVMFunctionVisitor(self.module)
        # Visit the CFG to define the Function and Create the Basic Blocks
        function = bb.visit(node.cfg)

        if self.viewcfg:
            dot = binding.get_function_cfg(function)
            gv: Source = binding.view_dot_graph(dot, f"{node.funcname}.ll.gv", False)
            gv.view(quiet=True, quiet_view=True)

    def optimize_ir(self, mod: ModuleRef, opt: Literal["ctm", "dce", "cfg", "all"]) -> None:
        # apply some optimization passes on module
        pmb = binding.create_pass_manager_builder()
        pm = binding.create_module_pass_manager()

        pmb.opt_level = 3
        pmb.size_level = 2
        pmb.loop_vectorize = True
        pmb.slp_vectorize = True
        if opt == "ctm" or opt == "all":
            # Sparse conditional constant propagation and merging
            pm.add_sccp_pass()
            # Merges duplicate global constants together
            pm.add_constant_merge_pass()
            # Combine inst to form fewer, simple inst
            # This pass also does algebraic simplification
            pm.add_instruction_combining_pass()
        if opt == "dce" or opt == "all":
            pm.add_dead_code_elimination_pass()
        if opt == "cfg" or opt == "all":
            # Performs dead code elimination and basic block merging
            pm.add_cfg_simplification_pass()

        pmb.populate(pm)
        pm.run(mod)

    def compile_ir(self, opt: Literal["ctm", "dce", "cfg", "all", None] = None) -> ModuleRef:
        """
        Compile the LLVM IR string with the given engine.
        The compiled module object is returned.
        """
        # Create a LLVM module object from the IR
        llvm_ir = str(self.module)
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        # Now add the module and make sure it is ready for execution
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()

        if opt:
            self.optimize_ir(mod, opt)
        return mod

    def save_ir(self, output_file: TextIO) -> None:
        output_file.write(str(self.module))

    def execute_ir(
        self, opt: Literal["ctm", "dce", "cfg", "all", None], opt_file: Optional[TextIO] = None
    ) -> int:
        mod = self.compile_ir(opt)
        if opt_file is not None:
            opt_file.write(str(mod))

        # Obtain a pointer to the compiled 'main' - it's the address of its JITed code in memory.
        main_ptr = self.engine.get_function_address("main")
        # To convert an address to an actual callable thing we have to use
        # CFUNCTYPE, and specify the arguments & return type.
        main_function = CFUNCTYPE(c_int)(main_ptr)
        # Now 'main_function' is an actual callable we can invoke
        return main_function()


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate LLVM IR. By default, this script runs the LLVM IR without any optimizations.",
        type=Path,
    )
    parser.add_argument(
        "-c",
        "--cfg",
        help="show the CFG of the optimized uCIR for each function in pdf format",
        action="store_true",
    )
    parser.add_argument(
        "--llvm-opt",
        default=None,
        choices=["ctm", "dce", "cfg", "all"],
        help="specify which llvm pass optimizations should be enabled",
    )
    args = parser.parse_args()

    create_cfg = args.cfg
    llvm_opt = args.llvm_opt

    # get input path
    input_file: Path = args.input_file

    # check if file exists
    if not input_file.exists():
        print("Input", input_file, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_file) as file:
        ast = p.parse(file)

    sema = Visitor()
    sema.visit(ast)

    gen = CodeGenerator()
    gen.visit(ast)

    opt = DataFlow()
    opt.visit(ast)

    llvm = LLVMCodeGenerator(create_cfg)
    llvm.visit(ast)
    # llvm.execute_ir(llvm_opt, None)
    with open("test.ll", "w") as irfile:
        llvm.save_ir(irfile)
