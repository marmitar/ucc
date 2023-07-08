from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Generator, Literal, Optional, TextIO, Union

from graphviz import Source
from llvmlite import binding, ir
from llvmlite.binding import ExecutionEngine, ModulePassManager, ModuleRef
from llvmlite.ir import Constant, Function, Module
from llvmlite.ir.builder import IRBuilder

from .analysis import DataFlow
from .ast import FuncDef, Program
from .block import (
    BasicBlock,
    Block,
    BlockVisitor,
    BranchBlock,
    CodeBlock,
    EntryBlock,
    FunctionBlock,
    GlobalBlock,
)
from .code import CodeGenerator
from .ir import (
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
    PrintInstr,
    ReadInstr,
    ReturnInstr,
    StoreInstr,
    SubInstr,
    Variable,
)
from .parser import UCParser
from .sema import NodeVisitor, Visitor
from .type import (
    ArrayType,
    BoolType,
    CharType,
    FloatType,
    FunctionType,
    IntType,
    StringType,
    VoidType,
)


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
    def __init__(self, whole_program: bool = False):
        super().__init__(self.build_module)
        self.whole_program = whole_program

    def build_module(self, program: Block) -> Module:
        assert isinstance(program, GlobalBlock)
        module = ir.Module(program.name)
        module.triple = binding.get_default_triple()
        return module

    def declare_printf(self, mod: Module) -> ir.Function:
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        return ir.Function(mod, printf_ty, name="printf")

    def declare_scanf(self, mod: Module) -> ir.Function:
        voidptr_ty = ir.IntType(8).as_pointer()
        scanf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        return ir.Function(mod, scanf_ty, name="scanf")

    def declare_exit(self, mod: Module) -> ir.Function:
        exit_ty = ir.FunctionType(ir.VoidType(), [ir.IntType(32)])
        exit_decl = ir.Function(mod, exit_ty, name="exit")
        exit_decl.attributes.add("noreturn")
        return exit_decl

    def declare_global(self, mod: Module, instr: GlobalInstr, *, constant: bool = False) -> None:
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
        if self.whole_program:
            var.unnamed_addr = True
            var.linkage = "internal"

    def declare_internal_string(self, mod: Module, name: str, value: str) -> None:
        data = make_bytearray(value.encode("utf8") + b"\0")
        var = ir.GlobalVariable(mod, data.type, name)
        var.global_constant = True
        var.unnamed_addr = True
        var.initializer = data
        var.linkage = "internal"

    def visit_GlobalBlock(self, program: GlobalBlock, module: Module) -> None:
        # declare format strings
        self.declare_internal_string(module, ".fmt.int", "%d")
        self.declare_internal_string(module, ".fmt.float", "%lf")
        self.declare_internal_string(module, ".fmt.char", "%c")
        self.declare_internal_string(module, ".fmt.bool", "%hhu")
        self.declare_internal_string(module, ".fmt.string", "%s")
        self.declare_internal_string(module, ".fmt.newline", "\n")
        # declare external functions
        self.declare_printf(module)
        self.declare_scanf(module)
        self.declare_exit(module)
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

    def get_global(self, var: Union[GlobalVariable, str]) -> ir.GlobalVariable | ir.Function:
        if isinstance(var, GlobalVariable):
            var = var.format()
        return self.module.get_global(var)

    def get_local(self, var: LocalVariable) -> ir.NamedValue:
        return super().__getitem__(var)

    def get_block(self, label: LabelName) -> ir.Block:
        return self.blocks[label]

    def __getitem__(self, var: Union[Variable, LabelName, str]) -> ir.NamedValue:
        if isinstance(var, LocalVariable):
            return self.get_local(var)
        elif isinstance(var, LabelName):
            return self.get_block(var)
        else:
            return self.get_global(var)


class LLVMInstructionBuilder:
    def __init__(self, block: ir.Block, vars: FunctionVariables) -> None:
        self.vars = vars
        self.builder = IRBuilder(block)
        self.params: list[ir.NamedValue] = []

    def builder_for(self, opname: str) -> Callable[[Instruction], None]:
        return getattr(self, f"build_{opname}", self.no_builder_found)

    def no_builder_found(self, instr: Instruction) -> None:
        raise NotImplementedError(f"no builder for: {instr}")

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
        if isinstance(instr.type, ArrayType):
            source = self.builder.load(source)
        self.builder.store(source, target)

    def build_literal(self, instr: LiteralInstr) -> None:
        self.vars[instr.target] = make_constant(instr.value)

    def build_elem(self, instr: ElemInstr) -> None:
        source, index = self.vars[instr.source], self.vars[instr.index]
        ptr = self.builder.bitcast(source, instr.type.as_llvm().as_pointer())
        target = self.builder.gep(ptr, [index], name=instr.target.format())
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

    def build_jump(self, target: JumpInstr | LabelName) -> None:
        if isinstance(target, JumpInstr):
            target = target.target
        self.builder.branch(self.vars[target])

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

    fmt_spec = {
        IntType: "int",
        FloatType: "float",
        CharType: "char",
        BoolType: "bool",
    }

    def get_format(self, spec: str) -> ir.CastInstr:
        fmt = self.vars[f".fmt.{spec}"]
        ptr_ty = ir.IntType(8).as_pointer()
        return self.builder.bitcast(fmt, ptr_ty)

    def build_read(self, instr: ReadInstr) -> None:
        spec = self.fmt_spec.get(instr.type, "string")
        fmt = self.get_format(spec)

        source = self.vars[instr.source]
        self.builder.call(self.vars["scanf"], [fmt, source])

    def build_print(self, instr: PrintInstr) -> None:
        if instr.source is None:
            args = [self.get_format("newline")]
        elif spec := self.fmt_spec.get(instr.type, None):
            args = [self.get_format(spec), self.vars[instr.source]]
        else:
            ptr_ty = ir.IntType(8).as_pointer()
            ptr = self.builder.bitcast(self.vars[instr.source], ptr_ty)
            args = [ptr]

        self.builder.call(self.vars["printf"], args)

    def build_exit(self, instr: ExitInstr) -> None:
        source = self.vars[instr.source]
        self.builder.call(self.vars["exit"], [source])
        self.builder.unreachable()


class LLVMFunctionVisitor(BlockVisitor[Function]):
    def __init__(self, module: Module, function: FunctionBlock) -> None:
        super().__init__(lambda _: self.function)
        self.module = module
        self.function = self.build_function(function)

    def build_function(self, block: FunctionBlock) -> Function:
        function = Function(self.module, block.fntype.as_llvm(), block.name)

        self.vars = FunctionVariables(self.module)
        for (_, _, temp), arg in zip(block.params, function.args):
            arg.name = temp.format()
            self.vars[temp] = arg
        return function

    def visit_FunctionBlock(self, block: FunctionBlock, func: Function) -> None:
        self.visit(block.entry, func)

    def visit_CodeBlock(
        self, block: CodeBlock, func: Function, jump: Optional[CodeBlock]
    ) -> LLVMInstructionBuilder:
        llvmblock = func.append_basic_block(block.name)
        builder = LLVMInstructionBuilder(llvmblock, self.vars)
        self.vars.blocks[block.label] = llvmblock

        for instr in block.instr:
            builder.build(instr)

        if block.next is not None:
            self.visit(block.next, func)

        if jump and not llvmblock.is_terminated:
            builder.build_jump(jump.label)

        return builder

    def visit_EntryBlock(self, block: EntryBlock, func: Function) -> None:
        self.visit_CodeBlock(block, func, block.next)

    def visit_BasicBlock(self, block: BasicBlock, func: Function) -> None:
        jump = block.jumps[0] if block.jumps else block.next or block
        self.visit_CodeBlock(block, func, jump)

    def visit_BranchBlock(self, block: BranchBlock, func: Function) -> None:
        builder = self.visit_CodeBlock(block, func, None)
        builder.build(block.cbranch)


Iterator = Generator[None, None, None]


class LLVMCodeGenerator(NodeVisitor[Optional[Iterator]]):
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
        self.main = node.uc_type
        bb = LLVMModuleVisitor(node.uc_type is not VoidType)
        self.module = bb.visit(node.cfg)
        # declare functions
        visitors: list[Iterator] = []
        for decl in node.gdecls:
            visitor = self.visit(decl)
            if visitor is not None:
                next(visitor)
                visitors.append(visitor)
        # finish definition
        for visitor in visitors:
            for _ in visitor:
                ...

    def visit_FuncDef(self, node: FuncDef) -> Iterator:
        # declare visitor, but don't visit yet
        bb = LLVMFunctionVisitor(
            self.module,
            node.cfg,
        )
        yield
        # Visit the CFG to define the Function and Create the Basic Blocks
        function = bb.visit(node.cfg)
        if isinstance(self.main, FunctionType) and self.main.funcname != node.cfg.name:
            function.linkage = "private"

        if self.viewcfg:
            dot = binding.get_function_cfg(function)
            gv: Source = binding.view_dot_graph(dot, f"{node.funcname}.ll.gv", False)
            gv.view(quiet=True, quiet_view=True)

    def add_pm_passes(
        self, pm: ModulePassManager, opt: Literal["ctm", "dce", "cfg", "all"]
    ) -> None:
        pm.add_type_based_alias_analysis_pass()
        pm.add_basic_alias_analysis_pass()
        if opt == "ctm" or opt == "all":
            # Sparse conditional constant propagation and merging
            pm.add_sccp_pass()
            # Merges duplicate global constants together
            pm.add_constant_merge_pass()
            # Combine inst to form fewer, simple inst
            # This pass also does algebraic simplification
            pm.add_instruction_combining_pass()
            pm.add_sroa_pass()
            pm.add_ipsccp_pass()
        if opt == "dce" or opt == "all":
            pm.add_dead_code_elimination_pass()
            pm.add_dead_arg_elimination_pass()
        if opt == "cfg" or opt == "all":
            pm.add_licm_pass()
            # Performs dead code elimination and basic block merging
            pm.add_cfg_simplification_pass()
        if opt == "all":
            pm.add_gvn_pass()
            pm.add_global_dce_pass()
            pm.add_global_optimizer_pass()
            pm.add_function_inlining_pass(10)
            pm.add_dead_code_elimination_pass()

    def optimize_ir(self, mod: ModuleRef, opt: Literal["ctm", "dce", "cfg", "all"]) -> None:
        # apply some optimization passes on module
        pmb = binding.create_pass_manager_builder()
        pm = binding.create_module_pass_manager()

        pmb.opt_level = 3
        pmb.size_level = 2
        pmb.loop_vectorize = True
        pmb.slp_vectorize = True

        self.add_pm_passes(pm, opt)
        if opt == "all":
            self.add_pm_passes(pm, "all")

        pmb.populate(pm)
        pm.run(mod)

    def compile_ir(self) -> ModuleRef:
        """
        Compile the LLVM IR string with the given engine.
        The compiled module object is returned.
        """
        llvm_ir = str(self.module)
        mod = binding.parse_assembly(llvm_ir)
        mod.name = self.module.name
        mod.verify()
        return mod

    def save_ir(self, output_file: TextIO) -> None:
        contents = str(self.compile_ir())
        output_file.write(contents)

    def get_function(self, function: FunctionType) -> Callable:
        # Obtain a pointer to the compiled function - it's the address of its JITed code in memory.
        func_ptr = self.engine.get_function_address(function.funcname)
        # To convert an address to an actual callable thing we have to use
        # CFUNCTYPE, and specify the arguments & return type.
        func_ty = function.as_ctype()
        # Now 'function' is an actual callable we can invoke
        return func_ty(func_ptr)

    def execute_ir(
        self, opt: Literal["ctm", "dce", "cfg", "all", None], opt_file: Optional[TextIO] = None
    ) -> Optional[int]:
        mod = self.compile_ir()
        # Now add the module and make sure it is ready for execution
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()

        if opt is not None:
            self.optimize_ir(mod, opt)
        if opt_file is not None:
            opt_file.write(str(mod))

        if isinstance(self.main, FunctionType):
            main = self.get_function(self.main)
            return main()


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=Path,
        help="""Path to file to be used to generate LLVM IR. By default, this script runs the
            LLVM IR without any optimizations.""",
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
    parser.add_argument("-s", "--save-ir", type=Path, help="Path to save generated LLVM IR")
    args = parser.parse_intermixed_args()

    create_cfg = args.cfg
    llvm_opt = args.llvm_opt

    # get input path
    input_file: Path = args.input_file
    save_path: Optional[Path] = args.save_ir

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

    if save_path:
        with open(f"{save_path}.ll", "w") as irfile:
            llvm.save_ir(irfile)
        with open(f"{save_path}.opt.ll", "w") as irfile:
            llvm.execute_ir(llvm_opt, irfile)
    else:
        llvm.execute_ir(llvm_opt, None)
