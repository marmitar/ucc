import argparse
import pathlib
import sys
from ctypes import CFUNCTYPE, c_int
from llvmlite import binding, ir
from uc.uc_ast import FuncDef
from uc.uc_block import BlockVisitor
from uc.uc_code import CodeGenerator
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor


def make_bytearray(buf):
    # Make a byte array constant from *buf*.
    b = bytearray(buf)
    n = len(b)
    return ir.Constant(ir.ArrayType(ir.IntType(8), n), b)


class LLVMFunctionVisitor(BlockVisitor):
    def __init__(self, module):
        self.module = module
        self.func = None
        self.builder = None
        self.loc = {}

    def _extract_operation(self, inst):
        _modifier = {}
        _ctype = None
        _aux = inst.split("_")
        _opcode = _aux[0]
        if _opcode not in {"fptosi", "sitofp", "jump", "cbranch", "define"}:
            _ctype = _aux[1]
            for i, _val in enumerate(_aux[2:]):
                if _val.isdigit():
                    _modifier["dim" + str(i)] = _val
                elif _val == "*":
                    _modifier["ptr" + str(i)] = _val
        return _opcode, _ctype, _modifier

    def _get_loc(self, target):
        try:
            if target[0] == "%":
                return self.loc[target]
            elif target[0] == "@":
                return self.module.get_global(target[1:])
        except KeyError:
            return None

    def _global_constant(self, builder_or_module, name, value, linkage="internal"):
        # Get or create a (LLVM module-)global constant with *name* or *value*.
        if isinstance(builder_or_module, ir.Module):
            mod = builder_or_module
        else:
            mod = builder_or_module.module
        data = ir.GlobalVariable(mod, value.type, name=name)
        data.linkage = linkage
        data.global_constant = True
        data.initializer = value
        data.align = 1
        return data

    def _cio(self, fname, format, *target):
        # Make global constant for string format
        mod = self.builder.module
        fmt_bytes = make_bytearray((format + "\00").encode("ascii"))
        global_fmt = self._global_constant(mod, mod.get_unique_name(".fmt"), fmt_bytes)
        fn = mod.get_global(fname)
        ptr_fmt = self.builder.bitcast(global_fmt, ir.IntType(8).as_pointer())
        return self.builder.call(fn, [ptr_fmt] + list(target))

    def _build_print(self, val_type, target):
        if target:
            # get the object assigned to target
            _value = self._get_loc(target)
            if val_type == "int":
                self._cio("printf", "%d", _value)
            elif val_type == "float":
                self._cio("printf", "%.2f", _value)
            elif val_type == "char":
                self._cio("printf", "%c", _value)
            elif val_type == "string":
                self._cio("printf", "%s", _value)
        else:
            self._cio("printf", "\n")

    def build(self, inst):
        opcode, ctype, modifier = self._extract_operation(inst[0])
        if hasattr(self, "_build_" + opcode):
            args = inst[1:] if len(inst) > 1 else (None,)
            if not modifier:
                getattr(self, "_build_" + opcode)(ctype, *args)
            else:
                getattr(self, "_build_" + opcode + "_")(ctype, *inst[1:], **modifier)
        else:
            print("Warning: No _build_" + opcode + "() method", flush=True)

    def visit_BasicBlock(self, block):
        # TODO: Complete
        # Create the LLVM function when visiting its first block
        # First visit of the block should create its LLVM equivalent
        # Second visit should create the LLVM instructions within the block
        pass

    def visit_ConditionBlock(self, block):
        # TODO: Complete
        # Create the LLVM function when visiting its first block
        # First visit of the block should create its LLVM equivalent
        # Second visit should create the LLVM instructions within the block
        pass


class LLVMCodeGenerator(NodeVisitor):
    def __init__(self, viewcfg):
        self.viewcfg = viewcfg
        self.binding = binding
        self.binding.initialize()
        self.binding.initialize_native_target()
        self.binding.initialize_native_asmprinter()

        self.module = ir.Module(name=__file__)
        self.module.triple = self.binding.get_default_triple()

        self.engine = self._create_execution_engine()

        # declare external functions
        self._declare_printf_function()
        self._declare_scanf_function()

    def _create_execution_engine(self):
        """
        Create an ExecutionEngine suitable for JIT code generation on
        the host CPU.  The engine is reusable for an arbitrary number of
        modules.
        """
        target = self.binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        # And an execution engine with an empty backing module
        backing_mod = binding.parse_assembly("")
        return binding.create_mcjit_compiler(backing_mod, target_machine)

    def _declare_printf_function(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        printf = ir.Function(self.module, printf_ty, name="printf")
        self.printf = printf

    def _declare_scanf_function(self):
        voidptr_ty = ir.IntType(8).as_pointer()
        scanf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
        scanf = ir.Function(self.module, scanf_ty, name="scanf")
        self.scanf = scanf

    def _compile_ir(self):
        """
        Compile the LLVM IR string with the given engine.
        The compiled module object is returned.
        """
        # Create a LLVM module object from the IR
        llvm_ir = str(self.module)
        mod = self.binding.parse_assembly(llvm_ir)
        mod.verify()
        # Now add the module and make sure it is ready for execution
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()
        return mod

    def save_ir(self, output_file):
        output_file.write(str(self.module))

    def execute_ir(self, opt, opt_file):
        mod = self._compile_ir()

        if opt:
            # apply some optimization passes on module
            pmb = self.binding.create_pass_manager_builder()
            pm = self.binding.create_module_pass_manager()

            pmb.opt_level = 0
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
            opt_file.write(str(mod))

        # Obtain a pointer to the compiled 'main' - it's the address of its JITed code in memory.
        main_ptr = self.engine.get_function_address("main")
        # To convert an address to an actual callable thing we have to use
        # CFUNCTYPE, and specify the arguments & return type.
        main_function = CFUNCTYPE(c_int)(main_ptr)
        # Now 'main_function' is an actual callable we can invoke
        res = main_function()

    def visit_Program(self, node):
        # node.text contains the global instructions into the Program node
        self._generate_global_instructions(node.text)
        # Visit all the function definitions and emit the llvm code from the
        # uCIR code stored inside basic blocks.
        for _decl in node.gdecls:
            if isinstance(_decl, FuncDef):
                # _decl.cfg contains the Control Flow Graph for the function
                bb = LLVMFunctionVisitor(self.module)
                # Visit the CFG to define the Function and Create the Basic Blocks
                bb.visit(_decl.cfg)
                # Visit CFG again to create the instructions inside Basic Blocks
                bb.visit(_decl.cfg)
                if self.viewcfg:
                    dot = binding.get_function_cfg(bb.func)
                    gv = binding.view_dot_graph(dot, _decl.decl.name.name, False)
                    gv.filename = _decl.decl.name.name + ".ll.gv"
                    gv.view()


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate LLVM IR. By default, this script runs the LLVM IR without any optimizations.",
        type=str,
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

    llvm = LLVMCodeGenerator(create_cfg)
    llvm.visit(ast)
    llvm.execute_ir(llvm_opt, None)
