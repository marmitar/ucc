# ---------------------------------------------------------------------------------
# uc: uc_interpreter.py
#
# uCInterpreter class: A simple interpreter for the uC intermediate representation
#                      see https://github.com/iviarcio/mc921
#
# Copyright (c) 2019-2020, Marcio M Pereira. All rights reserved.
#
# This software is provided by the author, "as is" without any warranties.
# Redistribution and use in source form with or without modification are
# permitted, but the source code must retain the above copyright notice.
# ---------------------------------------------------------------------------------
from __future__ import annotations
import re
import sys
from enum import Enum, unique
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Union
from uc.uc_ast import sizeof
from uc.uc_block import CodeList
from uc.uc_ir import (
    AddInstr,
    AllocInstr,
    AndInstr,
    BinaryOpInstruction,
    CallInstr,
    CBranchInstr,
    DataVariable,
    DefineInstr,
    ElemInstr,
    EqInstr,
    ExitInstr,
    GeInstr,
    GetInstr,
    GlobalInstr,
    GlobalVariable,
    GtInstr,
    Instruction,
    JumpInstr,
    LabelInstr,
    LabelName,
    LeInstr,
    LiteralInstr,
    LoadInstr,
    LocalVariable,
    LtInstr,
    ModInstr,
    MulInstr,
    NeInstr,
    NotInstr,
    OrInstr,
    ParamInstr,
    PrintInstr,
    ReadInstr,
    ReturnInstr,
    StoreInstr,
    SubInstr,
    TempVariable,
    UnaryOpInstruction,
    Variable,
)
from uc.uc_type import ArrayType, CharType, FloatType, IntType, StringType, uCType


def printerr(*args) -> None:
    """Show error or warning messages."""
    print(*args, file=sys.stderr, flush=True)


@unique
class Uninitialized(Enum):
    __slots__ = ()

    def __str__(self) -> str:
        return "XXXX"

    def __repr__(self) -> str:
        return "Uninit"

    def _ignore(self, *_) -> Uninitialized:
        return self

    __add__ = _ignore
    __radd__ = _ignore
    __sub__ = _ignore
    __rsub__ = _ignore
    __mul__ = _ignore
    __rmul__ = _ignore
    __div__ = _ignore
    __rdiv__ = _ignore
    __mod__ = _ignore
    __rmod__ = _ignore
    __truediv__ = _ignore
    __rtruediv__ = _ignore
    __and__ = _ignore
    __rand__ = _ignore
    __or__ = _ignore
    __ror__ = _ignore
    __neg__ = _ignore
    __pos__ = _ignore

    def _cmp(self, *_) -> bool:
        return False

    __eq__ = _cmp
    __ne__ = _cmp
    __lt__ = _cmp
    __le__ = _cmp
    __gt__ = _cmp
    __ge__ = _cmp
    __bool__ = _cmp

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)

    Uninit = ()


Uninit = Uninitialized.Uninit

Value = Union[str, int, float, Literal[Uninit]]
Size = Union[int, uCType]
Scope = Dict[Union[LocalVariable, LabelName], Value]

# Data memory
M: list[Value] = []


class Interpreter:
    """
    Runs an interpreter on the uC intermediate code generated for
    uC compiler.   The implementation idea is as follows.  Given
    a sequence of instruction tuples such as:

         code = [
              ('literal_int', 1, '%1'),
              ('literal_int', 2, '%2'),
              ('add_int', '%1', '%2, '%3')
              ('print_int', '%3')
              ...
         ]

    The class executes methods self.run_opcode(args).  For example:

             self.run_literal_int(1, '%1')
             self.run_literal_int(2, '%2')
             self.run_add_int('%1', '%2', '%3')
             self.run_print_int('%3')

    Instructions for use:
        1. Instantiate an object of the Interpreter class
        2. Call the run method of this object passing the produced
           code as a parameter
    """

    def __init__(self, debug: bool = False):
        global M
        self.input: Optional[str] = None
        M = 10000 * [Uninit]  # Memory for global & local vars

        # Dictionary of address of global vars & constants
        self.globals: dict[GlobalVariable, int] = {}
        # Dictionary of address of local vars relative to sp
        self.vars: Scope = {}
        # offset for all labels in each function
        self.labels: dict[GlobalVariable, list[tuple[LabelName, int]]] = {}

        # offset (index) of local & global vars. Note that
        # each instance of var has absolute address in Memory
        self.offset = 0
        # Stack to save address of vars between calls
        self.stack: list[Scope] = []
        # Stack to save & restore the last offset
        self.sp: list[int] = []

        # List of parameters from caller (value)
        self.params: list[Value] = []
        # list of register to store result from call instruction
        self.retval: list[TempVariable] = []
        # Stack of return addresses (program counters)
        self.returns: list[int] = []

        self.pc: int = 0  # Program Counter
        self.lastpc: int = 0  # last pc
        self.start: Optional[int] = None  # PC of the main function
        self.debug: bool = debug  # Set the debug mode

    # # # # # # #
    # DEBUGGER  #

    def _show_idb_help(self):
        msg = """
          s, step: run in step mode;
          g, go <pc>:  goto the program counter;
          l, list {<start> <end>}? : List the ir code;
          e, ex {<vars>}+ : Examine the variables;
          a, assign <var> <type> <value>: Assign the value of given type to var;
          v, view : show he current line of execution;
          r, run : run (terminate) the program in normal mode;
          q, quit : quit (abort) the program;
          h, help: print this text.
        """
        printerr(msg)

    def _idb(self, pos: int) -> Optional[int]:
        init = pos - 2
        if init < 1:
            init = 1
        end = pos + 3
        if end >= self.lastpc:
            end = self.lastpc
        for i in range(init, end):
            mark = ": >> " if i == pos else ":    "
            printerr(str(i) + mark + self.code[i].format())
        printerr()
        return self._parse_input()

    def _assign_location(self, loc: str, uc_type: str, value: str) -> None:
        val = value
        if uc_type == "int":
            val = int(val)
        elif uc_type == "float":
            val = float(val)
        var = re.split(r"\[|\]", loc)
        if len(var) == 1:
            if loc.startswith("%"):
                M[self.vars[loc]] = val
            elif loc.startswith("@"):
                M[self.globals[loc]] = val
            else:
                printerr(loc + ": unrecognized var or temp")
        elif len(var) == 3:
            address = var[0]
            if var[1].isdigit():
                idx = int(var[1])
                if loc.startswith("%"):
                    M[self.vars[address] + idx] = val
                elif loc.startswith("@"):
                    M[self.globals[address] + idx] = val
                else:
                    printerr(loc + ": unrecognized var or temp")
            else:
                printerr(loc + ": only assign single var or temp at time")
        else:
            printerr("Construction not supported. For matrices, linearize it.")

    def _view_location(self, loc: str) -> None:
        var: list[str] = re.split(r"\[|\]", loc)
        if len(var) == 1:
            if loc.startswith("%"):
                printerr(loc + " : " + str(M[self.vars[loc]]))
            elif loc.startswith("@"):
                printerr(loc + " : " + str(M[self.globals[loc]]))
            else:
                printerr(loc + ": unrecognized var or temp")
        elif len(var) == 3:
            address = var[0]
            if var[1].isdigit():
                idx = int(var[1])
                if loc.startswith("%"):
                    printerr(loc + " : " + str(M[self.vars[address] + idx]))
                elif loc.startswith("@"):
                    printerr(loc + " : " + str(M[self.globals[address] + idx]))
                else:
                    printerr(loc + ": unrecognized var or temp")
            else:
                tmp = re.split(":", var[1])
                i = int(tmp[0])
                j = int(tmp[1]) + 1
                if loc.startswith("%"):
                    printerr(loc + " : " + str(M[self.vars[address] + i : self.vars[address] + j]))
                elif loc.startswith("@"):
                    printerr(
                        loc + " : " + str(M[self.globals[address] + i : self.globals[address] + j])
                    )
                else:
                    printerr(loc + ": unrecognized var or temp")
        else:
            printerr("Construction not supported. For matrices, linearize it.")

    def _parse_input(self) -> Optional[int]:
        while True:
            try:
                cmd = list(input("idb> ").split())
                if cmd[0] == "s" or cmd[0] == "step":
                    return None
                elif cmd[0] == "g" or cmd[0] == "go":
                    return int(cmd[1])
                elif cmd[0] == "e" or cmd[0] == "ex":
                    for var in cmd[1:]:
                        self._view_location(var)
                elif cmd[0] == "a" or cmd[0] == "assign":
                    if len(cmd) != 4:
                        printerr(
                            "Cmd assign error: Just only single var and type must be specified."
                        )
                    else:
                        self._assign_location(cmd[1], cmd[2], cmd[3])
                elif cmd[0] == "l" or cmd[0] == "list":
                    if len(cmd) == 3:
                        start = int(cmd[1])
                        end = int(cmd[2])
                    else:
                        start = 1
                        end = self.lastpc
                    for i in range(start, end):
                        printerr(str(i) + ":    " + self.code[i].format())
                elif cmd[0] == "v" or cmd[0] == "view":
                    self._idb(self.pc)
                elif cmd[0] == "r" or cmd[0] == "run":
                    self.debug = False
                    return None
                elif cmd[0] == "q" or cmd[0] == "quit":
                    return 0
                elif cmd[0] == "h" or cmd[0] == "help":
                    self._show_idb_help()
                else:
                    printerr(cmd[0] + " : unrecognized command")
            except Exception:
                printerr("unrecognized command")

    # # # # # # # #
    # MEMORY & IO #

    def _alloc_labels(self, funcname: GlobalVariable) -> None:
        # Alloc labels for current function definition
        for label, offset in self.labels[funcname]:
            self.vars[label] = offset

    def _alloc_data(self, size: Size) -> int:
        # Alloc space in memory and save the offset in the dictionary
        # for new vars or temporaries, only.
        offset = self.offset
        self.offset += int(size)
        # return new address
        return offset

    def _alloc_reg(self, target: TempVariable) -> None:
        if target not in self.vars:
            self.vars[target] = Uninit

    def _get_value(self, source: Variable) -> Value:
        if isinstance(source, GlobalVariable):
            return self.globals[source]
        else:
            return self.vars.get(source, Uninit)

    def _get_literal(self, source: Union[Value, Variable]) -> Value:
        if isinstance(source, Variable):
            return self._get_value(source)
        else:
            return source

    def _load_multiple(self, address: int, size: Size) -> list[Value]:
        return M[address : address + int(size)]

    def _store_multiple(self, address: int, value: list[Union[Variable, Value]]) -> None:
        # overwrite data, if size is wrong
        M[address : address + len(value)] = map(self._get_literal, value)

    def _push(self, target: Optional[TempVariable] = None) -> None:
        # prepare register for return value
        reg = target or TempVariable(0)

        # save the addresses of the vars from caller & their last offset
        self.stack.append(self.vars)
        self.sp.append(self.offset)
        self.retval.append(reg)
        self.returns.append(self.pc)

    def _pop(self) -> None:
        # get return value
        retval = self.vars.get(TempVariable(0), Uninit)
        # restore the vars of the caller
        self.vars = self.stack.pop()
        # set the return value
        register = self.retval.pop()
        self.vars[register] = retval
        # restore the last offset from the caller
        self.offset = self.sp.pop()
        # jump to the return point in the caller
        self.pc = self.returns.pop()

    def _read_line(self) -> str:
        while not self.input:
            line = sys.stdin.readline()
            if not line:
                printerr("Unexpected end of input file.")
            self.input = line.rstrip("\n")
        return self.input

    def _read_word(self) -> str:
        line = self._read_line()
        # split at first space
        [word, remainder] = line.split(" ", maxsplit=1)
        self.input = remainder
        return word

    def _read_char(self) -> str:
        line = self._read_line()
        char = line[0]
        self.input = line[1:]
        return char

    # # # # # # #
    # EXECUTION #

    def _split_data(self, literal: Union[Any, list[Any]]) -> list[Union[Value, Variable]]:
        def flatten(value: Union[Any, list[Any], None]) -> Iterator[Union[Value, Variable]]:
            if isinstance(value, str):
                yield from value
            elif isinstance(value, (bytes, list, tuple)):
                for item in value:
                    yield from flatten(item)
            elif value is not None:
                yield value

        return list(flatten(literal))

    def _prepare_globals(self) -> int:
        """Allocate global variables and find label offsets."""

        # name and pc for current function
        current_function: Optional[DataVariable] = None
        for pc, instr in enumerate(self.code):
            # allocate global variables
            if isinstance(instr, GlobalInstr):
                self.globals[instr.varname] = self.offset
                if instr.value != None:
                    value = self._split_data(instr.value)
                    self._store_multiple(self.offset, value)
                self.offset += sizeof(instr.type)
            # allocate function reference
            elif isinstance(instr, DefineInstr):
                current_function = instr.source
                self.globals[current_function] = pc
                self.labels[current_function] = []
            # store label address
            elif isinstance(instr, LabelInstr):
                self.labels[current_function].append((instr.name, pc))

        return pc + 1

    def run(self, ircode: CodeList) -> None:
        """
        Run intermediate code in the interpreter.  ircode is a list
        of instruction tuples.  Each instruction (opcode, *args) is
        dispatched to a method self.run_opcode(*args)
        """
        # First, store the global vars & constants
        # Also, set the start pc to the main function entry
        self.code = list(ircode.with_start())
        self.offset = 0
        self.lastpc = self._prepare_globals()

        # Now, running the program starting from the main function
        # If run in debug mode, show the available command lines.
        if self.debug:
            printerr("Interpreter running in debug mode:")
            self._show_idb_help()

        self.pc = self.globals.get(DataVariable(".start"))
        if self.pc is None:
            self.pc = self.lastpc

        _breakpoint: Optional[int] = None
        while True:
            try:
                if _breakpoint is not None:
                    if _breakpoint == 0:
                        sys.exit(0)
                    if self.pc == _breakpoint:
                        _breakpoint = self._idb(self.pc)
                elif self.debug:
                    _breakpoint = self._idb(self.pc)
                instr = self.code[self.pc]
            except IndexError:
                break
            self.pc += 1
            # get instruction runner
            executor = getattr(self, f"run_{instr.opname}", None)
            if executor is not None:
                executor(instr)
            elif not isinstance(instr, LabelInstr):
                printerr(f"Warning: No run_{instr.opname}() method")

    #
    # Run Operations, except Binary, Relational & Cast
    #
    def run_alloc(self, alloc: AllocInstr) -> None:
        self.vars[alloc.varname] = self._alloc_data(alloc.type)

    def run_call(self, call: CallInstr) -> None:
        # save the return pc in the return stack
        self._push(call.target)
        # jump to the calle function
        self.pc = self._get_value(call.source)

    def run_cbranch(self, branch: CBranchInstr) -> None:
        if self._get_value(branch.expr_test):
            self.pc = self.vars[branch.true_target]
        else:
            self.pc = self.vars[branch.false_target]

    # Enter the function
    def run_define(self, define: DefineInstr) -> None:
        # clear the dictionary of caller local vars and their offsets in memory
        self.vars = {}
        # load parameters in register bank
        for _, register in reversed(define.args):
            # Note that arrays (size >=1) are passed by reference only.
            if self.params:
                self.vars[register] = self.params.pop()
            else:
                self.vars[register] = Uninit

        self.params = []
        # prepare function
        self._alloc_labels(define.source)

    def run_elem(self, elem: ElemInstr) -> None:
        base = self._get_value(elem.source)
        idx = self._get_value(elem.index)
        # calculate and access address
        self.vars[elem.target] = base + idx * sizeof(elem.type)

    def run_exit(self, exit: ExitInstr) -> None:
        # We reach the end of main function, so return to system
        # with the code returned by main in the return register.
        print(end="", flush=True)
        # exit with return value
        retval = self._get_value(exit.source)
        sys.exit(retval)

    def run_get(self, get: GetInstr) -> None:
        self.vars[get.target] = self._get_value(get.source)

    def run_jump(self, jump: JumpInstr) -> None:
        self.pc = self.vars[jump.target]

    # load literals into registers
    def run_literal(self, literal: LiteralInstr) -> None:
        self.vars[literal.target] = self._get_literal(literal.value)

    def run_load(self, load: LoadInstr) -> None:
        address = self._get_value(load.varname)
        self.vars[load.target] = M[address]

    def run_param(self, param: ParamInstr) -> None:
        source = self._get_value(param.source)
        self.params.append(source)

    def run_print(self, op: PrintInstr) -> None:
        if op.source is None:
            print(flush=True)
        elif isinstance(op.type, StringType):
            address = self._get_value(op.source)
            data = self._load_multiple(address, op.type)
            text = str(bytes(data), "utf8")
            print(text, sep="", end="", flush=True)
        else:
            data = self._get_value(op.source)
            print(data, end="", flush=True)

    def run_read(self, read: ReadInstr) -> None:
        try:
            # read value
            if read.type is IntType:
                value = int(self._read_word())
            elif read.type is FloatType:
                value = float(self._read_word())
            elif read.type is CharType:
                value = self._read_char()
            else:
                value = list(self._read_line())
            # and store in variable
            source = self._get_value(read.source)
            self._store_value(source, value)
        # may evoke parsing errors
        except ValueError:
            printerr("Illegal input value.")

    def run_return(self, ret: ReturnInstr) -> None:
        # set return value
        if ret.target:
            value = self._get_value(ret.target)
            self.vars[TempVariable(0)] = value
        # and return pc
        self._pop()

    def run_store(self, store: StoreInstr) -> None:
        target = self._get_value(store.target)
        if isinstance(store.type, ArrayType):
            source = self._get_value(store.source)
            data = self._load_multiple(source, store.type)
            M[target : target + len(data)] = data
        else:
            value = self._get_value(store.source)
            M[target] = value

    #
    # perform binary, relational & cast operations
    #
    def _run_binop(self, instr: BinaryOpInstruction, op: Callable[[Value, Value], Value]) -> None:
        left = self._get_value(instr.left)
        right = self._get_value(instr.right)
        self.vars[instr.target] = op(left, right)

    def run_add(self, instr: AddInstr) -> None:
        self._run_binop(instr, lambda x, y: x + y)

    def run_sub(self, instr: SubInstr) -> None:
        self._run_binop(instr, lambda x, y: x - y)

    def run_mul(self, instr: MulInstr) -> None:
        self._run_binop(instr, lambda x, y: x * y)

    def run_mod(self, instr: ModInstr) -> None:
        self._run_binop(instr, lambda x, y: x % y)

    def run_div(self, instr: AddInstr) -> None:
        if instr.type is FloatType:
            self._run_binop(instr, lambda x, y: x / y)
        else:
            self._run_binop(instr, lambda x, y: x // y)

    # Integer comparisons

    def run_lt(self, instr: LtInstr) -> None:
        self._run_binop(instr, lambda x, y: x < y)

    def run_le(self, instr: LeInstr) -> None:
        self._run_binop(instr, lambda x, y: x <= y)

    def run_gt(self, instr: GtInstr) -> None:
        self._run_binop(instr, lambda x, y: x > y)

    def run_ge(self, instr: GeInstr) -> None:
        self._run_binop(instr, lambda x, y: x >= y)

    def run_eq(self, instr: EqInstr) -> None:
        self._run_binop(instr, lambda x, y: x == y)

    def run_ne(self, instr: NeInstr) -> None:
        self._run_binop(instr, lambda x, y: x != y)

    def run_and(self, instr: AndInstr) -> None:
        self._run_binop(instr, lambda x, y: x and y)

    def run_or(self, instr: OrInstr) -> None:
        self._run_binop(instr, lambda x, y: x or y)

    # Unary ops

    def _run_unop(self, instr: UnaryOpInstruction, op: Callable[[Value], Value]) -> None:
        expr = self._get_value(instr.expr)
        self.vars[instr.target] = op(expr)

    def run_not(self, instr: NotInstr) -> None:
        self._run_unop(instr, lambda x: not x)

    def run_sitofp(self, instr) -> None:
        self._run_unop(instr, lambda x: float(x))

    def run_fptosi(self, instr) -> None:
        self._run_unop(instr, lambda x: int(x))
