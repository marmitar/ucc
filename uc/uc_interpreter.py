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
from typing import Iterator, Optional, Union
from uc.uc_block import Instruction


def format_instruction(t: tuple[str, ...]) -> str:
    operand = t[0].split("_")
    op = operand[0]
    ty = operand[1] if len(operand) > 1 else None
    if len(operand) >= 3:
        for qual in operand[2:]:
            if qual == "*":
                ty += "*"
            else:
                ty += f"[{qual}]"
    if len(t) > 1:
        if op == "define":
            return f"\n{op} {ty} {t[1]} (" + ", ".join(" ".join(el) for el in t[2]) + ")"
        else:
            _str = "" if op == "global" else "  "
            if op == "jump":
                _str += f"{op} label {t[1]}"
            elif op == "cbranch":
                _str += f"{op} {t[1]} label {t[2]} label {t[3]}"
            elif op == "global":
                if ty.startswith("string"):
                    _str += f"{t[1]} = {op} {ty} '{t[2]}'"
                elif len(t) > 2:
                    _str += f"{t[1]} = {op} {ty} {t[2]}"
                else:
                    _str += f"{t[1]} = {op} {ty}"
            elif op == "return" or op == "print":
                _str += f"{op} {ty} {t[1]}"
            elif op == "sitofp" or op == "fptosi":
                _str += f"{t[2]} = {op} {t[1]}"
            elif op == "store" or op == "param":
                _str += f"{op} {ty} "
                for el in t[1:]:
                    _str += f"{el} "
            else:
                _str += f"{t[-1]} = {op} {ty} "
                for el in t[1:-1]:
                    _str += f"{el} "
            return _str
    elif ty == "void":
        return f"  {op}"
    else:
        return f"{op}"


Value = Union[str, int, float, None]


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
        global inputline, M
        inputline: list[str] = []
        M: list[Value] = 10000 * [None]  # Memory for global & local vars

        self.globals: dict[str, int] = {}  # Dictionary of address of global vars & constants
        self.vars: dict[str, int] = {}  # Dictionary of address of local vars relative to sp

        self.offset = 0  # offset (index) of local & global vars. Note that
        # each instance of var has absolute address in Memory
        self.stack: list[dict[str, int]] = []  # Stack to save address of vars between calls
        self.sp: list[int] = []  # Stack to save & restore the last offset

        self.params: list[int] = []  # List of parameters from caller (address)
        self.result = None  # Result Value (address) from the callee

        self.registers: list[str] = []  # Stack of register names (in the caller) to return value
        self.returns: list[int] = []  # Stack of return addresses (program counters)

        self.pc = 0  # Program Counter
        self.lastpc = 0  # last pc
        self.start = 0  # PC of the main function
        self.debug = debug  # Set the debug mode

    def _extract_operation(self, source: str) -> tuple[str, dict[str, str]]:
        aux = source.split("_")
        if aux[0] not in {"fptosi", "sitofp", "label", "jump", "cbranch", "call"}:
            opcode = aux[0] + "_" + aux[1]
            modifier = {}
            for i, val in enumerate(aux[2:]):
                if val.isdigit():
                    modifier[f"dim{i}"] = val
                elif val == "*":
                    modifier[f"ptr{i}"] = val

            return opcode, modifier
        else:
            opcode = aux[0]
            return opcode, {}

    def _copy_data(
        self, address: int, size: int, value: Union[str, list[Union[str, list[str]]]]
    ) -> None:
        def flatten(item: Union[str, list[Union[str, list[str]]]]) -> Iterator[str]:
            if isinstance(item, str):
                yield item
                return
            for subitem in item:
                for value in flatten(subitem):
                    yield value

        value = [item for item in flatten(value)]
        M[address : address + size] = value

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
        print(msg)

    def _idb(self, pos: int) -> Optional[int]:
        init = pos - 2
        if init < 1:
            init = 1
        end = pos + 3
        if end >= self.lastpc:
            end = self.lastpc
        for i in range(init, end):
            mark = ": >> " if i == pos else ":    "
            print(str(i) + mark + format_instruction(self.code[i]))
        print()
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
                print(loc + ": unrecognized var or temp")
        elif len(var) == 3:
            address = var[0]
            if var[1].isdigit():
                idx = int(var[1])
                if loc.startswith("%"):
                    M[self.vars[address] + idx] = val
                elif loc.startswith("@"):
                    M[self.globals[address] + idx] = val
                else:
                    print(loc + ": unrecognized var or temp")
            else:
                print(loc + ": only assign single var or temp at time")
        else:
            print("Construction not supported. For matrices, linearize it.")

    def _view_location(self, loc: str) -> None:
        var: list[str] = re.split(r"\[|\]", loc)
        if len(var) == 1:
            if loc.startswith("%"):
                print(loc + " : " + str(M[self.vars[loc]]))
            elif loc.startswith("@"):
                print(loc + " : " + str(M[self.globals[loc]]))
            else:
                print(loc + ": unrecognized var or temp")
        elif len(var) == 3:
            address = var[0]
            if var[1].isdigit():
                idx = int(var[1])
                if loc.startswith("%"):
                    print(loc + " : " + str(M[self.vars[address] + idx]))
                elif loc.startswith("@"):
                    print(loc + " : " + str(M[self.globals[address] + idx]))
                else:
                    print(loc + ": unrecognized var or temp")
            else:
                tmp = re.split(":", var[1])
                i = int(tmp[0])
                j = int(tmp[1]) + 1
                if loc.startswith("%"):
                    print(loc + " : " + str(M[self.vars[address] + i : self.vars[address] + j]))
                elif loc.startswith("@"):
                    print(
                        loc + " : " + str(M[self.globals[address] + i : self.globals[address] + j])
                    )
                else:
                    print(loc + ": unrecognized var or temp")
        else:
            print("Construction not supported. For matrices, linearize it.")

    def _parse_input(self) -> Optional[int]:
        while True:
            try:
                cmd = list(input("idb> ").strip().split(" "))
                if cmd[0] == "s" or cmd[0] == "step":
                    return None
                elif cmd[0] == "g" or cmd[0] == "go":
                    return int(cmd[1])
                elif cmd[0] == "e" or cmd[0] == "ex":
                    for i in range(1, len(cmd)):
                        self._view_location(cmd[i])
                elif cmd[0] == "a" or cmd[0] == "assign":
                    if len(cmd) != 4:
                        print("Cmd assign error: Just only single var and type must be specified.")
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
                        print(str(i) + ":    " + format_instruction(self.code[i]))
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
                    print(cmd[0] + " : unrecognized command")
            except Exception:
                print("unrecognized command")

    def run(self, ircode: list[Instruction]) -> None:
        """
        Run intermediate code in the interpreter.  ircode is a list
        of instruction tuples.  Each instruction (opcode, *args) is
        dispatched to a method self.run_opcode(*args)
        """
        # First, store the global vars & constants
        # Also, set the start pc to the main function entry
        self.code = [ir.as_tuple() for ir in ircode]
        self.pc = 0
        self.offset = 0
        while True:
            try:
                op = self.code[self.pc]
            except IndexError:
                break
            if len(op) > 1:  # that is, instruction is not a label
                opcode, modifier = self._extract_operation(op[0])
                if opcode.startswith("global"):
                    self.globals[op[1]] = self.offset
                    # get the size of global var
                    if not modifier:
                        # size equals 1 or is a constant, so we use only
                        # one slot in the memory to make it simple.
                        if len(op) == 3:
                            M[self.offset] = op[2]
                        self.offset += 1
                    else:
                        _len = 1
                        for args in modifier.values():
                            if args.isdigit():
                                _len *= int(args)
                        if len(op) == 3:
                            self._copy_data(self.offset, _len, op[2])
                        self.offset += _len
                elif opcode.startswith("define"):
                    self.globals[op[1]] = self.offset
                    M[self.offset] = self.pc
                    self.offset += 1
                    if op[1] == "@main":
                        self.start = self.pc
            self.pc += 1

        # Now, running the program starting from the main function
        # If run in debug mode, show the available command lines.
        if self.debug:
            print("Interpreter running in debug mode:")
            self._show_idb_help()
        self.lastpc = self.pc - 1
        self.pc = self.start
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
                op = ircode[self.pc]
            except IndexError:
                break
            self.pc += 1
            if len(op) > 1 or op[0] == "return_void" or op[0] == "print_void":
                opcode, modifier = self._extract_operation(op[0])
                if hasattr(self, "run_" + opcode):
                    if not modifier:
                        getattr(self, "run_" + opcode)(*op[1:])
                    else:
                        getattr(self, "run_" + opcode + "_")(*op[1:], **modifier)
                else:
                    print("Warning: No run_" + opcode + "() method", flush=True)

    #
    # Auxiliary methods
    #
    def _alloc_labels(self) -> None:
        # Alloc labels for current function definition. Due to the uCIR and due to
        # the chosen memory model, this is done every time we enter a function.
        lpc = self.pc
        while True:
            try:
                op = self.code[lpc]
                opcode = op[0]
                lpc += 1
                if opcode.startswith("define"):
                    break
                elif len(op) == 1 and opcode != "return_void":
                    # labels don't go to memory, just store the pc on dictionary
                    # labels appears as name:, so we need to extract just the name
                    self.vars["%" + opcode[:-1]] = lpc
            except IndexError:
                break

    def _alloc_reg(self, target: str) -> None:
        # Alloc space in memory and save the offset in the dictionary
        # for new vars or temporaries, only.
        if target not in self.vars:
            self.vars[target] = self.offset
            self.offset += 1

    def _get_address(self, source: str) -> int:
        if source.startswith("@"):
            return self.globals[source]
        else:
            return self.vars[source]

    def _get_input(self) -> None:
        global inputline
        while True:
            if len(inputline) > 0:
                break
            inputline = sys.stdin.readline()
            if not inputline:
                print("Unexpected end of input file.", flush=True)
            inputline = inputline[:-1].strip().split()

    def _get_value(self, source: str) -> Value:
        if source.startswith("@"):
            return M[self.globals[source]]
        else:
            return M[self.vars[source]]

    def _load_multiple_values(self, size: int, varname: str, target: str) -> None:
        self.vars[target] = self.offset
        self.offset += size
        self._store_multiple_values(size, target, varname)

    def _push(self, locs: list[str], no_return: bool) -> None:
        # save the addresses of the vars from caller & their last offset
        self.stack.append(self.vars)
        self.sp.append(self.offset)

        # clear the dictionary of caller local vars and their offsets in memory
        # alloc the temporary with reg %0 in case of void function and initialize
        # the memory with None value. Copy the parameters passed to the callee in
        # their local vars. Finally, cleanup parameters list used to transfer vars
        self.vars = {}

        if no_return:
            self._alloc_reg("%0")
            M[self.vars["%0"]] = None

        for idx, val in enumerate(self.params):
            # Note that arrays (size >=1) are passed by reference only.
            self.vars[locs[idx]] = self.offset
            M[self.offset] = M[val]
            self.offset += 1
        self.params = []
        self._alloc_labels()

    def _pop(self, target: Optional[int]) -> None:
        if self.returns:
            # get the return value
            if target:
                value = M[target]
            else:
                value = None
            # restore the vars of the caller
            self.vars = self.stack.pop()
            # store in the caller return register the _value
            M[self.vars[self.registers.pop()]] = value
            # restore the last offset from the caller
            self.offset = self.sp.pop()
            # jump to the return point in the caller
            self.pc = self.returns.pop()
        else:
            # We reach the end of main function, so return to system
            # with the code returned by main in the return register.
            print(end="", flush=True)
            if target is None:
                # void main () was defined, so exit with value 0
                sys.exit(0)
            else:
                sys.exit(M[target])

    def _store_deref(self, target: str, value: Optional[int]) -> None:
        if target.startswith("@"):
            M[M[self.globals[target]]] = value
        else:
            M[M[self.vars[target]]] = value

    def _store_multiple_values(self, dim: int, target: str, value: str) -> None:
        left = self._get_address(target)
        right = self._get_address(value)
        if value.startswith("@"):
            if isinstance(M[right], str):
                _value = list(M[right])
                M[left : left + dim] = _value
                return
        M[left : left + dim] = M[right : right + dim]

    def _store_value(self, target: str, value: Value) -> None:
        if target.startswith("@"):
            M[self.globals[target]] = value
        else:
            M[self.vars[target]] = value

    #
    # Run Operations, except Binary, Relational & Cast
    #
    def run_alloc_int(self, varname: str) -> None:
        self._alloc_reg(varname)
        M[self.vars[varname]] = 0

    run_alloc_float = run_alloc_int
    run_alloc_char = run_alloc_int

    def run_alloc_int_(self, varname: str, **kwargs: str) -> None:
        dim = 1
        for arg in kwargs.values():
            if arg.isdigit():
                dim *= int(arg)
        self.vars[varname] = self.offset
        M[self.offset : self.offset + dim] = dim * [0]
        self.offset += dim

    run_alloc_float_ = run_alloc_int_
    run_alloc_char_ = run_alloc_int_

    def run_call(self, source: str, target: str) -> None:
        # alloc register to return and append it to register stack
        self._alloc_reg(target)
        self.registers.append(target)
        # save the return pc in the return stack
        self.returns.append(self.pc)
        # jump to the calle function
        if source.startswith("@"):
            self.pc = M[self.globals[source]]
        else:
            self.pc = M[self.vars[source]]

    def run_cbranch(self, expr_test: str, true_target: str, false_target: str) -> None:
        if M[self.vars[expr_test]]:
            self.pc = self.vars[true_target]
        else:
            self.pc = self.vars[false_target]

    # Enter the function
    def run_define_int(self, source: str, args: list[tuple[str, ...]]) -> None:
        if source == "@main":
            # alloc register to the return value but initialize it with "None".
            # We use the "None" value when main function returns void.
            self._alloc_reg("%0")
            # alloc the labels with respective pc's
            self._alloc_labels()
        else:
            # extract the location names of function args
            locs = [el[1] for el in args]
            self._push(locs, False)

    run_define_float = run_define_int
    run_define_char = run_define_int

    def run_define_void(self, source: str, args: list[tuple[str, ...]]) -> None:
        if source == "@main":
            # alloc register to the return value but not initialize it.
            # We use the "None" value to check if main function returns void.
            self._alloc_reg("%0")
            # alloc the labels with respective pc's
            self._alloc_labels()
        else:
            # extract the location names of function args
            locs = [el[1] for el in args]
            self._push(locs, True)

    def run_elem_int(self, source: str, index: str, target: str) -> None:
        self._alloc_reg(target)
        aux = self._get_address(source)
        idx = self._get_value(index)
        address = aux + idx
        self._store_value(target, address)

    run_elem_float = run_elem_int
    run_elem_char = run_elem_int

    def run_get_int(self, source: str, target: str) -> None:
        # We never generate this code without * (ref) but we need to define it
        pass

    def run_get_int_(self, source: str, target: str, **kwargs) -> None:
        # kwargs always contain * (ref), so we ignore it.
        self._store_value(target, self._get_address(source))

    run_get_float_ = run_get_int_
    run_get_char_ = run_get_int_

    def run_jump(self, target: str) -> None:
        self.pc = self.vars[target]

    # load literals into registers
    def run_literal_int(self, value: Optional[int], target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = value

    run_literal_float = run_literal_int

    def run_literal_char(self, value: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = value.strip("'")

    # Load/stores
    def run_load_int(self, varname: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = self._get_value(varname)

    run_load_float = run_load_int
    run_load_char = run_load_int
    run_load_bool = run_load_int

    def run_load_int_(self, varname: str, target: str, **kwargs: str) -> None:
        ref = 0
        dim = 1
        for arg in kwargs.values():
            if arg.isdigit():
                dim *= int(arg)
            elif arg == "*":
                ref += 1
        if ref == 0:
            self._load_multiple_values(dim, varname, target)
        elif dim == 1 and ref == 1:
            self._alloc_reg(target)
            M[self.vars[target]] = M[self._get_value(varname)]

    run_load_float_ = run_load_int_
    run_load_char_ = run_load_int_

    def run_param_int(self, source: str) -> None:
        self.params.append(self.vars[source])

    run_param_float = run_param_int
    run_param_char = run_param_int

    def run_param_int_(self, source: str, **kwargs) -> None:
        # Note that arrays are passed by reference
        self.params.append(self.vars[source])

    run_param_float_ = run_param_int_
    run_param_char_ = run_param_int_

    def run_print_string(self, source: str) -> None:
        c = list(self._get_value(source))
        print(c, end="", flush=True)

    def run_print_int(self, source: str) -> None:
        print(self._get_value(source), end="", flush=True)

    run_print_float = run_print_int
    run_print_char = run_print_int
    run_print_bool = run_print_int

    def run_print_void(self) -> None:
        print(flush=True)

    def _read_int(self) -> Optional[int]:
        global inputline
        self._get_input()
        try:
            v1 = inputline[0]
            inputline = inputline[1:]
            return int(v1)
        except Exception:
            print("Illegal input value.", flush=True)

    def run_read_int(self, source: str) -> None:
        value = self._read_int()
        self._store_value(source, value)

    def run_read_int_(self, source: str, **kwargs) -> None:
        value = self._read_int()
        self._store_deref(source, value)

    def _read_float(self) -> None:
        global inputline
        self._get_input()
        try:
            v1 = inputline[0]
            inputline = inputline[1:]
            return float(v1)
        except Exception:
            print("Illegal input value.", flush=True)

    def run_read_float(self, source: str) -> None:
        value = self._read_float()
        self._store_value(source, value)

    def run_read_float_(self, source: str, **kwargs) -> None:
        value = self._read_float()
        self._store_deref(source, value)

    def run_read_char(self, source: str) -> None:
        global inputline
        self._get_input()
        v1 = inputline[0]
        inputline = inputline[1:]
        self._store_value(source, v1)

    def run_read_char_(self, source: str, **kwargs) -> None:
        global inputline
        self._get_input()
        v1 = inputline[0]
        inputline = inputline[1:]
        self._store_deref(source, v1)

    def run_return_int(self, target: str) -> None:
        self._pop(self.vars[target])

    run_return_float = run_return_int
    run_return_char = run_return_int

    def run_return_void(self) -> None:
        self._pop(M[self.vars["%0"]])

    def run_store_int(self, source: str, target: str) -> None:
        self._store_value(target, self._get_value(source))

    run_store_float = run_store_int
    run_store_char = run_store_int
    run_store_bool = run_store_int

    def run_store_int_(self, source: str, target: str, **kwargs: str) -> None:
        ref = 0
        dim = 1
        for arg in kwargs.values():
            if arg.isdigit():
                dim *= int(arg)
            elif arg == "*":
                ref += 1
        if ref == 0:
            self._store_multiple_values(dim, target, source)
        elif dim == 1 and ref == 1:
            self._store_deref(target, self._get_value(source))

    run_store_float_ = run_store_int_
    run_store_char_ = run_store_int_

    #
    # perform binary, relational & cast operations
    #
    def run_add_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] + M[self.vars[right]]

    def run_sub_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] - M[self.vars[right]]

    def run_mul_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] * M[self.vars[right]]

    def run_mod_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] % M[self.vars[right]]

    def run_div_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] // M[self.vars[right]]

    def run_div_float(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] / M[self.vars[right]]

    # Floating point ops (same as int)
    run_add_float = run_add_int
    run_sub_float = run_sub_int
    run_mul_float = run_mul_int

    # Integer comparisons
    def run_lt_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] < M[self.vars[right]]

    def run_le_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] <= M[self.vars[right]]

    def run_gt_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] > M[self.vars[right]]

    def run_ge_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] >= M[self.vars[right]]

    def run_eq_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] == M[self.vars[right]]

    def run_ne_int(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] != M[self.vars[right]]

    # Float comparisons
    run_lt_float = run_lt_int
    run_le_float = run_le_int
    run_gt_float = run_gt_int
    run_ge_float = run_ge_int
    run_eq_float = run_eq_int
    run_ne_float = run_ne_int

    # String comparisons
    run_lt_char = run_lt_int
    run_le_char = run_le_int
    run_gt_char = run_gt_int
    run_ge_char = run_ge_int
    run_eq_char = run_eq_int
    run_ne_char = run_ne_int

    # Bool comparisons
    run_eq_bool = run_eq_int
    run_ne_bool = run_ne_int

    def run_and_bool(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] and M[self.vars[right]]

    def run_or_bool(self, left: str, right: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = M[self.vars[left]] or M[self.vars[right]]

    def run_not_bool(self, source: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = not self._get_value(source)

    def run_sitofp(self, source: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = float(self._get_value(source))

    def run_fptosi(self, source: str, target: str) -> None:
        self._alloc_reg(target)
        M[self.vars[target]] = int(self._get_value(source))
