#!/usr/bin/env python3

# ============================================================
# uc -- uC (a.k.a. micro C) language compiler
#
# This is the main program for the uc compiler, which just
# invokes the different stages of the compiler proper.
# ============================================================

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    TextIO,
    Union,
)
from uc.uc_analysis import DataFlow
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_llvm import LLVMCodeGenerator
from uc.uc_parser import UCParser
from uc.uc_sema import Visitor

"""
One of the most important (and difficult) parts of writing a compiler
is reliable reporting of error messages back to the user.  This file
defines some generic functionality for dealing with errors throughout
the compiler project. Error handling is based on a subscription/logging
based approach.

To report errors in uc compiler, we use the error() function. For example:

       error("Some kind of compiler error message", lineno)

where lineno is the line number on which the error occurred.

Error handling is based on a subscription based model using context-managers
and the subscribe_errors() function. For example, to route error messages to
standard output, use this:

       with subscribe_errors(print):
            run_compiler()

To send messages to standard error, you can do this:

       import sys
       from functools import partial
       with subscribe_errors(partial(print, file=sys.stderr)):
            run_compiler()

To route messages to a logger, you can do this:

       import logging
       log = logging.getLogger("somelogger")
       with subscribe_errors(log.error):
            run_compiler()

To collect error messages for the purpose of unit testing, do this:

       errs = []
       with subscribe_errors(errs.append):
            run_compiler()
       # Check errs for specific errors

The utility function errors_reported() returns the total number of
errors reported so far.  Different stages of the compiler might use
this to decide whether or not to keep processing or not.

Use clear_errors() to clear the total number of errors.
"""

_subscribers: List[Callable[[str], Any]] = []
_num_errors = 0


def error(
    message: Union[str, Exception], lineno: Optional[int] = None, filename: Optional[str] = None
):
    """Report a compiler error to all subscribers"""
    global _num_errors
    if not filename:
        if not lineno:
            errmsg = "{}".format(message)
        else:
            errmsg = "{}: {}".format(lineno, message)
    else:
        if not lineno:
            errmsg = "{}: {}".format(filename, message)
        else:
            errmsg = "{}:{}: {}".format(filename, lineno, message)
    for subscriber in _subscribers:
        subscriber(errmsg)
    _num_errors += 1


def errors_reported() -> int:
    """Return number of errors reported."""
    return _num_errors


def clear_errors() -> None:
    """Clear the total number of errors reported."""
    global _num_errors
    _num_errors = 0


@contextmanager
def subscribe_errors(handler: Callable[[str], Any]) -> Iterator[None]:
    """Context manager that allows monitoring of compiler error messages.
    Use as follows where handler is a callable taking a single argument
    which is the error message string:

    with subscribe_errors(handler):
        ... do compiler ops ...
    """
    _subscribers.append(handler)
    try:
        yield
    finally:
        _subscribers.remove(handler)


class Args(Protocol):
    filename: Path
    yaml: bool
    ast: bool
    sem: bool
    ir: bool
    no_run: bool
    idb: bool
    cfg: bool
    opt: bool
    verbose: bool
    llvm: bool
    llvm_opt: Literal["ctm", "dce", "cfg", "all"]


def printerr(*args: Any) -> None:
    print(*args, file=sys.stderr)


class Compiler:
    """This object encapsulates the compiler and serves as a
    facade interface to the 'meat' of the compiler underneath.
    """

    def __init__(self, cl_args: Args):
        self.code: Optional[str] = None
        self.total_errors = 0
        self.total_warnings = 0
        self.args = cl_args
        self.file: dict[str, TextIO] = {}

    def _parse(self) -> None:
        """Parses the source code. If ast_file != None,
        prints out the abstract syntax tree.
        """
        try:
            self.parser = UCParser()
            with open(self.args.filename) as code:
                self.ast = self.parser.parse(code)
            if not self.args.yaml and (file := self.file.get("ast")):
                self.ast.show(buf=file, showcoord=True)
        except AssertionError as e:
            error(e)

    def _sema(self) -> None:
        """Decorate AST with semantic actions. If sem_file != None,
        prints out the abstract syntax tree."""
        try:
            self.sema = Visitor()
            self.sema.visit(self.ast)
            if not self.args.yaml and (file := self.file.get("sem")):
                self.ast.show(buf=file, showcoord=True)
        except AssertionError as e:
            error(e)

    def _codegen(self) -> None:
        self.gen = CodeGenerator(self.args.cfg)
        self.gen.visit(self.ast)
        self.gencode = self.gen.code
        if not self.args.yaml and (file := self.file.get("ir")):
            self.gen.show(buf=file)

    def _opt(self):
        self.opt = DataFlow(self.args.cfg)
        self.opt.visit(self.ast)
        self.optcode = self.opt.code
        if not self.args.yaml and (file := self.file.get("opt")):
            self.opt.show(buf=file)

    def _llvm(self):
        self.llvm = LLVMCodeGenerator(self.args.cfg)
        self.llvm.visit(self.ast)
        if not self.args.yaml and (file := self.file.get("llvm")):
            self.llvm.save_ir(file)
        if self.run:
            if self.args.llvm_opt:
                self.llvm.execute_ir(self.args.llvm_opt, self.file.get("llvm-opt"))
            else:
                self.llvm.execute_ir(self.args.llvm_opt, self.file.get("llvm"))

    def _do_compile(self) -> None:
        """Compiles the code to the given source file."""
        self._parse()
        if not errors_reported():
            self._sema()
        if not errors_reported():
            self._codegen()
            if self.args.opt:
                self._opt()
            if self.args.llvm:
                self._llvm()

    def compile(self) -> int:
        """Compiles the given  filename"""

        if self.args.filename.suffix == ".uc":
            path = self.args.filename.parent
            filename = self.args.filename.stem
        else:
            path = self.args.filename.parent
            filename = self.args.filename.name

        if self.args.ast and not self.args.yaml:
            ast_filename = path.joinpath(filename + ".ast")
            printerr(f"Outputting the AST to {ast_filename}.")
            self.file["ast"] = open(ast_filename, "w")

        if self.args.sem and not self.args.yaml:
            sem_filename = path.joinpath(filename + ".sem")
            printerr(f"Outputting the sem to {sem_filename}.")
            self.file["sem"] = open(sem_filename, "w")

        if self.args.ir and not self.args.yaml:
            ir_filename = path.joinpath(filename + ".ir")
            printerr(f"Outputting the uCIR to {ir_filename}.")
            self.file["ir"] = open(ir_filename, "w")

        if self.args.opt and not self.args.yaml:
            opt_filename = path.joinpath(filename + ".opt")
            printerr(f"Outputting the optimized uCIR to {opt_filename}.")
            self.file["opt"] = open(opt_filename, "w")

        if self.args.llvm and not self.args.yaml:
            llvm_filename = path.joinpath(filename + ".ll")
            printerr(f"Outputting the LLVM IR to {llvm_filename}.")
            self.file["llvm"] = open(llvm_filename, "w")

        if self.args.llvm_opt and not self.args.yaml:
            llvm_opt_filename = path.joinpath(filename + ".opt.ll")
            printerr(f"Outputting the optimized LLVM IR to {llvm_opt_filename}.")
            self.file["llvm-opt"] = open(llvm_opt_filename, "w")

        self.run = not self.args.no_run
        if self.args.verbose:
            printerr(f"Compiling {filename}:")
        with subscribe_errors(printerr):
            self._do_compile()
            if n := errors_reported():
                printerr(f"{n} error(s) encountered.")
            elif not self.args.llvm:
                if self.args.opt:
                    speedup = len(self.gencode) / len(self.optcode)
                    printerr(
                        f"default = {len(self.gencode)}, optimized = {len(self.optcode)}, speedup = {speedup:.2f}"
                    )
                if self.run and not self.args.cfg:
                    vm = Interpreter(self.args.idb)
                    if self.args.opt:
                        vm.run(self.optcode)
                    else:
                        vm.run(self.gencode)

        while self.file:
            _, file = self.file.popitem()
            file.close()
        return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=Path)
    parser.add_argument(
        "-y",
        "--yaml",
        help="run in the CI (Continuous Integration) mode",
        action="store_true",
    )
    parser.add_argument(
        "-a", "--ast", help="dump the AST in the 'filename'.ast", action="store_true"
    )
    parser.add_argument(
        "-s",
        "--sem",
        help="dump the decorated AST in the 'filename'.sem",
        action="store_true",
    )
    parser.add_argument(
        "-i", "--ir", help="dump the uCIR in the 'filename'.ir", action="store_true"
    )
    parser.add_argument("-n", "--no-run", help="do not execute the program", action="store_true")
    parser.add_argument(
        "-d", "--idb", help="run the interpreter in debug mode", action="store_true"
    )
    parser.add_argument(
        "-c",
        "--cfg",
        help="show the CFG for each function in pdf format",
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--opt",
        help="optimize the uCIR with const prop and dce",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="print in the stderr some data analysis informations",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--llvm",
        help="generate LLVM IR code in the 'filename'.ll",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--llvm-opt",
        choices=["ctm", "dce", "cfg", "all"],
        help="specify which llvm pass optimizations is enabled",
    )
    args = parser.parse_args()

    retval = Compiler(args).compile()
    sys.exit(retval)
