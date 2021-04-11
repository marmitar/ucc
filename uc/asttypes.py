from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Optional, Tuple, Union
from ply.yacc import YaccProduction


def set_terminal_lineinfo(p: YaccProduction, sym: int = 1) -> Tuple[int, int]:
    line, index = p.lineno(sym), p.lexpos(sym)
    p.set_lineno(0, line)
    p.set_lexpos(0, index)
    return line, index


# # # # # # # # #
# DECLARATIONS  #


class Initializer:
    ...


class Declarator:
    ...


@dataclass(frozen=True)
class ArrayDeclarator(Declarator):
    declarator: Declarator
    expression: Optional[Expression]


@dataclass(frozen=True)
class InitDeclarator:
    declarator: Declarator
    initializer: Optional[Initializer]


class ArrayInit(List[Initializer], Initializer):
    ...


@dataclass(frozen=True)
class Declaration:
    specifier: TypeSpec
    init: List[InitDeclarator]


@dataclass(frozen=True)
class Parameter:
    specifier: TypeSpec
    declaration: Declarator


@dataclass(frozen=True)
class FunctionDeclarator(Declarator):
    declarator: Declarator
    parameters: List[Parameter]


@dataclass(frozen=True)
class FunctionDef:
    type_spec: TypeSpec
    declarator: Declarator
    declarations: List[Declaration]
    body: CompoundStmt


class Program(List[Union[FunctionDef, Declaration]]):
    ...


# # # # # # # #
# STATEMENTS  #


class Statement:
    def set_lineinfo(self, p: YaccProduction) -> Statement:
        set_terminal_lineinfo(p)
        return self


@dataclass(frozen=True)
class CompoundStmt(Statement):
    declarations: List[Declaration]
    statements: List[Statement]


@dataclass(frozen=True)
class IfStmt(Statement):
    condition: Expression
    true_branch: Statement
    else_branch: Optional[Statement]


@dataclass(frozen=True)
class WhileStmt(Statement):
    condition: Expression
    body: Statement


@dataclass(frozen=True)
class ForStmt(Statement):
    initialization: Union[None, Expression, Declaration]
    condition: Optional[Expression]
    update: Optional[Expression]
    body: Statement


@dataclass(frozen=True)
class ExprStmt(Statement):
    expr: Optional[Expression]


@dataclass(frozen=True)
class BreakStmt(Statement):
    ...


@dataclass(frozen=True)
class ReturnStmt(Statement):
    expr: Optional[Expression]


@dataclass(frozen=True)
class AssertStmt(Statement):
    expr: Expression


@dataclass(frozen=True)
class PrintStmt(Statement):
    expr: Optional[Expression]


@dataclass(frozen=True)
class ReadStmt(Statement):
    args: List[Expression]


# # # # # # # #
# EXPRESSIONS #


class Expression(Initializer):
    ...


@dataclass(frozen=True)
class AssignExpr(Expression):
    item: Expression
    value: Expression


@dataclass(frozen=True)
class CallExpr(Expression):
    item: Expression
    args: List[Expression]


@dataclass(frozen=True)
class AccessExpr(Expression):
    item: Expression
    at: Expression


@dataclass(frozen=True)
class UnOp(Expression):
    op: Operator
    item: Expression


@dataclass(frozen=True)
class BinOp(Expression):
    op: Operator
    left: Expression
    right: Expression


# # # # # # # # # # #
# TERMINAL  SYMBOLS #


class TerminalSymbol(Expression):
    @classmethod
    def from_token(cls, p: YaccProduction) -> TerminalSymbol:
        info = set_terminal_lineinfo(p)
        return cls(cls.parse(p[1]), info)

    @classmethod
    @property
    def symbol(cls) -> str:
        return cls.__name__

    @classmethod
    def parse(cls, text):
        return text

    def __str__(self) -> str:
        return str(self.value)

    def __value__(self):
        return self.value

    def __repr__(self) -> str:
        value = self.__value__()
        return f"{self.symbol}::{value}"


@unique
class TypeSpec(TerminalSymbol, Enum):
    VOID = "void"
    CHAR = "char"
    INT = "int"

    @classmethod
    def from_token(cls, p: YaccProduction) -> TypeSpec:
        set_terminal_lineinfo(p)
        return cls(p[1])


@dataclass(frozen=True, repr=False)
class Int(TerminalSymbol):
    value: int
    position: Tuple[int, int]

    @classmethod
    def __parse__(cls, text: str) -> int:
        return int(text)


@dataclass(frozen=True, repr=False)
class Char(TerminalSymbol):
    value: str
    position: Tuple[int, int]

    @classmethod
    def __parse__(cls, text: str) -> str:
        return text.strip("'")

    def __value__(self) -> str:
        return f"'{self.value}'"


@dataclass(frozen=True, repr=False)
class Ident(TerminalSymbol, Declarator):
    value: str
    position: Tuple[int, int]


@dataclass(frozen=True, repr=False)
class String(TerminalSymbol):
    value: str
    position: Tuple[int, int]

    def __value__(self) -> str:
        return repr(self.value)


@dataclass(frozen=True, repr=False)
class Operator(TerminalSymbol):
    value: str

    @classmethod
    def from_token(cls, p: YaccProduction, *, set_info=True) -> TypeSpec:
        if set_info:
            set_terminal_lineinfo(p)
        return cls(p[1])
