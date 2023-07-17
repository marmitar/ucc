from __future__ import annotations

from types import UnionType
from typing import (
    Any,
    Callable,
    Final,
    Iterator,
    LiteralString,
    NoReturn,
    Protocol,
    TypeVar,
)

from ply.lex import Lexer, LexError  # type: ignore


class Token(Protocol):
    """Strongly typed version of ply.lex.LexToken"""

    value: str
    type: str
    lineno: int
    lexpos: int
    lexer: Lexer


_K = TypeVar("_K", bound=LiteralString)


def _frozen(*elements: _K) -> frozenset[_K]:
    """Variadic constructor for a frozenset."""
    return frozenset(elements)


# Reserved keywords
keywords: Final = _frozen(
    "ASSERT",
    "BOOL",
    "BREAK",
    "CHAR",
    "ELSE",
    "FALSE",
    "FLOAT",
    "FOR",
    "IF",
    "INT",
    "PRINT",
    "READ",
    "RETURN",
    "TRUE",
    "VOID",
    "WHILE",
)

# All the tokens recognized by the lexer
tokens: Final = (
    *keywords,
    # Identifiers
    "ID",
    # constants
    "INT_CONST",
    "STRING_LITERAL",
    "CHAR_CONST",
    "FLOAT_CONST",
    # delimiters
    "LPAREN",
    "RPAREN",
    "LBRACE",
    "RBRACE",
    "LBRACKET",
    "RBRACKET",
    # comparators
    "EQ",
    "NE",
    "LT",
    "GT",
    "LE",
    "GE",
    # operators
    "EQUALS",
    "PLUS",
    "MOD",
    "MINUS",
    "TIMES",
    "DIVIDE",
    "AND",
    "OR",
    "NOT",
    "ADDRESS",
    # punctuation
    "SEMI",
    "COMMA",
)

# # # # #
# Rules #

t_ignore = " \t"

# delimiters
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_LBRACE = r"\{"
t_RBRACE = r"\}"
t_LBRACKET = r"\["
t_RBRACKET = r"\]"

# comparators
t_EQ = r"=="
t_NE = r"!="
t_LT = r"<"
t_GT = r">"
t_LE = r"<="
t_GE = r">="

# operators
t_EQUALS = r"="
t_PLUS = r"\+"
t_MOD = r"%"
t_MINUS = r"-"
t_TIMES = r"\*"
t_DIVIDE = r"/"
t_AND = r"&&"
t_OR = r"\|\|"
t_NOT = r"!"
t_ADDRESS = r"&"

# punctuation
t_SEMI = r";"
t_COMMA = r","

# constants
t_INT_CONST = r"\d(\d|_\d)*"
t_CHAR_CONST = r"\'(\\.|.)+?\'"
t_FLOAT_CONST = r"""
    (
        (   # numbers like 123.
            \d(\d|_\d)*\.(\d(\d|_\d)*)?
        )|( # and .456
            (\d(\d|_\d)*)?\.\d(\d|_\d)*
        )   # can be 1.23e-2
        ((E|e)(\+|-)?\d(\d|_\d)*)?
    ) | (   # fractional isn't required with exponents
        \d(\d|_\d)(E|e)(\+|-)?\d(\d|_\d)
    )
"""


def t_STRING_LITERAL(tok: Token, /) -> Token:
    r"[\"](\\.|.|\n)*?[\"]"
    tok.value = tok.value[1:-1]
    return tok


def t_ID(tok: Token, /) -> Token:
    r"[^\d\W]\w*"
    kind = tok.value.upper()
    if kind in keywords:
        tok.type = kind
    return tok


# newlines
def t_NEWLINE(tok: Token, /) -> None:
    r"\n+"
    tok.lexer.lineno += tok.value.count("\n")


def t_comment(tok: Token, /) -> None:
    r"(/\*(.|\n)*?\*/)|(//.*)"
    tok.lexer.lineno += tok.value.count("\n")


# errors
def _error(message: str, tok: Token, /, *, skip: int) -> NoReturn:
    tok.lexer.skip(skip)  # type: ignore

    line = tok.lineno
    line_start = str(tok.lexer.lexdata).rfind("\n", 0, tok.lexpos)  # type: ignore
    column = tok.lexpos - line_start
    raise LexError(f"{message} at {line}:{column}", tok.value)


def t_unterminated_string(tok: Token, /) -> NoReturn:
    r"\"(\\.|.|\n)*"
    # must come after 't_STRING_LITERAL'
    _error("Unterminated string", tok, skip=len('"'))


def t_unterminated_comment(tok: Token, /) -> NoReturn:
    r"/\*(.|\n)*"
    # must come after 't_comment'
    _error("Unterminated comment", tok, skip=len("\\*"))


def t_error(tok: Token, /) -> NoReturn:
    char = tok.value[0]
    _error(f"Illegal character {char!r}", tok, skip=len(char))


def _global(name: _K, /, *, kind: type | UnionType = object) -> _K:
    """Verifies that variable 'name' is in the global scope and has type 'kind'."""
    try:
        item = globals()[name]
    except KeyError:
        raise KeyError(f"global variable {name!r} not found")

    if not isinstance(item, kind):
        raise TypeError(f"variable {name!r} is not of type {kind!r}")

    return name


def __dir__() -> Iterator[LiteralString]:
    """ply.lex use 'dir' to resolve items in a module"""

    yield _global("__file__", kind=str)
    yield _global("__package__", kind=str)
    # token types
    yield _global("keywords", kind=frozenset)
    yield _global("tokens", kind=tuple)
    # token regexes
    TokenMatcher = str | Callable[..., Any]
    yield _global("t_ignore", kind=TokenMatcher)
    yield _global("t_NEWLINE", kind=TokenMatcher)
    yield _global("t_comment", kind=TokenMatcher)
    yield _global("t_unterminated_string", kind=TokenMatcher)
    yield _global("t_unterminated_comment", kind=TokenMatcher)
    yield _global("t_error", kind=TokenMatcher)
    for kind in tokens:
        if kind not in keywords:
            yield _global(f"t_{kind}", kind=TokenMatcher)
