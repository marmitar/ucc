from __future__ import annotations

from typing import Final, Iterator, LiteralString, NoReturn, Protocol, TypeVar

from ply.lex import Lexer, LexError


class Token(Protocol):
    """Strongly typed version of ply.lex.LexToken"""

    @property
    def value(self) -> str:
        ...

    @property
    def type(self) -> str:
        ...

    @property
    def lineno(self) -> int:
        ...

    @property
    def lexpos(self) -> int:
        ...

    @property
    def lexer(self) -> Lexer:
        ...


K = TypeVar("K", bound=LiteralString)


def frozen(*elements: K) -> frozenset[K]:
    """variadic constructor for a frozenset."""
    return frozenset(elements)


# Reserved keywords
keywords: Final = frozen(
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


def t_STRING_LITERAL(tok: Token) -> Token:
    r"[\"](\\.|.|\n)*?[\"]"
    tok.value = tok.value[1:-1]
    return tok


def t_ID(tok: Token) -> Token:
    r"[^\d\W]\w*"
    kind = tok.value.upper()
    if kind in keywords:
        tok.type = kind
    return tok


# newlines
def t_NEWLINE(tok: Token) -> None:
    r"\n+"
    tok.lexer.lineno += tok.value.count("\n")


def t_comment(tok: Token) -> None:
    r"(/\*(.|\n)*?\*/)|(//.*)"
    tok.lexer.lineno += tok.value.count("\n")


# errors
def _error(message: str, tok: Token) -> NoReturn:
    tok.lexer.skip(1)

    line = tok.lineno
    line_start = str(tok.lexer.lexdata).rfind("\n", 0, tok.lexpos)
    column = tok.lexpos - line_start
    raise LexError(f"{message} at {line}:{column}", tok.value)


def t_unterminated_string(tok: Token) -> NoReturn:
    r"\"(\\.|.|\n)*"
    # must come after 't_STRING_LITERAL'
    _error("Unterminated string", tok)


def t_unterminated_comment(tok: Token) -> NoReturn:
    r"/\*(.|\n)*"
    # must come after 't_comment'
    _error("Unterminated comment", tok)


def t_error(tok: Token) -> NoReturn:
    _error(f"Illegal character {tok.value[0]!r}", tok)


def __dir__() -> Iterator[LiteralString]:
    """ply.lex use 'dir' to resolve items in a module"""

    yield "__file__"
    yield "__package__"
    # token types
    yield "keywords"
    yield "tokens"
    # token regexes
    yield "t_ignore"
    yield "t_NEWLINE"
    yield "t_comment"
    yield "t_unterminated_string"
    yield "t_unterminated_comment"
    yield "t_error"
    for kind in tokens:
        if kind not in keywords:
            yield f"t_{kind}"
