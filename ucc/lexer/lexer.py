from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Iterator, NoReturn, Self, TextIO, final

from ply.lex import Lexer, LexError, LexToken, lex

from . import tokens


@final
@dataclass(frozen=True, slots=True)
class TokenStream(Iterator[LexToken]):
    lexer: Lexer
    handle_error: Callable[[LexError], None] = field(kw_only=True)

    def next_token(self) -> LexToken | None:
        tok = self.lexer.token()
        if isinstance(tok, LexToken):
            return tok
        else:
            return None

    def token(self) -> LexToken | None:
        while True:
            try:
                return self.next_token()
            except LexError as error:
                self.handle_error(error)

    def __next__(self) -> LexToken:
        if (tok := self.token()) is not None:
            return tok
        else:
            raise StopIteration()

    def __iter__(self) -> Self:
        return self


def abort_on_lex_error(error: LexError) -> NoReturn:
    print_lex_error(error)
    sys.exit(1)


def ignore_lex_error(error: LexError) -> NoReturn:
    pass


def print_lex_error(error: LexError, *, file: TextIO = sys.stdout):
    message = str(error.args[0])
    print("LexerError:", message, file=file, flush=True)


def reraise_lex_error(error: LexError) -> NoReturn:
    raise error


@final
@dataclass(frozen=True, slots=True)
class UCLexer:
    ABORT: ClassVar = abort_on_lex_error
    IGNORE: ClassVar = ignore_lex_error
    PRINT: ClassVar = print_lex_error
    RERAISE: ClassVar = reraise_lex_error

    on_error: Callable[[LexError], None] = field(default=RERAISE, kw_only=True)

    @property
    def tokens(self):
        return tokens.tokens

    def lexer(self) -> Lexer:
        return lex(module=tokens, optimize=False, reflags=re.VERBOSE)

    def tokenize(self, text: str) -> TokenStream:
        lexer = self.lexer()
        lexer.input(text)
        return TokenStream(lexer, handle_error=self.on_error)
