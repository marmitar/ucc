from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar, Iterator, Self, final

from ply.lex import Lexer, LexError, LexToken, lex

from . import tokens


@final
@dataclass(frozen=True, slots=True)
class TokenStream(Iterator[LexToken]):
    lexer: Lexer

    def next_token(self) -> LexToken | None:
        tok = self.lexer.token()
        if isinstance(tok, LexToken):
            return tok
        else:
            return None

    def __next__(self) -> LexToken:
        if (tok := self.next_token()) is not None:
            return tok
        else:
            raise StopIteration()

    def __iter__(self) -> Self:
        return self


@final
@dataclass(frozen=True, slots=True)
class UCLexer:
    Error: ClassVar = LexError

    @property
    def tokens(self):
        return tokens.tokens

    def lexer(self) -> Lexer:
        return lex(module=tokens, optimize=False, reflags=re.VERBOSE)

    def tokenize(self, text: str) -> TokenStream:
        lexer = self.lexer()
        lexer.input(text)
        return TokenStream(lexer)
