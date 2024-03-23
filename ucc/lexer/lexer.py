from __future__ import annotations

import re
from typing import ClassVar, Iterator, Self, final

from ply.lex import Lexer, LexError, LexToken, lex  # type: ignore

from ..common import immutable
from . import tokens

# @final
# @dataclass
# class UCToken:
#     value: str
#     type: str
#     lexpos: int
#     coord: ...

#     @staticmethod
#     def of(tok: tokens.Token) -> UCToken:
#         line = tok.lineno
#         line_start = str(tok.lexer.lexdata).rfind("\n", 0, tok.lexpos)
#         column = tok.lexpos - line_start

#         UCToken(str(tok.value), str(tok.type), int(tok.lexpos), ...)


@final
@immutable
class TokenStream(Iterator[LexToken]):
    lexer: Lexer

    def next_token(self) -> LexToken | None:
        tok = self.lexer.token()  # type: ignore
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
@immutable
class UCLexer:
    Error: ClassVar = LexError

    @property
    def tokens(self):
        return tokens.tokens

    def lexer(self) -> Lexer:
        return lex(module=tokens, optimize=False, reflags=re.VERBOSE)  # type: ignore

    def tokenize(self, text: str) -> TokenStream:
        lexer = self.lexer()
        lexer.input(text)  # type: ignore
        return TokenStream(lexer)
