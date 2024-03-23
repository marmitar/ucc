from __future__ import annotations

from typing import Final, Iterator, LiteralString, NoReturn, Protocol, TypeVar

from ply.lex import Lexer, LexError


class LexerError(LexError):
    ...
