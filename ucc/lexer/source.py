from __future__ import annotations

import sys
from argparse import ArgumentTypeError
from os import PathLike
from pathlib import Path
from typing import SupportsInt, TypeVar, final, overload

from ..common import immutable

sys.stdin.name

Idx = TypeVar("Idx", bound=SupportsInt)


def _unwrap_span(span: range | slice | tuple[Idx, Idx], /) -> tuple[int, int]:
    match span:
        case range(start=start, stop=stop, step=1):
            return int(start), int(stop)
        case range():
            raise ValueError(f"invalid span {span!r}: step must be 1")
        case slice(start=start, stop=stop, step=None):
            return int(start), int(stop)
        case slice():
            raise ValueError(f"invalid span {span!r}: step must be None")
        case (start, stop):
            return int(start), int(stop)
        case tuple():
            raise ValueError(f"invalid span {span!r}: too many values")
        case _:
            raise ValueError(f"invalid span {span!r}")


@immutable()
class Source:
    filename: Path | str
    contents: str

    def resolve_line_column(self, position: int, /) -> tuple[int, int]:
        last_newline = self.contents.rfind("\n", 0, position)
        line = self.contents.count("\n", 0, last_newline)
        column = position - last_newline
        return line, column


@final
@immutable()
class SourceFile:
    filename: Path
    contents: str

    def __init__(self, filename: str | PathLike[str], /) -> None:
        pass

    def resolve_line_column(self, position: int, /) -> tuple[int, int]:
        last_newline = self.contents.rfind("\n", 0, position)
        line = self.contents.count("\n", 0, last_newline)
        column = position - last_newline
        return line, column

    @staticmethod
    def load(input_path: str | Path, /) -> SourceFile:
        """Resolves a source file from its path."""

        # input_path must resolve to an absolute path
        try:
            filename = Path(input_path).resolve(strict=False)
        except ValueError as error:  # invalid paths like '\0'
            raise ValueError(f"invalid path {str(input_path)!r}: {error}")

        # filename, when resolved, must point to a regular file
        if not filename.is_file():
            raise ValueError(f"source {str(input_path)!r} is not a regular file")

        # file must be valid utf8 text
        try:
            contents = filename.read_text(encoding="utf8", errors="strict")
        except ValueError as error:
            raise ValueError(f"source file {str(input_path)!r} is not UTF-8")

        # finally we have a valid source file
        return SourceFile(filename, contents)

    @staticmethod
    def load_argument(input_path: str, /) -> SourceFile:
        """Loads a file, but turn errors into ArgumentTypeError."""
        try:
            return SourceFile.load(input_path)
        except Exception as error:
            raise ArgumentTypeError(f"{error}")

    def __len__(self) -> int:
        return len(self.contents)

    def resolve(self, index: Idx, /) -> int:
        position = int(index)
        if position not in range(len(self)):
            raise IndexError(f"position {index} outside range")
        else:
            return position

    def resolve_span(self, span: range | slice | tuple[Idx, Idx], /) -> tuple[int, int]:
        try:
            start, stop = _unwrap_span(span)
        except ValueError as error:
            raise IndexError(str(error))

        try:
            start = self.resolve(start)
            stop = self.resolve(stop)
        except IndexError as error:
            raise IndexError(f"invalid {span!r}: {error}")

        return start, stop

    def __getitem__(self, span: range | slice | tuple[Idx, Idx], /) -> str:
        start, stop = self.resolve_span(span)
        return self.contents[start:stop]


@final
@immutable()
class Coord:
    line: int
    column: int
    position: int

    @staticmethod
    def resolve(file: SourceFile, /, position: Idx) -> Coord:
        pos = file.resolve(position)

        line_start = file.contents.rfind("\n", 0, pos) + 1
        line = file.contents.count("\n", 0, line_start)
        column = pos - line_start
        return Coord(line, column, pos)

    def __int__(self) -> int:
        return self.position


@final
@immutable()
class Span:
    file: SourceFile
    start: Coord
    end: Coord

    @property
    def text(self) -> str:
        return self.file[self.start : self.end]

    @staticmethod
    def resolve(file: SourceFile, /, span: range | slice | tuple[Idx, Idx]) -> Span:
        start_pos, stop_pos = file.resolve_span(span)

        start = Coord.resolve(file, position=start_pos)
        end = Coord.resolve(file, position=stop_pos)
        return Span(file, start, end)
