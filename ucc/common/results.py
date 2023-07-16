from __future__ import annotations

from typing import Iterable, Iterator, Self, TypeVar, final

from result import Err, Ok, Result

from .immutable import immutable

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)


@final
@immutable()
class ResultIterator(Iterator[Result[T, E]]):
    """Adapts an iterator to catch errors of type E while iterating."""

    items: Iterator[T]
    ErrorType: type[E]

    def __init__(self, items: Iterable[T], /, *, ErrorType: type[E] = BaseException) -> None:
        self.items = iter(items)
        self.ErrorType = ErrorType

    def __next__(self) -> Result[T, E]:
        try:
            value = self.items.__next__()
            return Ok(value)
        except StopIteration:
            raise
        except self.ErrorType as error:
            return Err(error)

    def __iter__(self) -> Self:
        return self


def results(it: Iterable[T], /, *, ErrorType: type[E] = BaseException) -> ResultIterator[T, E]:
    """Catches errors of type E while iterating."""
    return ResultIterator(it, ErrorType=ErrorType)
