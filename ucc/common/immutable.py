from __future__ import annotations

import inspect
from dataclasses import Field, dataclass, field
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    dataclass_transform,
    final,
    overload,
)

T = TypeVar("T")


class _Bundle(dict[str, Any]):
    """A dictionary with keys that can be accessed like an attribute."""

    __slots__ = ("__orig_class__",)

    def __setattr__(self, name: str, value: Any, /) -> None:
        try:
            object.__setattr__(self, name, value)
        except AttributeError:
            self.__setitem__(name, value)

    def __getattr__(self, name: str, /) -> Any:
        return super().__getitem__(name)


def _compose_init(cls: type[T], pre_init: Callable[..., None], /):  # type: ignore
    init = cls.__init__

    @wraps(pre_init)
    def __init__(self: T, *args: Any, **kwargs: Any) -> None:
        # store modification in a bundle
        data = _Bundle()
        pre_init(data, *args, **kwargs)
        # then send the bundle to the dataclass
        init(self, **data)

    cls.__init__ = __init__  # type: ignore


# cache attribute yet to be initialized
_UNINITIALIZED = object()

# dataclass field for a cached attribute
_CACHED_FIELD = field(default=_UNINITIALIZED, init=False, repr=False, hash=False, compare=False)


def _create_cache_fields(cls: type, /) -> None:
    """Marks annotation and dataclasses Field for internal cache attributes."""
    cached_properties: list[str] = []

    annotations = dict(inspect.get_annotations(cls))
    for attrname, prop in cls.__dict__.items():
        if attrname not in annotations and isinstance(prop, immutable_property):
            return_type = inspect.signature(prop.func).return_annotation
            cache_name = prop.attrname

            annotations[cache_name] = return_type
            cached_properties.append(cache_name)

    cls.__annotations__ = annotations
    for cache_name in cached_properties:
        setattr(cls, cache_name, _CACHED_FIELD)


# a standard class decorator
Decorator: TypeAlias = Callable[[type[T]], type[T]]


@staticmethod
def _freeze(decorator: Decorator[T], cls: type[T], /) -> type[T]:
    """Turn cls into an immutable dataclass."""
    pre_init = cls.__dict__.get("__init__")
    if callable(pre_init):
        del cls.__init__

    _create_cache_fields(cls)
    decorated = decorator(cls)

    if callable(pre_init):
        _compose_init(decorated, pre_init)

    return final(decorated)


class immutable_property(Generic[T]):
    """Like functool's cached_property, but works with immutable classes."""

    def __init__(self, func: Callable[[Any], T], /):
        self.func = func
        self.__doc__ = func.__doc__
        self._attrname: str | None = None

    @classmethod
    @property
    def prefix(cls) -> str:
        return cls.__name__

    @property
    def attrname(self) -> str:
        if self._attrname is None:
            raise TypeError(f"cannot read unnamed {self.prefix}")

        return self._attrname

    @attrname.setter
    def attrname(self, name: str) -> None:
        if self._attrname is not None and self._attrname != name:
            raise TypeError(f"cannot set a new name for {self.prefix}")

        self._attrname = name

    def __set_name__(self, _owner: Any, name: str) -> None:
        self.attrname = f"__{self.prefix}__{name}__"

    # fmt: off
    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...
    @overload
    def __get__(self, instance: Any, owner: type) -> T: ...
    # fmt: on
    def __get__(self, instance: Any, _owner: type) -> Any:
        if instance is None:
            return self

        value = getattr(instance, self.attrname, _UNINITIALIZED)
        if value is _UNINITIALIZED:
            value = self.func(instance)
            object.__setattr__(instance, self.attrname, value)

        return value


# fmt: off
@overload
@dataclass_transform(field_specifiers=(Field[Any], field))
def immutable(cls: type[T], /) -> type[T]:
    ...
@overload
@dataclass_transform(field_specifiers=(Field[Any], field))
def immutable(*, eq: bool = True, order: bool = False, unsafe_hash: bool = False) -> Decorator[T]:
    ...
# fmt: on
@dataclass_transform(field_specifiers=(Field[Any], field))
def immutable(
    cls: type[T] | None = None,
    /,
    *,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
) -> type[T] | Decorator[T]:
    """Decorator to create an immutable dataclass."""

    decorator = dataclass(
        init=True, frozen=True, slots=True, eq=eq, order=order, unsafe_hash=unsafe_hash
    )
    if cls is not None:
        return _freeze(decorator, cls)
    else:
        return partial(_freeze, decorator)  # type: ignore
