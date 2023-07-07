from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Final, Literal, final


__version__: Final = '0.1.dev'


@final
@dataclass(frozen=True)
class _version_info:
    major: int
    minor: int
    patch: Literal['dev'] | int

    def __str__(self) -> str:
        return f'{self.major}.{self.minor}.{self.patch}'

    @staticmethod
    def parse(version_string: str) -> _version_info:
        match = re.fullmatch('(?P<major>\d+).(?P<minor>\d+).(?P<patch>dev|\d+)', version_string)
        assert match is not None

        major = int(match.group('major'))
        minor = int(match.group('minor'))
        patch = match.group('patch')

        if patch == 'dev':
            return _version_info(major, minor, patch='dev')
        else:
            return _version_info(major, minor, int(patch))


version_info: Final = _version_info.parse(__version__)
