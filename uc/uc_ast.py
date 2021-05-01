from __future__ import annotations
import sys
from typing import List, Literal, Optional, Protocol, Sequence, Tuple, Union, overload


class Coordinates(Protocol):
    """Protocol for 'uc_parser.Coord'."""

    line: int
    column: Optional[int]


Coord = Optional[Coordinates]


def represent_node(obj, indent):
    def _repr(obj, indent, printed_set):
        """
        Get the representation of an object, with dedicated pprint-like format for lists.
        """
        if isinstance(obj, list):
            indent += 1
            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            return "[" + (sep.join((_repr(e, indent, printed_set) for e in obj))) + final_sep + "]"
        elif isinstance(obj, Node):
            if obj in printed_set:
                return ""
            else:
                printed_set.add(obj)
            result = obj.__class__.__name__ + "("
            indent += len(obj.__class__.__name__) + 1
            attrs = []
            for name in obj.__slots__[:-1]:
                if name == "bind":
                    continue
                value = getattr(obj, name)
                value_str = _repr(value, indent + len(name) + 1, printed_set)
                attrs.append(name + "=" + value_str)
            sep = ",\n" + (" " * indent)
            final_sep = ",\n" + (" " * (indent - 1))
            result += sep.join(attrs)
            result += ")"
            return result
        elif isinstance(obj, str):
            return obj
        else:
            return repr(obj)

    # avoid infinite recursion with printed_set
    printed_set = set()
    return _repr(obj, indent, printed_set)


class Node:
    """Abstract base class for AST nodes."""

    __slots__ = ("coord",)
    attr_names: Sequence[str] = ()

    def __init__(self, coord: Coord = None):
        self.coord = coord

    def __repr__(self) -> str:
        """Generates a python representation of the current node"""
        return represent_node(self, 0)

    def children(self) -> Sequence[Tuple[str, Node]]:
        """A sequence of all children that are Nodes"""
        nodelist = []
        for attr in self.__slots__:
            if attr in Node.__slots__ or attr in self.attr_names:
                continue
            # treat attributes not in attr_names as children
            value = getattr(self, attr)
            if isinstance(value, Sequence):
                for i, child in enumerate(value):
                    name = f"{attr}[{i}]"
                    nodelist.append((name, child))
            elif value is not None:
                nodelist.append((attr, value))
        return tuple(nodelist)

    def show(
        self,
        buf=sys.stdout,
        offset=0,
        attrnames=False,
        nodenames=False,
        showcoord=False,
        _my_node_name: Optional[str] = None,
    ) -> None:
        """Pretty print the Node and all its attributes and children (recursively) to a buffer.
        buf:
            Open IO buffer into which the Node is printed.
        offset:
            Initial offset (amount of leading spaces)
        attrnames:
            True if you want to see the attribute names in name=value pairs. False to only see the values.
        nodenames:
            True if you want to see the actual node names within their parents.
        showcoord:
            Do you want the coordinates of each Node to be displayed.
        """
        lead = " " * offset
        buf.write(lead)
        inner_offset = 0

        if nodenames and _my_node_name is not None:
            buf.write(f"<{_my_node_name}> ")
            inner_offset += len(f"<{_my_node_name}> ")

        buf.write(self.__class__.__name__ + ":")
        inner_offset += len(self.__class__.__name__ + ":")

        if self.attr_names:
            if attrnames:
                nvlist = [
                    (n, represent_node(getattr(self, n), offset + inner_offset + 1 + len(n) + 1))
                    for n in self.attr_names
                    if getattr(self, n) is not None
                ]
                attrstr = ", ".join(f"{n}={v}" for n, v in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attr_names]
                attrstr = ", ".join(represent_node(v, offset + inner_offset + 1) for v in vlist)
            buf.write(" " + attrstr)

        if showcoord:
            if self.coord and self.coord.line != 0:
                buf.write(" %s" % self.coord)
        buf.write("\n")

        for (child_name, child) in self.children():
            child.show(buf, offset + 4, attrnames, nodenames, showcoord, child_name)


# # # # # # # # #
# DECLARATIONS  #


class ArrayDecl:
    pass


class Decl:
    pass


class DeclList:
    pass


class FuncDecl:
    pass


class FuncDef:
    pass


class GlobalDecl:
    pass


class InitList:
    pass


class ParamList:
    pass


class Program(Node):
    __slots__ = "gdecls", "coord"
    attr_names = ()

    def __init__(self, gdecls: List[GlobalDecl], coord: Coord = None):
        super().__init__(coord)
        self.gdecls = gdecls


class VarDecl:
    pass


# # # # # # # #
# STATEMENTS  #


class Assert:
    pass


class Break:
    pass


class Compound:
    pass


class EmptyStatement:
    pass


class For:
    pass


class If:
    pass


class Print:
    pass


class Read:
    pass


class Return:
    pass


class While:
    pass


# # # # # # # #
# EXPRESSIONS #


class ArrayRef:
    pass


class Assignment:
    pass


class BinaryOp(Node):
    __slots__ = "op", "lvalue", "rvalue", "coord"
    attr_names = ("op",)

    def __init__(self, op: str, left: Node, right: Node, coord: Coordinates):
        super().__init__(coord)
        self.op = op
        self.lvalue = left
        self.rvalue = right


class ExprList:
    pass


class FuncCall:
    pass


# # # # # # # # #
# BASIC SYMBOLS #


class Constant(Node):
    __slots__ = "type", "value", "coord"
    attr_names = "type", "value"

    # fmt: off
    @overload
    def __init__(self, type: Literal["int"], value: int, coord: Coordinates): ...
    @overload
    def __init__(self, type: Literal["char", "string"], value: str, coord: Coordinates): ...
    # fmt: on
    def __init__(self, type: str, value: Union[int, str], coord: Coordinates):
        super().__init__(coord)
        self.type = type
        self.value = value


class ID(Node):
    __slots__ = "name", "coord"
    attr_names = ("name",)

    def __init__(self, name: str, coord: Coordinates):
        super().__init__(coord)
        self.name = name


class Type(Node):
    __slots__ = "name", "coord"
    attr_names = ("name",)

    def __init__(self, name: str, coord: Coordinates):
        super().__init__(coord)
        self.name = name


class UnaryOp(Node):
    __slots__ = "op", "coord"
    attr_names = ("op",)

    def __init__(self, op: str, coord: Coordinates):
        super().__init__(coord)
        self.op = op
