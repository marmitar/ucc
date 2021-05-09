from __future__ import annotations
import sys
from typing import (
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    overload,
)


class Coord(Protocol):
    """Protocol for 'uc_parser.Coord'."""

    line: int
    column: Optional[int]


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
            result = obj.classname + "("
            indent += len(obj.classname) + 1
            attrs = []
            for name in obj.__slots__:
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

    def __init__(self, coord: Optional[Coord] = None):
        self.coord = coord

    def __repr__(self) -> str:
        """Generates a python representation of the current node"""
        return represent_node(self, 0)

    @classmethod
    @property
    def classname(cls) -> str:
        """Name for the Node specialized class"""
        if cls is Node:
            raise NotImplementedError("'Node' is an abstract base class")
        return cls.__name__

    def children(self) -> Iterable[Tuple[str, Node]]:
        """A sequence of all children that are Nodes"""
        nodelist = []
        for attr in self.__slots__:
            if attr in self.attr_names:
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

        buf.write(self.classname + ":")
        inner_offset += len(self.classname + ":")

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


class ArrayDecl(Node):
    __slots__ = "type", "size"
    attr_names = ()

    def __init__(self, size: Optional[Node], type: Optional[Node] = None):
        super().__init__()
        self.type = type
        self.size = size


class Decl(Node):
    __slots__ = "name", "type", "init"
    attr_names = ("name",)

    def __init__(
        self, name: Optional[ID], type: Node, init: Optional[Node], coord: Optional[Coord] = None
    ):
        super().__init__(coord)
        self.name = name
        self.type = type
        self.init = init


class DeclList(Node):
    __slots__ = ("decls",)
    attr_names = ()

    def __init__(self, decls: List[Decl] = [], coord: Optional[Coord] = None):
        super().__init__(coord)
        self.decls = tuple(decls)

    def show(self, *args, **kwargs):
        # only show when not empty
        if len(self.decls) > 0:
            super().show(*args, **kwargs)


class FuncDecl(Node):
    __slots__ = "params", "type"
    attr_names = ()

    def __init__(self, params: Optional[ParamList], type: Optional[Node] = None):
        super().__init__()
        self.type = type
        self.params = params


class FuncDef(Node):
    __slots__ = "return_type", "declaration", "decl_list", "implementation"
    attr_names = ()

    def __init__(
        self,
        return_type: Type,
        declaration: Decl,
        decl_list: DeclList,
        implementation: Optional[Node],
    ):
        super().__init__()
        self.return_type = return_type
        self.declaration = declaration
        self.decl_list = decl_list
        self.implementation = implementation


class GlobalDecl(Node):
    __slots__ = ("decl",)
    attr_names = ()

    def __init__(self, decl: List[Decl]):
        super().__init__()
        self.decl = tuple(decl)


class InitList(Node):
    __slots__ = ("init",)
    attr_names = ()

    def __init__(self, head: Node):
        super().__init__(head.coord)
        self.init = (head,)

    def append(self, node: Node) -> None:
        self.init += (node,)


class ParamList(Node):
    __slots__ = ("params",)
    attr_names = ()

    def __init__(self, head: Node):
        super().__init__(head.coord)
        self.params = (head,)

    def append(self, node: Node) -> None:
        self.params += (node,)


class Program(Node):
    __slots__ = ("gdecls",)
    attr_names = ()

    def __init__(self, gdecls: List[Node]):
        super().__init__()
        self.gdecls = tuple(gdecls)


class VarDecl(Node):
    __slots__ = "type", "declname"
    attr_names = ()

    def __init__(self, declname: ID, type: Optional[Type] = None):
        super().__init__()
        self.declname = declname
        self.type = type

    def children(self) -> Iterable[Tuple[str, Node]]:
        if self.type is not None:
            yield "type", self.type


# # # # # # # #
# STATEMENTS  #


class Assert(Node):
    __slots__ = ("param",)
    attr_names = ()

    def __init__(self, param: Node, coord: Coord):
        super().__init__(coord)
        self.param = param


class Break(Node):
    __slots__ = ()
    attr_names = ()

    def __init__(self, coord: Coord):
        super().__init__(coord)


class Compound(Node):
    __slots__ = "declarations", "statements"
    attr_names = ()

    def __init__(self, declarations: List[List[Node]], statements: List[Node], coord: Coord):
        super().__init__(coord)
        self.declarations = sum((tuple(d) for d in declarations), ())
        self.statements = tuple(statements)


class EmptyStatement(Node):
    pass


class For(Node):
    __slots__ = "declaration", "condition", "update", "stmt"
    attr_names = ()

    def __init__(
        self,
        declaration: Optional[Node],
        condition: Optional[Node],
        update: Optional[Node],
        stmt: Optional[Node],
        coord: Coord,
    ):
        super().__init__(coord)
        self.declaration = declaration
        self.condition = condition
        self.update = update
        self.stmt = stmt


class If(Node):
    __slots__ = "condition", "true_stmt", "false_stmt"
    attr_names = ()

    def __init__(
        self,
        condition: Node,
        true_stmt: Optional[Node],
        false_stmt: Optional[Node],
        coord: Coord,
    ):
        super().__init__(coord)
        self.condition = condition
        self.true_stmt = true_stmt
        self.false_stmt = false_stmt


class Print(Node):
    __slots__ = ("param",)
    attr_names = ()

    def __init__(self, param: Optional[Node], coord: Coord):
        super().__init__(coord)
        self.param = param


class Read(Node):
    __slots__ = ("param",)
    attr_names = ()

    def __init__(self, param: Node, coord: Coord):
        super().__init__(coord)
        self.param = param


class Return(Node):
    __slots__ = ("result",)
    attr_names = ()

    def __init__(self, result: Optional[Node], coord: Coord):
        super().__init__(coord)
        self.result = result


class While(Node):
    __slots__ = "expression", "stmt"
    attr_names = ()

    def __init__(self, expression: Node, stmt: Optional[Node], coord: Coord):
        super().__init__(coord)
        self.expression = expression
        self.stmt = stmt


# # # # # # # #
# EXPRESSIONS #


class ArrayRef(Node):
    __slots__ = "array", "index"
    attr_names = ()

    def __init__(self, array: Node, index: Node):
        super().__init__(array.coord)
        self.array = array
        self.index = index


class Assignment(Node):
    __slots__ = "op", "lvalue", "expr"
    attr_names = ("op",)

    def __init__(self, op: str, lvalue: Node, expr: Node):
        super().__init__(lvalue.coord)
        self.op = op
        self.lvalue = lvalue
        self.expr = expr


class BinaryOp(Node):
    __slots__ = "op", "left", "right"
    attr_names = ("op",)

    def __init__(self, op: str, left: Node, right: Node):
        super().__init__(left.coord)
        self.op = op
        self.left = left
        self.right = right


class ExprList(Node):
    __slots__ = ("expr",)
    attr_names = ()

    def __init__(self, head: Node):
        super().__init__(head.coord)
        self.expr = (head,)

    def append(self, expr: Node) -> None:
        self.expr += (expr,)

    def show(self, *args, **kwargs) -> None:
        # hide list when containing a single symbol
        if len(self.expr) < 2:
            self.expr[0].show(*args, **kwargs)
        else:
            super().show(*args, **kwargs)


class FuncCall(Node):
    __slots__ = "callable", "params"
    attr_names = ()

    def __init__(self, callable: Node, params: Optional[Node]):
        super().__init__(callable.coord)
        self.callable = callable
        self.params = params


class UnaryOp(Node):
    __slots__ = "op", "expr"
    attr_names = ("op",)

    def __init__(self, op: str, expr: Node):
        super().__init__(expr.coord)
        self.op = op
        self.expr = expr


# # # # # # # # #
# BASIC SYMBOLS #


class Constant(Node):
    __slots__ = "type", "value"
    attr_names = "type", "value"

    # fmt: off
    @overload
    def __init__(self, type: Literal["int"], value: int, coord: Coord): ...
    @overload
    def __init__(self, type: Literal["char", "string"], value: str, coord: Coord): ...
    # fmt: on
    def __init__(self, type: str, value: Union[int, str], coord: Coord):
        super().__init__(coord)
        self.type = type
        self.value = value


class ID(Node):
    __slots__ = ("name",)
    attr_names = ("name",)

    def __init__(self, name: str, coord: Coord):
        super().__init__(coord)
        self.name = name


class Type(Node):
    __slots__ = ("name",)
    attr_names = ("name",)

    def __init__(self, name: str, coord: Coord):
        super().__init__(coord)
        self.name = name
