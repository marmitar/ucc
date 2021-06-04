from __future__ import annotations
import inspect
import sys
from collections.abc import Sequence
from typing import Any, Literal, Optional, Protocol, TextIO, Tuple, Union
from uc.uc_type import ArrayType, FunctionType, PointerType, PrimaryType, uCType


class Coord(Protocol):
    """Protocol for 'uc_parser.Coord'."""

    line: int
    column: Optional[int]


def represent_node(obj, indent: int) -> str:
    def _repr(obj, indent: int, printed_set: set) -> str:
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
            for name in obj.attributes():
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

    __slots__ = "coord", "uc_type"
    attr_names: Tuple[str, ...] = ("uc_type",)
    special_attr: Tuple[str, ...] = ("coord",)

    uc_type: uCType

    def __init__(self, coord: Optional[Coord] = None):
        self.coord = coord
        self.uc_type = None

    def __repr__(self) -> str:
        """Generates a python representation of the current node"""
        return represent_node(self, 0)

    @property
    def classname(self) -> str:
        """Name for the specialized Node class"""
        if self.__class__ is Node:
            raise NotImplementedError("'Node' is an abstract base class")
        return self.__class__.__name__

    @classmethod
    def _get_tuple_attrs(cls, attr_name: str) -> Tuple[str, ...]:
        """Get class and parent attributes with 'attr_name'."""
        attr = []
        for base_class in inspect.getmro(cls):
            attr.extend(getattr(base_class, attr_name, ()))
        return tuple(attr)

    @classmethod
    def attributes(cls) -> Tuple[str, ...]:
        """Names of predefined node attributes."""
        return cls._get_tuple_attrs("attr_names")

    def children(self) -> Tuple[Tuple[str, Node], ...]:
        """A sequence of all children that are Nodes"""
        attr_names = frozenset(self.attributes())
        special = frozenset(self._get_tuple_attrs("special_attr"))

        nodelist = []
        for attr in self._get_tuple_attrs("__slots__"):
            if attr in attr_names or attr in special:
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
        buf: TextIO = sys.stdout,
        offset: int = 0,
        attrnames: bool = False,
        nodenames: bool = False,
        showcoord: bool = False,
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

        if self.attributes():
            if attrnames:
                nvlist = [
                    (n, represent_node(getattr(self, n), offset + inner_offset + 1 + len(n) + 1))
                    for n in self.attributes()
                    if getattr(self, n, None) is not None
                ]
                attrstr = ", ".join(f"{n}={v}" for n, v in nvlist)
            else:
                vlist = [getattr(self, n) for n in self.attributes()]
                attrstr = ", ".join(represent_node(v, offset + inner_offset + 1) for v in vlist)
            buf.write(" " + attrstr)

        if showcoord:
            if self.coord and self.coord.line != 0:
                buf.write(" %s" % self.coord)
        buf.write("\n")

        for child_name, child in self.children():
            child.show(buf, offset + 4, attrnames, nodenames, showcoord, child_name)

    def lvalue_name(self) -> Optional[ID]:
        """Return node lvalue name, if it is an lvalue."""
        return None  # default


# # # # # # # # #
# DECLARATIONS  #


class Program(Node):
    __slots__ = "gdecls", "name"
    attr_names = ()
    special_attr = ("name",)

    def __init__(self, gdecls: list[Union[GlobalDecl, FuncDef]]):
        super().__init__()
        self.gdecls = tuple(gdecls)
        self.name: Optional[str] = None


class DeclList(Node):
    __slots__ = ("decls",)
    attr_names = ()

    def __init__(self, decls: Sequence[Decl] = []):
        super().__init__(decls[0].coord if decls else None)
        self.decls = tuple(decls)

    def extend(self, decl: DeclList) -> None:
        if not self.decls:
            self.coord = decl.coord
        self.decls += tuple(decl.decls)

    def append(self, decl: Decl) -> None:
        # does not update 'coord'
        self.decls += (decl,)

    def remove(self, decl: Decl) -> None:
        for i, d in enumerate(self.decls):
            if d is decl:
                self.decls = self.decls[:i] + self.decls[i + 1 :]
                return  # avoid index problem

    def show(self, *args, **kwargs):
        # only show when not empty
        if len(self.decls) > 0:
            super().show(*args, **kwargs)


class GlobalDecl(DeclList):
    __slots__ = ()
    attr_names = ()

    def __init__(self, decl: DeclList):
        super().__init__(decl.decls)


class FuncDef(Node):
    __slots__ = (
        "return_type",
        "declaration",
        "decl_list",
        "implementation",
        "return_list",
    )
    attr_names = ()
    special_attr = ("return_list",)

    def __init__(
        self,
        return_type: TypeSpec,
        declaration: Decl,
        decl_list: DeclList,
        implementation: Compound,
    ):
        super().__init__(declaration.coord)
        self.return_type = return_type
        self.declaration = declaration
        self.decl_list = decl_list
        self.implementation = implementation
        self.return_list: Tuple[Return, ...] = ()

    def add_return(self, node: Return) -> None:
        self.return_list += (node,)


class ParamList(Node):
    __slots__ = ("params",)
    attr_names = ()

    def __init__(self, head: Optional[Decl] = None):
        super().__init__()
        self.params: Tuple[Decl, ...] = ()
        if head is not None:
            self.append(head)

    def append(self, node: Decl) -> None:
        self.params += (node,)


class InitList(Node):
    __slots__ = ("init",)
    attr_names = ()

    uc_type: ArrayType

    def __init__(self, head: Optional[Node] = None):
        super().__init__()
        self.init: Tuple[Node, ...] = ()
        self.append(head)

    def append(self, node: Optional[Node]) -> None:
        if node is not None:
            if len(self) == 0:
                self.coord = node.coord
            self.init += (node,)

    def __len__(self) -> int:
        return len(self.init)


class Decl(Node):
    __slots__ = "name", "type", "init"
    attr_names = ("name",)

    name: ID

    def __init__(self, type: Union[ArrayDecl, FuncDecl, VarDecl], init: Optional[Node]):
        super().__init__(type.coord)
        self.type = type
        self.init = init
        if isinstance(init, InitList):
            init.coord = type.coord


class Modifier(Node):
    __slots__ = ("type", "_declname")
    attr_names = ()
    special_attr = ("_declname",)

    type: Union[Modifier, VarDecl]

    def __init__(self):
        super().__init__()
        self.type = None
        self._declname: Optional[ID] = None

    def set_type(self, type: Union[Modifier, VarDecl]) -> None:
        self.type = type
        self.coord = type.coord

    @property
    def declname(self) -> ID:
        if self._declname is None:
            modtype = self
            while not isinstance(modtype, VarDecl):
                modtype = modtype.type
            self._declname = modtype.declname
        return self._declname


class ArrayDecl(Modifier):
    __slots__ = ("size",)
    attr_names = ()

    uc_type: ArrayType

    def __init__(self, size: Optional[Node]):
        super().__init__()
        self.size = size


class FuncDecl(Modifier):
    __slots__ = ("param_list",)
    attr_names = ()

    uc_type: FunctionType

    def __init__(self, params: ParamList):
        super().__init__()
        self.param_list = params


class PointerDecl(Modifier):
    __slots__ = ()
    attr_names = ()

    uc_type: PointerType


class VarDecl(Node):
    __slots__ = "type", "declname"
    attr_names = ()
    special_attr = ("declname",)

    type: TypeSpec
    uc_type: PrimaryType

    def __init__(self, declname: ID):
        super().__init__(declname.coord)
        self.declname = declname
        self.type = None


# # # # # # # #
# STATEMENTS  #


class Assert(Node):
    __slots__ = ("param",)
    attr_names = ()

    def __init__(self, param: ExprList, coord: Coord):
        super().__init__(coord)
        self.param = param


class Break(Node):
    __slots__ = ("iteration",)
    attr_names = ()
    special_attr = ("iteration",)

    iteration: IterationStmt

    def __init__(self, coord: Coord):
        super().__init__(coord)

    def bind(self, iteration_stmt: IterationStmt) -> None:
        iteration_stmt.add_break(self)
        self.iteration = iteration_stmt


class Compound(Node):
    __slots__ = "declarations", "statements"
    attr_names = ()

    def __init__(self, declarations: DeclList, statements: list[Node], coord: Coord):
        super().__init__(coord)
        self.declarations = declarations
        self.statements = tuple(statements)

    def last_statement(self) -> Optional[Node]:
        return self.statements[-1] if self.statements else None

    def append(self, statement: Node) -> None:
        self.statements += (statement,)


class EmptyStatement(Node):
    pass


class IterationStmt(Node):
    __slots__ = "condition", "body", "break_locations"
    attr_names = ()
    special_attr = ("break_locations",)

    child_order: Tuple[str, ...]

    def __init__(self, condition: Optional[ExprList], body: Optional[Node], coord: Coord):
        super().__init__(coord)
        self.condition = condition
        self.body = body
        self.break_locations: Tuple[Break, ...] = ()

    def add_break(self, node: Break) -> None:
        self.break_locations += (node,)

    def children(self) -> Tuple[Tuple[str, Node], ...]:
        child_list = []
        # follow child order
        for attr in self.child_order:
            child = getattr(self, attr, None)
            if child is not None:
                child_list.append((attr, child))
        return tuple(child_list)


class For(IterationStmt):
    __slots__ = "declaration", "update"
    attr_names = ()

    child_order = "declaration", "condition", "update", "body"

    def __init__(
        self,
        declaration: Union[DeclList, Optional[ExprList]],
        condition: Optional[ExprList],
        update: Optional[ExprList],
        body: Optional[Node],
        coord: Coord,
    ):
        super().__init__(condition, body, coord)
        self.declaration = declaration
        self.update = update


class While(IterationStmt):
    __slots__ = ()
    attr_names = ()

    child_order = "condition", "body"

    condition: ExprList

    def __init__(self, condition: ExprList, body: Optional[Node], coord: Coord):
        super().__init__(condition, body, coord)


class If(Node):
    __slots__ = "condition", "true_stmt", "false_stmt"
    attr_names = ()

    def __init__(
        self,
        condition: ExprList,
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

    def __init__(self, param: Optional[ExprList], coord: Coord):
        super().__init__(coord)
        self.param = param


class Read(Node):
    __slots__ = ("param",)
    attr_names = ()

    def __init__(self, param: ExprList, coord: Coord):
        super().__init__(coord)
        self.param = param


class Return(Node):
    __slots__ = "result", "function"
    attr_names = ()
    special_attr = ("function",)

    function: FuncDef

    def __init__(self, result: Optional[ExprList], coord: Coord):
        super().__init__(coord)
        self.result = result

    def bind(self, function: FuncDef) -> None:
        function.add_return(self)
        self.function = function


# # # # # # # #
# EXPRESSIONS #


class ArrayRef(Node):
    __slots__ = "array", "index"
    attr_names = ()

    def __init__(self, array: Node, index: Node):
        super().__init__(array.coord)
        self.array = array
        self.index = index

    def lvalue_name(self) -> Optional[ID]:
        # use array name, when referencing
        return self.array.lvalue_name()


class BinaryOp(Node):
    __slots__ = "op", "left", "right"
    attr_names = ("op",)

    def __init__(self, op: str, left: Node, right: Node):
        super().__init__(left.coord)
        self.op = op
        self.left = left
        self.right = right


class Assignment(BinaryOp):
    __slots__ = ()
    attr_names = ()


class ExprList(Node):
    __slots__ = ("expr",)
    attr_names = ()

    def __init__(self, head: Node, *rest: Node):
        super().__init__(head.coord)
        self.expr = (head,) + rest

    def append(self, *expr: Node) -> None:
        self.expr += expr

    def as_comma_op(self) -> Node:
        """To evaluate as the comma operator (e.g. 'while (a++, i < 10)')"""
        return self.expr[-1]  # always has an element

    def show(self, *args, **kwargs) -> None:
        # hide list when containing a single symbol
        if len(self.expr) == 1:
            self.expr[0].show(*args, **kwargs)
        else:
            super().show(*args, **kwargs)

    def lvalue_name(self) -> Optional[ID]:
        return self.as_comma_op().lvalue_name()


class FuncCall(Node):
    __slots__ = "callable", "params"
    attr_names = ()

    def __init__(self, callable: Node, params: Optional[ExprList] = None):
        super().__init__(callable.coord)
        self.callable = callable
        self.params = params

    def parameters(self) -> Tuple[Node, ...]:
        if self.params:
            return self.params.expr
        else:
            return ()


class RelationOp(BinaryOp):
    __slots__ = ()
    attr_names = ()


class UnaryOp(Node):
    __slots__ = "op", "expr"
    attr_names = ("op",)

    def __init__(self, op: str, expr: Node):
        super().__init__(expr.coord)
        self.op = op
        self.expr = expr

    def lvalue_name(self) -> Optional[ID]:
        return self.expr.lvalue_name()


class AddressOp(UnaryOp):
    __slots__ = ()
    attr_names = ()

    uc_type: PointerType

    def __init__(self, op: Literal["&", "*"], expr: Node, coord: Coord):
        super().__init__(op, expr)
        self.coord = coord


# # # # # # # # #
# BASIC SYMBOLS #


class Constant(Node):
    __slots__ = ("value",)
    attr_names = ("value",)

    value: Any


class StringConstant(Constant):
    __slots__ = ()
    attr_names = ()

    value: str

    def __init__(self, value: str, coord: Coord):
        super().__init__(coord)
        self.value = value


class IntConstant(Constant):
    __slots__ = ()
    attr_names = ()

    value: int

    def __init__(self, value: int, coord: Coord):
        super().__init__(coord)
        self.value = value


class FloatConstant(Constant):
    __slots__ = ()
    attr_names = ()

    value: float

    def __init__(self, value: float, coord: Coord):
        super().__init__(coord)
        self.value = float


class CharConstant(Constant):
    __slots__ = ()
    attr_names = ()

    value: str

    def __init__(self, value: str, coord: Coord):
        super().__init__(coord)
        self.value = value


class BoolConstant(Constant):
    __slots__ = ()
    attr_names = ()

    value: bool

    def __init__(self, value: bool, coord: Coord):
        super().__init__(coord)
        self.value = value


class ID(Node):
    __slots__ = "name", "is_global"
    attr_names = ("name",)
    special_attr = ("is_global",)

    def __init__(self, name: str, coord: Coord):
        super().__init__(coord)
        self.name = name
        self.is_global = False

    def lvalue_name(self) -> ID:
        return self


class TypeSpec(Node):
    __slots__ = ("name",)
    attr_names = ("name",)

    uc_type: PrimaryType

    def __init__(self, name: str, coord: Coord):
        super().__init__(coord)
        self.name = name
