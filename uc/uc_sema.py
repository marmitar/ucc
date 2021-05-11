from __future__ import annotations
import pathlib
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from itertools import zip_longest
from typing import Dict, Iterator, List, Optional
from uc.uc_ast import (
    ID,
    ArrayRef,
    Assignment,
    BinaryOp,
    Constant,
    Decl,
    ExprList,
    FuncCall,
    Node,
    Program,
    RelationOp,
    Type,
    UnaryOp,
)
from uc.uc_parser import Coord, UCParser
from uc.uc_type import (
    ArrayType,
    BoolType,
    CharType,
    FunctionType,
    IntType,
    VoidType,
    uCType,
)


class SymbolTable:
    """Class representing a symbol table. It should provide functionality
    for adding and looking up nodes associated with identifiers.
    """

    def __init__(self):
        # stack of scoped symbols
        self.scope_stack: List[Dict[str, ID]] = []

    def push_scope(self) -> None:
        """Create new scope for symbol declarations."""
        self.scope_stack.append({})

    def pop_scope(self) -> Dict[str, ID]:
        """Remove latest scope from table stack."""
        return self.scope_stack.pop()

    @contextmanager
    def new_scope(self) -> Iterator[SymbolTable]:
        """Context manager that automatically closes the scope."""
        try:
            self.push_scope()
            yield self
        finally:
            self.pop_scope()

    @property
    def current_scope(self) -> Dict[str, ID]:
        """The innermost scope for current 'Node'."""
        return self.scope_stack[-1]

    def add(self, name: str, definition: ID) -> bool:
        """Add or change symbol definition in current scope."""
        self.current_scope[name] = definition

    def lookup(self, name: str) -> Optional[ID]:
        """Find symbol type from inner to outermost scope."""
        for scope in reversed(self.scope_stack):
            ident = scope.get(name, None)
            if ident is not None:
                return ident


# # # # # # # #
# EXCEPTIONS  #


class SemanticError(Exception):
    """Abstract Exception for any kind of semantic error."""

    __slots__ = "message", "coord"

    def __init__(self, msg: str, coord: Optional[Coord]):
        self.message = msg
        self.coord = coord
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message} {self.coord or ''}"


class UndefinedIdentifier(SemanticError):  # msg_code: 1
    def __init__(self, ident: ID):
        msg = f"{ident.name} is not defined"
        super().__init__(msg, ident.coord)


class NameAlreadyDefined(SemanticError):  # msg_code: 25
    def __init__(self, ident: ID):
        msg = f"Name {ident.name} is already defined in this scope"
        super().__init__(msg, ident.coord)


class NodeIsNotAVariable(SemanticError):  # msg_code: 23
    def __init__(self, node: Node):
        super().__init__(f"{node} is not a variable", node.coord)


# # # # # # # #
# TYPE ERRORS #


class UnexpectedType(SemanticError):
    """Abstract Exception for all type checking errors."""

    error_format = "{item} must be of {expected}"
    item: str
    expected: uCType

    def __init__(self, symbol: Optional[Node] = None, *, coord: Optional[Coord] = None):
        uc_type = repr(symbol and symbol.uc_type)
        msg = self.error_format.format(item=self.item, expected=repr(self.expected), type=uc_type)
        # uses 'coord' if given, or symbol coordinates
        super().__init__(msg, coord or (symbol and symbol.coord))


class InvalidSubscriptType(UnexpectedType):  # msg_code: 2
    error_format = UnexpectedType.error_format + ", not {type}"
    item = "subscript"
    expected = IntType


class InvalidBooleanExpression(UnexpectedType):  # msg_code: 3
    item = "Expression"
    expected = BoolType


class InvalidConditionalExpression(InvalidBooleanExpression):  # msg_code: 15
    error_format = "{item} is {type}, not {expected}"
    item = "conditional expression"


class InvalidLoopCondition(InvalidBooleanExpression):  # msg_code: 19
    item = "The condition expression"


class InvalidReturnType(UnexpectedType):  # msg_code: 24
    error_format = "Return of {type} is incompatible with {expected} function definition"


class InvalidAssignmentType(UnexpectedType):  # msg_code: 4
    error_format = "Cannot assign {type} to {expected}"

    def __init__(self, assign: Assignment):
        self.expected = assign.left.uc_type
        super().__init__(assign.right, coord=assign.coord)


class InvalidInitializationType(UnexpectedType):  # msg_code: 11
    error_format = "{item} initialization type mismatch"

    def __init__(self, ident: ID):
        self.item = ident.name
        super().__init__(symbol=ident)


class ExprIsNotAFunction(UnexpectedType):  # msg_code: 16
    error_format = "{item} is not a function"

    def __init__(self, item: Node):
        self.item = getattr(item, "name", "<expression>")
        super().__init__(symbol=item)


class InvalidParameterType(UnexpectedType):  # msg_code: 18
    error_format = "Type mismatch with parameter {item}"

    def __init__(self, name: str, value: Node):
        self.item = name
        super().__init__(symbol=value)


class ExprHasCompoundType(UnexpectedType):  # msg_code: 21
    error_format = "Expression is not of basic type"


class VariableHasCompoundType(ExprHasCompoundType):  # msg_code: 22
    error_format = "{item} does not reference a variable of basic type"

    def __init__(self, variable: ID):
        self.item = variable.name
        super().__init__(symbol=variable)


# # # # # # # #
# AST VISITOR #


class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes.
    Subclass it and define your own visit_NODE methods, where
    NODE is the class name you want to visit with these
    methods.
    """

    _method_cache = {}

    def visit(self, node: Node) -> None:
        """Visit a node."""

        visitor = self._method_cache.get(node.classname, None)
        if visitor is None:
            method = "visit_" + node.classname
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.classname] = visitor

        visitor(node)

    def generic_visit(self, node: Node) -> None:
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """
        for _, child in node.children():
            self.visit(child)


class Visitor(NodeVisitor):
    """
    Program visitor class. This class uses the visitor pattern. You need to define methods
    of the form visit_NODE() for each kind of AST node that you want to process.
    """

    def __init__(self):
        # Initialize the symbol table
        self.symtab = SymbolTable()
        self.typemap = {
            "int": IntType,
            "char": CharType,
            "void": VoidType,
            "string": ArrayType(CharType),
            "bool": BoolType,
            # TODO
        }
        # TODO: Complete...

    def _assert_semantic(
        self,
        condition: bool,
        msg_code=None,
        coord: Optional[Coord] = None,
        name="NONAME",
        ltype="NOLTYPE",
        rtype="NORTYPE",
    ) -> None:
        """Check condition, if false print selected error message and exit"""
        error_msgs = {
            # 1: f"{name} is not defined",
            # 2: f"subscript must be of type(int), not {rtype}",
            # 3: "Expression must be of type(bool)",
            # 4: f"Cannot assign {rtype} to {ltype}",
            5: f"Assignment operator {name} is not supported by {ltype}",
            6: f"Binary operator {name} does not have matching LHS/RHS types",
            7: f"Binary operator {name} is not supported by {ltype}",
            8: "Break statement must be inside a loop",
            9: "Array dimension mismatch",
            10: f"Size mismatch on {name} initialization",
            # 11: f"{name} initialization type mismatch",
            12: f"{name} initialization must be a single element",
            13: "Lists have different sizes",
            14: "List & variable have different sizes",
            # 15: f"conditional expression is {ltype}, not type(bool)",
            # 16: f"{name} is not a function",
            17: f"no. arguments to call {name} function mismatch",
            # 18: f"Type mismatch with parameter {name}",
            # 19: "The condition expression must be of type(bool)",
            20: "Expression must be a constant",
            # 21: "Expression is not of basic type",
            # 22: f"{name} does not reference a variable of basic type",
            # 23: f"{name} is not a variable",
            # 24: f"Return of {ltype} is incompatible with {rtype} function definition",
            # 25: f"Name {name} is already defined in this scope",
            26: f"Unary operator {name} is not supported",
            27: "Undefined error",
        }
        if not condition:
            msg = error_msgs[msg_code or len(error_msgs) - 1]
            print(f"SemanticError: {msg} {coord or ''}", file=sys.stdout)
            sys.exit(1)

    # # # # # # # # #
    # DECLARATIONS  #

    def visit_Program(self, node: Program) -> None:
        # global scope
        with self.symtab.new_scope():
            # Visit all of the global declarations
            self.generic_visit(node)

    def visit_Decl(self, node: Decl) -> None:
        # Visit the types of the declaration
        self.visit(node.type)
        ltype = node.type.uc_type
        # define the function or variable
        self.visit_ID(node.name, ltype)
        # If there is an initial value defined, visit it
        if node.init is not None:
            self.visit(node.init)
            rtype = node.init.uc_type
            # check if initilization is valid
            if ltype != rtype:
                raise InvalidInitializationType(node.name)
            # TODO: arrays

    # # # # # # # #
    # STATEMENTS  #

    # # # # # # # #
    # EXPRESSIONS #

    def visit_BinaryOp(self, node: BinaryOp, kind="binary_ops", errno=(6, 7)) -> None:
        # Visit the left and right expression
        self.generic_visit(node)
        ltype = node.left.uc_type
        rtype = node.right.uc_type
        # Make sure left and right operands have the same type
        self._assert_semantic(ltype == rtype, errno[0], node.coord, node.op, ltype, rtype)
        # Make sure the operation is supported
        self._assert_semantic(
            node.op in getattr(ltype, kind, {}), errno[1], node.coord, node.op, ltype, rtype
        )
        # Assign the result type
        node.uc_type = ltype

    def visit_RelationOp(self, node: RelationOp) -> None:
        self.visit_BinaryOp(node, "rel_ops")
        node.uc_type = BoolType

    def visit_Assignment(self, node: Assignment) -> None:
        self.visit_BinaryOp(node, "assign_ops", errno=(4, 5))
        node.uc_type = VoidType  # TODO

    def visit_UnaryOp(self, node: UnaryOp) -> None:
        self.generic_visit(node)
        rtype = node.expr.uc_type
        # Make sure the operation is supported
        self._assert_semantic(node.op in rtype.unary_ops, 26, node.op)
        # Assign the result type
        node.uc_type = rtype

    def visit_ExprList(self, node: ExprList) -> None:
        self.generic_visit(node)
        # same type as last expression
        node.uc_type = node.expr[-1].uc_type

    def visit_ArrayRef(self, node: ArrayRef) -> None:
        self.generic_visit(node)
        # ltype must be an array type
        uc_type = node.array.uc_type
        self._assert_semantic(isinstance(uc_type, ArrayType), ltype=uc_type)
        # index must be 'int'
        if node.index.uc_type != IntType:
            raise InvalidSubscriptType(node.index)
        # TODO: check size?

    def visit_FuncCall(self, node: FuncCall) -> None:
        self.generic_visit(node)
        # ltype must be a function type
        ltype = node.callable.uc_type
        if not isinstance(ltype, FunctionType):
            raise ExprIsNotAFunction(node.callable)
        # check length and types # TODO: check NONE
        for param, value in zip_longest(ltype.params, node.params.expr):
            self._assert_semantic(
                param is not None and value is not None, 17, value.coord, ltype.typename
            )
            name, type = param
            if value.uc_type != type:
                raise InvalidParameterType(name, value)

        node.uc_type = ltype.type

    # # # # # # # # #
    # BASIC SYMBOLS #

    def visit_ID(self, node: ID, uctype: Optional[uCType] = None) -> None:
        if uctype is None:
            # Look for its declaration in the symbol table
            definition = self.symtab.lookup(node.name)
            if definition is None:
                raise UndefinedIdentifier(node)
            # bind identifier to its associated symbol type
            node.uc_type = definition.uc_type

        else:
            if node.name in self.symtab.current_scope:
                raise NameAlreadyDefined(node)
            # initialize the type, # TODO kind, and scope attributes
            node.uc_type = uctype
            self.symtab.add(node.name, node)

    def visit_Constant(self, node: Constant) -> None:
        # Get the matching uCType
        node.uc_type = self.typemap[node.rawtype]

    def visit_Type(self, node: Type) -> None:
        # Get the matching basic uCType
        node.uc_type = self.typemap[node.name]


if __name__ == "__main__":

    # create argument parser
    parser = ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be semantically checked", type=str)
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())
        sema = Visitor()
        sema.visit(ast)
