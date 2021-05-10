from __future__ import annotations
import pathlib
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional
from uc.uc_ast import (
    ID,
    Assignment,
    BinaryOp,
    Constant,
    Decl,
    Node,
    Program,
    RelationOp,
    Type,
)
from uc.uc_parser import Coord, UCParser
from uc.uc_type import ArrayType, BoolType, CharType, IntType, VoidType, uCType


class SymbolTable:
    """Class representing a symbol table. It should provide functionality
    for adding and looking up nodes associated with identifiers.
    """

    def __init__(self):
        # stack of scoped symbols
        self.scope_stack: List[Dict[str, uCType]] = []

    def push_scope(self) -> None:
        """Create new scope for symbol declarations."""
        self.scope_stack.append({})

    def pop_scope(self) -> Dict[str, uCType]:
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

    def add(self, name: str, value: uCType) -> bool:
        """Add symbol in latest scope."""
        if name in self.scope_stack[-1]:
            return False

        self.scope_stack[-1][name] = value
        return True

    def lookup(self, name: str) -> Optional[uCType]:
        """Find symbol type from inner to outer scope."""
        for scope in reversed(self.scope_stack):
            uctype = scope.get(name, None)
            if uctype is not None:
                return uctype


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
        msg_code: int,
        coord: Optional[Coord],
        name="NONAME",
        ltype="NOLTYPE",
        rtype="NORTYPE",
    ) -> None:
        """Check condition, if false print selected error message and exit"""
        error_msgs = {
            1: f"{name} is not defined",
            2: f"subscript must be of type(int), not {ltype}",
            3: "Expression must be of type(bool)",
            4: f"Cannot assign {rtype} to {ltype}",
            5: f"Assignment operator {name} is not supported by {ltype}",
            6: f"Binary operator {name} does not have matching LHS/RHS types",
            7: f"Binary operator {name} is not supported by {ltype}",
            8: "Break statement must be inside a loop",
            9: "Array dimension mismatch",
            10: f"Size mismatch on {name} initialization",
            11: f"{name} initialization type mismatch",
            12: f"{name} initialization must be a single element",
            13: "Lists have different sizes",
            14: "List & variable have different sizes",
            15: f"conditional expression is {ltype}, not type(bool)",
            16: f"{name} is not a function",
            17: f"no. arguments to call {name} function mismatch",
            18: f"Type mismatch with parameter {name}",
            19: "The condition expression must be of type(bool)",
            20: "Expression must be a constant",
            21: "Expression is not of basic type",
            22: f"{name} does not reference a variable of basic type",
            23: f"{name} is not a variable",
            24: f"Return of {ltype} is incompatible with {rtype} function definition",
            25: f"Name {name} is already defined in this scope",
            26: f"Unary operator {name} is not supported",
            27: "Undefined error",
        }
        if not condition:
            msg = error_msgs.get(msg_code)
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
            self._assert_semantic(ltype == rtype, 11, node.coord, node.op)
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

    # # # # # # # # #
    # BASIC SYMBOLS #

    def visit_ID(self, node: ID, uctype: Optional[uCType] = None) -> None:
        if uctype is None:
            # Look for its declaration in the symbol table
            uctype = self.symtab.lookup(node.name)
            self._assert_semantic(uctype is not None, 1, node.coord, node.name)
        else:
            # initialize the type, kind, and scope attributes
            ok = self.symtab.add(node.name, uctype)
            self._assert_semantic(ok, 25, node.coord, node.name)

        # bind identifier to its associated symbol type
        node.uc_type = uctype

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
