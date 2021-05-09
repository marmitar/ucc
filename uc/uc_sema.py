import pathlib
import sys
from argparse import ArgumentParser
from typing import Optional
from uc.uc_ast import ID, Assignment, BinaryOp, Node, Program
from uc.uc_parser import Coord, UCParser
from uc.uc_type import CharType, IntType, uCType


class SymbolTable(dict):
    """Class representing a symbol table. It should provide functionality
    for adding and looking up nodes associated with identifiers.
    """

    def __init__(self):
        super().__init__()

    def add(self, name: str, value: uCType) -> None:
        self[name] = value

    def lookup(self, name: str) -> Optional[uCType]:
        return self.get(name, None)


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
        # Visit all of the global declarations
        for decl in node.gdecls:
            self.visit(decl)
        # TODO: Manage the symbol table

    # # # # # # # #
    # STATEMENTS  #

    # # # # # # # #
    # EXPRESSIONS #

    def visit_BinaryOp(self, node: BinaryOp) -> None:
        # Visit the left and right expression
        self.visit(node.left)
        ltype = node.left.uc_type
        self.visit(node.right)
        rtype = node.right.uc_type
        # Make sure left and right operands have the same type
        self._assert_semantic(ltype == rtype, 6, node.coord, node.op, ltype, rtype)
        # Make sure the operation is supported
        self._assert_semantic(node.op in ltype.binary_ops, 7, node.coord, node.op, ltype)
        # Assign the result type
        node.uc_type = ltype

    def visit_Assignment(self, node: Assignment) -> None:
        # visit right side
        self.visit(node.expr)
        rtype = node.expr.uc_type
        # visit left side (must be a location)
        var = node.lvalue
        self.visit(var)
        if isinstance(var, ID):
            self._assert_semantic(var.scope is not None, 1, node.coord, var.name)
        ltype = var.uc_type
        # Check that assignment is allowed
        self._assert_semantic(ltype == rtype, 4, node.coord, ltype=ltype, rtype=rtype)
        # Check that assign_ops is supported by the type
        self._assert_semantic(node.op in ltype.assign_ops, 5, node.coord, node.op, ltype)

    # # # # # # # # #
    # BASIC SYMBOLS #


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
