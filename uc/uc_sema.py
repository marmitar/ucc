from __future__ import annotations
import pathlib
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import cache
from itertools import zip_longest
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from uc.uc_ast import (
    ID,
    ArrayRef,
    Assignment,
    BinaryOp,
    Break,
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


class ExprIsNotAnArray(SemanticError):
    def __init__(self, node: Node):
        super().__init__("expression is not an array", node.coord)


class ExprIsNotConstant(SemanticError):  # msg_code: 20
    def __init__(self, node: Node):
        super().__init__("Expression must be a constant", node.coord)


class BreakOutsideLoop(SemanticError):  # msg_code: 8
    def __init__(self, stmt: Break):
        super().__init__("Break statement must be inside a loop", stmt.coord)


class UndefinedError(SemanticError):  # msg_code: 27
    def __init__(self, coord: Optional[Coord]):
        super().__init__("Undefined error", coord)


# # # # # # # #
# TYPE ERRORS #


class UnexpectedType(SemanticError):
    """Abstract Exception for all type checking errors."""

    error_format = "{item} must be of {expected}"
    item: str
    expected: uCType

    def __init__(self, symbol: Optional[Node] = None, *, coord: Optional[Coord] = None):
        uc_type = symbol and symbol.uc_type
        msg = self.error_format.format(item=self.item, expected=self.expected, type=uc_type)
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


# # # # # # # # # # # # #
# OPERATION TYPE ERRORS #


class InvalidOperation(SemanticError):
    """Abstract Exception for invalid operations."""

    kind: str
    problem: str
    type: Optional[uCType]

    def __init__(self, expr: Union[BinaryOp, UnaryOp], msg: Optional[str] = None):
        if msg is None:
            # default message, if not given
            msg = f"{self.kind} operator {expr.op} {self.problem}"

            if getattr(self, "type", None) is not None:
                msg += f" by {self.type}"

        super().__init__(msg, expr.coord)


class OperationTypeDoesNotMatch(InvalidOperation):  # msg_code: 4, 6
    problem = "does not have matching LHS/RHS types"

    def __init__(self, expr: BinaryOp):
        if isinstance(expr, Assignment):
            # special message on assignments
            ltype, rtype = expr.left.uc_type, expr.right.uc_type
            super().__init__(expr, f"Cannot assign {rtype} to {ltype}")
        else:
            self.kind = "Binary"
            super().__init__(expr)


class UnsupportedOperation(InvalidOperation):  # msg_code: 5, 7, 26
    problem = "is not supported"

    def __init__(self, expr: Union[BinaryOp, UnaryOp]):
        if isinstance(expr, UnaryOp):
            self.kind = "Unary"
            # don't show type
        elif isinstance(expr, Assignment):
            self.kind = "Assignment"
            self.type = expr.left.uc_type
        else:
            self.kind = "Binary"
            self.type = expr.left.uc_type

        super().__init__(expr)


# # # # # # # # # # # # # # # #
# HIGHER ORDER PARAM MISMATCH #


class ExprParamMismatch(SemanticError):
    """Abstract Exception for sizes mismatches."""

    error_fmt: str

    def __init__(self, expr: Node, *, default: str = "<UNNAMED>"):
        # extract name from declaration or identifier
        if isinstance(expr, Decl):
            msg = self.error_fmt.format(name=expr.name.name)
        elif isinstance(expr, ID):
            msg = self.error_fmt.format(name=expr.name)
        # or use the default name
        else:
            msg = self.error_fmt.format(name=default)

        super().__init__(msg, expr.coord)


class ArrayDimensionMismatch(ExprParamMismatch):  # msg: 9
    error_fmt = "Array dimension mismatch"


class ArraySizeMismatch(ExprParamMismatch):  # msg: 10
    error_fmt = "Size mismatch on {name} initialization"


class ArrayHigherSizeMismatch(ArraySizeMismatch):  # msg: 14
    error_fmt = "List & variable have different sizes"


class ArrayIsNotHomogeneous(ArrayDimensionMismatch):  # msg: 13
    error_fmt = "Lists have different sizes"


class VariableIsNotArray(ExprParamMismatch):  # msg: 12
    error_fmt = "{name} initialization must be a single element"


class FuncParamsLengthMismatch(ExprParamMismatch):  # msg: 17
    error_fmt = "no. arguments to call {name} function mismatch"


# # # # # # # #
# AST VISITOR #

T, U = TypeVar("T"), TypeVar("U")

# type hint for 'zip_longest' as used here
@overload
def zip_longest(i0: Iterable[T], i1: Iterable[U]) -> Iterator[Tuple[Optional[T], Optional[U]]]:
    ...


class NodeVisitor:
    """A base NodeVisitor class for visiting uc_ast nodes. This class
    uses the visitor pattern. You need to define methods of the form
    visit_NODE() for each kind of AST node that you want to process.
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

    @cache
    def visitor_for(self, node: str) -> Callable[[Node], None]:
        """Find visitor method for a node class."""
        return getattr(self, f"visit_{node}", self.generic_visit)

    def visit(self, node: Node) -> None:
        """Visit a node."""
        visitor = self.visitor_for(node.classname)
        visitor(node)

    def generic_visit(self, node: Node) -> None:
        """Called if no explicit visitor function exists for a
        node. Implements preorder visiting of the node.
        """
        for _, child in node.children():
            self.visit(child)

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

    def visit_BinaryOp(self, node: BinaryOp, kind: str = "binary_ops") -> None:
        # Visit the left and right expression
        self.generic_visit(node)
        ltype = node.left.uc_type
        rtype = node.right.uc_type
        # Make sure left and right operands have the same type
        if ltype != rtype:
            raise OperationTypeDoesNotMatch(node)
        # Make sure the operation is supported
        if node.op not in getattr(ltype, kind, {}):
            raise UnsupportedOperation(node)
        # Assign the result type
        node.uc_type = ltype

    def visit_RelationOp(self, node: RelationOp) -> None:
        self.visit_BinaryOp(node, kind="rel_ops")
        # comparison results in boolean
        node.uc_type = BoolType

    def visit_Assignment(self, node: Assignment) -> None:
        self.visit_BinaryOp(node, kind="assign_ops")
        # TODO: arrays

    def visit_UnaryOp(self, node: UnaryOp) -> None:
        self.generic_visit(node)
        uctype = node.expr.uc_type
        # Make sure the operation is supported
        if node.op not in uctype.unary_ops:
            raise UnsupportedOperation(node)
        # Assign the result type
        node.uc_type = uctype

    def visit_ExprList(self, node: ExprList) -> None:
        self.generic_visit(node)
        # same type as last expression
        node.uc_type = node.expr[-1].uc_type

    def visit_ArrayRef(self, node: ArrayRef) -> None:
        self.generic_visit(node)
        # ltype must be an array type
        uc_type = node.array.uc_type
        if not isinstance(uc_type, ArrayType):
            raise ExprIsNotAnArray(node.array)
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
        # check parameters types and length
        for param, value in zip_longest(ltype.params, node.parameters()):
            if param is None or value is None:
                raise FuncParamsLengthMismatch(node.callable, default=ltype.funcname)
            if value.uc_type != param.type:
                raise InvalidParameterType(param.name, value)

        node.uc_type = ltype.rettype

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


class Visitor:
    """
    Program visitor class.
    """

    def __init__(self):
        self.node_visitor = NodeVisitor()

    def visit(self, node: Node) -> None:
        """Print and exit in case of semantic errors."""
        try:
            self.node_visitor.visit(node)
        except SemanticError as err:
            print("SemanticError:", err, file=sys.stdout)
            sys.exit(1)


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
