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
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from uc.uc_ast import (
    ID,
    ArrayDecl,
    ArrayRef,
    Assert,
    Assignment,
    BinaryOp,
    Break,
    Compound,
    Constant,
    Decl,
    ExprList,
    For,
    FuncCall,
    FuncDecl,
    FuncDef,
    If,
    InitList,
    IterationStmt,
    Node,
    Print,
    Program,
    Read,
    RelationOp,
    Return,
    Type,
    UnaryOp,
    VarDecl,
    While,
)
from uc.uc_parser import Coord, UCParser
from uc.uc_type import (
    ArrayType,
    BoolType,
    CharType,
    FunctionType,
    IntType,
    PrimaryType,
    VoidType,
    uCType,
)


class Symbol(NamedTuple):
    """Symbol information."""

    definition: ID

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def type(self) -> uCType:
        return self.definition.uc_type


class Scope(Dict[str, Symbol]):
    """Scope with symbol mappings."""

    __slots__ = ()

    def add(self, symb: Symbol) -> None:
        """Add or change symbol definition in scope."""
        self[symb.name] = symb


class IterationScope(Scope):
    """Scope inside iteration statement."""

    __slots__ = ("statement",)

    def __init__(self, iteration_stmt: IterationStmt):
        super().__init__()
        self.statement = iteration_stmt


class FunctionScope(Scope):
    """Scope inside function body."""

    __slots__ = ("definition",)

    def __init__(self, definition: FuncDef):
        self.definition = definition

    @property
    def ident(self) -> ID:
        return self.definition.declaration.name

    @property
    def name(self) -> str:
        return self.ident.name

    @property
    def return_type(self) -> uCType:
        return self.definition.return_type.uc_type


class SymbolTable:
    """Class representing a symbol table. It should provide functionality
    for adding and looking up nodes associated with identifiers.
    """

    def __init__(self):
        # lookup stack of scope
        self.stack: List[Scope] = []

    @contextmanager
    def new(self, scope: Optional[Scope] = None) -> Iterator[Scope]:
        """
        Insert new scope in the lookup stack and automatically
        removes it when closed.
        """
        new_scope = scope or Scope()
        self.stack.append(new_scope)
        try:
            yield new_scope
        finally:
            self.stack.pop()

    @property
    def current_scope(self) -> Scope:
        """The innermost scope for current 'Node'."""
        return self.stack[-1]

    def add(self, definition: ID) -> bool:
        """Add or change symbol definition in current scope."""
        self.current_scope.add(Symbol(definition))

    def find(self, matches: Callable[[Scope], bool]) -> Optional[Scope]:
        """Find scope with given property, from inner to outermost."""
        for scope in reversed(self.stack):
            if matches(scope):
                return scope

    def lookup(self, name: str) -> Optional[Symbol]:
        """Find symbol type from inner to outermost scope."""
        scope = self.find(lambda scope: name in scope)
        if scope is not None:
            return scope[name]


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


class NegativeArraySize(SemanticError):
    def __init__(self, size: Constant):
        super().__init__("Array size can't be negative", size.coord)


# # # # # # # #
# TYPE ERRORS #


class UnexpectedType(SemanticError):
    """Abstract Exception for all type checking errors."""

    error_format = "{item} must be of {expected}"
    item: str
    expected: uCType

    def __init__(self, symbol: Optional[Node] = None, *, coord: Optional[Coord] = None):
        expected = getattr(self, "expected", "<UNKNOWN>")
        item = getattr(self, "item", "<UNKNOWN>")
        uc_type = symbol and symbol.uc_type

        msg = self.error_format.format(item=item, expected=expected, type=uc_type)
        # uses 'coord' if given, or symbol coordinates
        super().__init__(msg, coord or (symbol and symbol.coord))


class UnknownType(UnexpectedType):
    error_format = "Unknown type {item}"

    def __init__(self, name: str, coord: Optional[Coord] = None):
        self.item = name
        super().__init__(coord=coord)


class InvalidSubscriptType(UnexpectedType):  # msg_code: 2
    error_format = UnexpectedType.error_format + ", not {type}"
    item = "subscript"
    expected = IntType


class InvalidBooleanExpression(UnexpectedType):  # msg_code: 3
    item = "Expression"
    expected = BoolType


class InvalidConditionalExpression(InvalidBooleanExpression):  # msg_code: 19
    item = "The condition expression"


class InvalidLoopCondition(InvalidBooleanExpression):  # msg_code: 15
    error_format = "{item} is {type}, not {expected}"
    item = "conditional expression"


class InvalidReturnType(UnexpectedType):  # msg_code: 24
    error_format = "Return of {type} is incompatible with {expected} function definition"


class IncompatibleListType(UnexpectedType):
    error_format = "initilization list has incompatible types"


class InvalidInitializationType(IncompatibleListType):  # msg_code: 11
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


class InvalidVariableType(UnexpectedType):
    error_format = "Cannot create {item} of {type}.."

    def __init__(self, symbol: Node, kind: str = "a variable"):
        self.item = kind
        super().__init__(symbol)


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

    __slots__ = ("symtab",)

    def __init__(self):
        # Initialize the symbol table
        self.symtab = SymbolTable()

    @cache
    def visitor_for(self, node: str) -> Callable[[Node], None]:
        """Find visitor method for a node class."""
        return getattr(self, f"visit_{node}", self.generic_visit)

    def visit(self, node: Node) -> None:
        """Visit a node."""
        visitor = self.visitor_for(node.classname)
        visitor(node)

    def generic_visit(self, node: Node) -> None:
        """Preorder visiting of the node's children."""
        for _, child in node.children():
            self.visit(child)
        # default declaration and statement type
        node.uc_type = VoidType

    # # # # # # # # #
    # DECLARATIONS  #

    def visit_Program(self, node: Program) -> None:
        # global scope
        with self.symtab.new():
            # Visit all of the global declarations
            self.generic_visit(node)

    def visit_Decl(self, node: Decl, *, visit_type: bool = True) -> None:
        # Visit the declaration type and initialization
        if visit_type:
            self.visit(node.type)
        ltype = node.type.uc_type
        # define the function or variable
        self.visit_ID(node.name, ltype)
        if node.init is None:
            return  # ok, just uninitialized
        self.visit(node.init)
        rtype = node.init.uc_type
        # check if initilization is valid
        if ltype != rtype:
            raise InvalidInitializationType(node.name)

    def visit_VarDecl(self, node: VarDecl) -> None:
        self.generic_visit(node)
        # just pass on the basic type
        node.uc_type = node.type.uc_type

    def visit_ArrayDecl(self, node: ArrayDecl) -> None:
        self.generic_visit(node)

        elem_type = node.type.uc_type
        if isinstance(elem_type, ArrayType) and elem_type.size is None:
            # only outer array modifier may be unsized
            raise ArrayDimensionMismatch(node.type)
        elif elem_type == VoidType:
            raise InvalidVariableType(node.type, "an array")

        if node.size is not None:
            size = node.size
            # check if size is int constant
            if not isinstance(size, Constant):
                raise ExprIsNotConstant(size)
            elif size.uc_type != IntType or not isinstance(size.value, int):
                raise InvalidSubscriptType(size)
            elif size.value < 0:
                raise NegativeArraySize(size)
            # define the type size
            array_size = size.value
        else:
            array_size = None

        node.uc_type = ArrayType(elem_type, array_size)

    def visit_FuncDecl(self, node: FuncDecl) -> None:
        # visit parameters
        self.generic_visit(node)
        # get funtion name
        scope = self.symtab.current_scope
        assert isinstance(scope, FunctionScope)
        name = scope.name

        # build the function type
        rettype = node.type.uc_type
        if node.param_list:
            params = [(p.name.name, p.type.uc_type) for p in node.param_list.params]
            uc_type = FunctionType(name, rettype, params)
        else:
            uc_type = FunctionType(name, rettype)
        # and bind it to the declaration
        node.uc_type = uc_type

    def visit_InitList(self, node: InitList) -> None:
        self.generic_visit(node)
        # init lists without elements have no type, but can be coerced
        if len(node) == 0:
            node.uc_type = ArrayType.empty_list()
            return

        elem_type = node.init[0].uc_type

        for elem in node.init:
            # types must match
            if elem.uc_type != elem_type:
                raise IncompatibleListType(elem)
            # sizes must match
            if isinstance(elem_type, ArrayType) and elem_type.size != elem.uc_type.size:
                raise ArrayIsNotHomogeneous(elem)
            # items must be constant or list
            if not isinstance(elem, (InitList, Constant)):
                raise ExprIsNotConstant(elem)

        node.uc_type = ArrayType(elem_type, len(node))

    # # # # # # # #
    # STATEMENTS  #

    def visit_Assert(self, node: Assert) -> None:
        self.generic_visit(node)
        # verify it is of boolean type
        if node.param.uc_type != BoolType:
            raise InvalidBooleanExpression(node.param)

    def visit_Break(self, node: Break) -> None:
        self.generic_visit(node)
        # Check the Break statement is inside a loop.
        loop = self.symtab.find(lambda scope: isinstance(scope, IterationScope))
        if not isinstance(loop, IterationScope):
            raise BreakOutsideLoop(node)
        # Bind it to the current loop node.
        node.bind(loop.statement)

    def visit_Compound(self, node: Compound) -> None:
        # open new scope
        with self.symtab.new():
            self.generic_visit(node)

    def _is_basic_type(self, type: uCType) -> bool:
        """Checker for basic types."""
        return (
            # any primary type, except void
            (isinstance(type, PrimaryType) and type != VoidType)
            or
            # or a char array, i.e. a string
            (isinstance(type, ArrayType) and type.elem_type == CharType)
        )

    def visit_Read(self, node: Read) -> None:
        self.generic_visit(node)

        for child in node.param.expr:
            # verify that all identifiers are variables
            ident = child.lvalue_name()
            if ident is None:
                raise NodeIsNotAVariable(child)
            # check if it is of basic type
            elif not self._is_basic_type(child.uc_type):
                raise VariableHasCompoundType(ident)

    def visit_Print(self, node: Print) -> None:
        self.generic_visit(node)

        if node.param is None:
            return  # newline
        for child in node.param.expr:
            if not self._is_basic_type(child.uc_type):
                raise ExprHasCompoundType(child)

    def visit_Return(self, node: Return) -> None:
        self.generic_visit(node)
        _ = node.result.uc_type if node.result else VoidType
        # TODO: check that its type is identical to the return type of the function definition

    def visit_IterationStmt(self, node: IterationStmt) -> None:
        # create new breakable scope
        with self.symtab.new(IterationScope(node)):
            self.generic_visit(node)
        # check if the conditional expression is of boolean type
        if node.condition is not None and node.condition.uc_type != BoolType:
            raise InvalidLoopCondition(node.condition)

    def visit_For(self, node: For) -> None:
        self.visit_IterationStmt(node)

    def visit_While(self, node: While) -> None:
        self.visit_IterationStmt(node)

    def visit_If(self, node: If) -> None:
        self.generic_visit(node)
        # check if the conditional expression is of boolean type
        if node.condition.uc_type != BoolType:
            raise InvalidConditionalExpression(node.condition)

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
        # when used as the comma operator
        node.uc_type = node.as_comma_op().uc_type

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
        node.uc_type = uc_type.elem_type

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
            node.uc_type = definition.type

        else:  # initialize the type
            node.uc_type = uctype
            if uctype == VoidType:
                raise InvalidVariableType("a variable", node)
            if node.name in self.symtab.current_scope:
                raise NameAlreadyDefined(node)
            # TODO kind, and scope attributes
            self.symtab.add(node)

    def _get_primary_type(self, typename: str, coord: Optional[Coord]) -> uCType:
        """Get primary type from type name."""
        uc_type = PrimaryType.get(typename)
        # check if type exists
        if uc_type is None:
            raise UnknownType(typename, coord)
        return uc_type

    def visit_Constant(self, node: Constant) -> None:
        # Get the matching uCType
        if node.rawtype == "string":
            node.uc_type = ArrayType(CharType, len(node.value) + 1)  # TODO: size?
        else:
            node.uc_type = self._get_primary_type(node.rawtype, node.coord)

    def visit_Type(self, node: Type) -> None:
        # Get the matching basic uCType
        node.uc_type = self._get_primary_type(node.name, node.coord)


class Visitor:
    """
    Program visitor class.
    """

    def __init__(self):
        self.node_visitor = NodeVisitor()

    def visit(self, node: Program) -> None:
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
