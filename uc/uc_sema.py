from __future__ import annotations
import itertools
import pathlib
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
from functools import cache
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
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

T, U = TypeVar("T"), TypeVar("U")

# type hint for 'zip_longest' in this file
def zip_longest(i0: Iterable[T], i1: Iterable[U]) -> Iterator[Tuple[Optional[T], Optional[U]]]:
    return itertools.zip_longest(i0, i1)


# # # # # # # # # # #
# SCOPE AND SYMBOLS #


class Symbol:
    """Symbol information."""

    __slots__ = ("definition",)

    definition: ID

    def __init__(self, definition: ID):
        self.definition = definition

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def type(self) -> uCType:
        return self.definition.uc_type

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return f"{self.name}: {self.type}"


class FuncDefSymbol(Symbol):
    """Symbol for Defined Functions."""

    __slots__ = ("body",)

    def __init__(self, definition: ID, body: FuncDef):
        super().__init__(definition)
        self.body = body

    @property
    def type(self) -> FunctionType:
        return self.definition.uc_type


class Scope:
    """Scope with symbol mappings."""

    __slots__ = "table", "block"

    def __init__(self, block: Union[Program, FuncDef]):
        self.table: Dict[str, Symbol] = {}
        self.block = block

    def add(self, symb: Symbol) -> None:
        """Add or change symbol definition in scope."""
        self.pop(symb.name)
        # inser new symbol
        self.table[symb.name] = symb
        self.block.symbols.append(symb)

    def get(self, name: str) -> Optional[Symbol]:
        return self.table.get(name, None)

    def pop(self, name: str) -> Optional[Symbol]:
        symbol = self.table.pop(name, None)
        if symbol is None:
            return
        # remove symbol if found
        for i, s in enumerate(self.block.symbols):
            if s in symbol:
                return self.block.symbols.pop(i)

    def __contains__(self, name: str) -> bool:
        return name in self.table

    def __str__(self) -> str:
        """Show 'Scope {NAME: SYMBOL, ...}'."""
        return f"{self.__class__.__name__} {super().__str__()}"


class EmptyScope(Scope):
    """Scope that doesn't define variables."""

    __slots__ = ()

    def __init__(self):
        super().__init__(None)

    def add(self, symb: Symbol) -> None:
        self.table[symb.name] = symb


class IterationScope(Scope):
    """Scope inside iteration statement."""

    __slots__ = ("statement",)

    def __init__(self, iteration_stmt: IterationStmt, block: Union[Program, FuncDef]):
        super().__init__(block)
        self.statement = iteration_stmt


class FunctionScope(Scope):
    """Scope inside function body."""

    block: FuncDef

    def __init__(self, definition: FuncDef):
        super().__init__(definition)

    @property
    def ident(self) -> ID:
        return self.block.declaration.name

    @property
    def type(self) -> FunctionType:
        return self.ident.uc_type

    @property
    def name(self) -> str:
        return self.ident.name

    @property
    def return_type(self) -> uCType:
        return self.block.return_type.uc_type

    def already_defined(self, outer: Scope) -> bool:
        """Check if function was already defined in a given scope."""
        sym = outer.get(self.name)
        return (
            # function can be defined if its a new name
            sym is not None
            and
            # or if old name was just a declaration with same uCType
            (isinstance(sym, FuncDefSymbol) or sym.type != self.type)
        )

    def definition_scope(self, outer: Scope) -> Scope:
        """Special scope for function definition."""

        class DeclScope(Scope):
            def add(this, sym: Symbol) -> None:
                assert sym.definition is self.ident
                # add function to outer scope
                outer.add(FuncDefSymbol(self.ident, self.block))

        return DeclScope(None)


class SymbolTable:
    """Class representing a symbol table. It should provide functionality
    for adding and looking up nodes associated with identifiers.
    """

    def __init__(self):
        # lookup stack of scope
        self.stack: List[Scope] = []
        self.block: Union[Program, FuncDef, None] = None

    @contextmanager
    def new(
        self, scope: Optional[Scope] = None, *, block: Union[Program, FuncDef, None] = None
    ) -> Iterator[Scope]:
        """
        Insert new scope in the lookup stack and automatically
        removes it when closed.
        """
        #
        # update block
        old_block = self.block
        if block is not None:
            self.block = block
        # open new scope
        if scope is None:
            scope = Scope(self.block)
        self.stack.append(scope)
        try:
            yield scope
        finally:
            # close scope and recover block
            self.stack.pop()
            self.block = old_block

    @property
    def current_scope(self) -> Scope:
        """The innermost scope for current 'Node'."""
        return self.stack[-1]

    def add(self, definition: ID) -> None:
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
            return scope.get(name)


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
        node.uc_type = ""  # delete type to match output
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


class ReturnOutsideFunction(SemanticError):
    def __init__(self, stmt: Return):
        super().__init__("Return statement must be inside a function", stmt.coord)


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
        uc_type = getattr(symbol, "uc_type", VoidType)

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

    def __init__(self, stmt: Return, expected: uCType):
        self.expected = expected
        super().__init__(symbol=stmt.result, coord=stmt.coord)


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


class IndexOutOfBounds(ExprParamMismatch):  # msg: 9
    error_fmt = "Index is out of bounds for array {name}"


class ArraySizeMismatch(ExprParamMismatch):  # msg: 10
    error_fmt = "Size mismatch on assignment"


class ArraySizeMismatchOnInit(ArraySizeMismatch):  # msg: 10
    error_fmt = "Size mismatch on {name} initialization"


class ArrayListSizeMismatch(ArraySizeMismatchOnInit):  # msg: 14
    error_fmt = "List & variable have different sizes"


class ArrayIsNotHomogeneous(ArrayDimensionMismatch):  # msg: 13
    error_fmt = "Lists have different sizes"


class VariableIsNotArray(ExprParamMismatch):  # msg: 12
    error_fmt = "{name} initialization must be a single element"


class FuncParamsLengthMismatch(ExprParamMismatch):  # msg: 17
    error_fmt = "no. arguments to call {name} function mismatch"


# # # # # # # #
# AST VISITOR #


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
    def visitor_for(self, node: str) -> Callable[[Node], Optional[uCType]]:
        """Find visitor method for a node class."""
        return getattr(self, f"visit_{node}", self.generic_visit)

    def visit(self, node: Node) -> uCType:
        """Visit a node, set and return its type."""
        visitor = self.visitor_for(node.classname)
        uc_type = visitor(node)
        # default declaration and statement type
        node.uc_type = uc_type or VoidType
        return node.uc_type

    def generic_visit(self, node: Node) -> Literal[VoidType]:
        """Preorder visiting of the node's children."""
        for _, child in node.children():
            self.visit(child)
        return VoidType

    # # # # # # # # #
    # DECLARATIONS  #

    def visit_Program(self, node: Program) -> None:
        # global scope
        with self.symtab.new(block=node):
            # Visit all of the global declarations
            self.generic_visit(node)

    def visit_Decl(self, node: Decl, *, ltype: Optional[uCType] = None) -> None:
        # Visit the declaration type and initialization
        if ltype is None:
            ltype = self.visit(node.type)
        # define the function or variable
        self.visit_ID(node.name, ltype)
        if node.init is None:
            return  # ok, just uninitialized
        rtype = self.visit(node.init)
        # check if initilization is valid
        if ltype != rtype:
            if isinstance(ltype, ArrayType):
                if isinstance(node.init, InitList):
                    raise ArrayListSizeMismatch(node)
                else:
                    raise ArraySizeMismatchOnInit(node.name)
            if isinstance(rtype, ArrayType):
                raise VariableIsNotArray(node)
            else:
                raise InvalidInitializationType(node.name)
        # update unsized arrays
        if isinstance(ltype, ArrayType):
            if ltype.size is None:
                ltype.size = rtype.size
            elif ltype.size != rtype.size:
                if isinstance(node.init, InitList):
                    raise ArrayListSizeMismatch(node)
                else:
                    raise ArraySizeMismatchOnInit(node.name)

    def visit_VarDecl(self, node: VarDecl) -> PrimaryType:
        # just pass on the basic type
        return self.visit_Type(node.type)

    def visit_ArrayDecl(self, node: ArrayDecl) -> ArrayType:
        elem_type = self.visit(node.type)
        if isinstance(elem_type, ArrayType) and elem_type.size is None:
            # only outer array modifier may be unsized
            raise ArrayDimensionMismatch(node.type)
        elif elem_type == VoidType:
            raise InvalidVariableType(node.type, "an array")

        if node.size is not None:
            size = node.size
            size_type = self.visit(size)
            # check if size is int constant
            if not isinstance(node.size, Constant):
                raise ExprIsNotConstant(size)
            elif size_type != IntType or not isinstance(size.value, int):
                raise InvalidSubscriptType(size)
            elif size.value < 0:
                raise NegativeArraySize(size)
            # define the type size
            array_size = size.value
        else:
            array_size = None

        return ArrayType(elem_type, array_size)

    def visit_FuncDecl(self, node: FuncDecl, scope: Optional[Scope] = None) -> FunctionType:
        rettype = self.visit(node.type)
        # visit parameters in given scope, if any
        with self.symtab.new(scope or EmptyScope()):
            self.visit(node.param_list)
        # get function name
        name = node.declname().name
        # build the function type
        params = [(p.name.name, p.type.uc_type) for p in node.param_list.params]
        return FunctionType(name, rettype, params)

    def _needs_implicit_return(self, stmt: Optional[Node]) -> bool:
        """Check if a given statement needs implicit return."""
        # compound depends on its last statement
        if isinstance(stmt, Compound):
            last = stmt.last_statement()
            return self._needs_implicit_return(last)
        # conditional depends on both branchs
        elif isinstance(stmt, If):
            # fmt: off
            return self._needs_implicit_return(stmt.true_stmt) \
                or self._needs_implicit_return(stmt.false_stmt)
            # fmt: on
        # 'for' is ok if it has no condition, no break and has a return
        elif isinstance(stmt, For):
            # fmt: off
            return stmt.condition is None \
                and len(stmt.break_locations) == 0 \
                and self._needs_implicit_return(stmt.body)
            # fmt: on
        # otherwise, only an explicit return don't require implicit
        else:
            return not isinstance(stmt, Return)

    def visit_FuncDef(self, node: FuncDef) -> None:
        # visit return type
        self.visit(node.return_type)
        global_scope = self.symtab.current_scope
        function_scope = FunctionScope(node)
        # declare parameters in function scope
        ltype = self.visit_FuncDecl(node.declaration.type, function_scope)
        function_scope.ident.uc_type = ltype
        # check if function was already defined
        if function_scope.already_defined(global_scope):
            raise NameAlreadyDefined(node.declaration.name)
        # declare function using special scope
        def_scope = function_scope.definition_scope(global_scope)
        with self.symtab.new(def_scope):
            self.visit(node.declaration)
        # declare the function body in the new scope as well
        with self.symtab.new(function_scope):
            self.visit(node.decl_list)
        self.visit_Compound(node.implementation, function_scope)
        # check if implicit return is needed
        if self._needs_implicit_return(node.implementation):
            stmt = Return(None, node.implementation.coord)
            node.implementation.append(stmt)
            # visit it inside function scope
            with self.symtab.new(function_scope):
                self.visit(stmt)

    def visit_InitList(self, node: InitList) -> ArrayType:
        self.generic_visit(node)
        # init lists without elements have no type, but can be coerced
        if len(node) == 0:
            return ArrayType.empty_list()

        elem_type = node.init[0].uc_type

        for elem in node.init:
            # types must match
            if elem.uc_type != elem_type:
                raise IncompatibleListType(elem)
            # sizes must match
            if isinstance(elem_type, ArrayType) and elem_type.size != elem.uc_type.size:
                raise ArrayIsNotHomogeneous(node)
            # items must be constant or list
            if not isinstance(elem, (InitList, Constant)):
                raise ExprIsNotConstant(elem)

        return ArrayType(elem_type, len(node))

    # # # # # # # #
    # STATEMENTS  #

    def visit_Assert(self, node: Assert) -> None:
        uc_type = self.visit(node.param)
        # verify it is of boolean type
        if uc_type != BoolType:
            raise InvalidBooleanExpression(node.param)

    def visit_Break(self, node: Break) -> None:
        # Check the Break statement is inside a loop.
        loop = self.symtab.find(lambda scope: isinstance(scope, IterationScope))
        if not isinstance(loop, IterationScope):
            raise BreakOutsideLoop(node)
        # Bind it to the current loop node.
        node.bind(loop.statement)

    def visit_Compound(self, node: Compound, scope: Optional[Scope] = None) -> None:
        # open new scope
        with self.symtab.new(scope):
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
        node.param.uc_type = VoidType
        # visit each name
        for child in node.param.expr:
            uc_type = self.visit(child)
            # verify that all identifiers are variables
            if child.lvalue_name() is None:
                raise NodeIsNotAVariable(child)
            # check if it is of basic type
            elif not self._is_basic_type(child.uc_type):
                raise VariableHasCompoundType(child.lvalue_name())

    def visit_Print(self, node: Print) -> None:
        if not node.param:
            return  # Ok, just a newline

        node.param.uc_type = VoidType
        # visit each expression and check if it is of basic type
        for child in node.param.expr:
            uc_type = self.visit(child)
            if not self._is_basic_type(child.uc_type):
                # special errors for variables
                if child.lvalue_name() is not None:
                    raise VariableHasCompoundType(child)
                else:
                    raise ExprHasCompoundType(child)

    def visit_Return(self, node: Return) -> None:
        if node.result:
            ret_type = self.visit(node.result)
        else:
            ret_type = VoidType
        # get function definition
        scope = self.symtab.find(lambda scope: isinstance(scope, FunctionScope))
        if not isinstance(scope, FunctionScope):
            raise ReturnOutsideFunction(node)
        # check that its type is identical to the return type of the function definition
        if ret_type != scope.return_type:
            raise InvalidReturnType(node, scope.return_type)
        # bind to function
        node.bind(scope.block)

    def visit_IterationStmt(self, node: IterationStmt) -> None:
        # create new breakable scope
        with self.symtab.new(IterationScope(node, self.symtab.block)) as scope:
            for _, child in node.children():
                # reuse iteration scope
                if isinstance(child, Compound):
                    uctype = self.visit_Compound(child, scope)
                else:
                    uctype = self.visit(child)
                # check if the conditional expression is of boolean type
                if child is node.condition:
                    if uctype != BoolType:
                        raise InvalidLoopCondition(node.condition, coord=node.coord)

    def visit_For(self, node: For) -> None:
        self.visit_IterationStmt(node)

    def visit_While(self, node: While) -> None:
        self.visit_IterationStmt(node)

    def visit_If(self, node: If) -> None:
        # check if the conditional expression is of boolean type
        cond_type = self.visit(node.condition)
        if cond_type != BoolType:
            raise InvalidConditionalExpression(node.condition)
        # visit statements
        if node.true_stmt:
            self.visit(node.true_stmt)
        if node.false_stmt:
            self.visit(node.false_stmt)

    # # # # # # # #
    # EXPRESSIONS #

    def visit_BinaryOp(self, node: BinaryOp, *, kind: str = "binary_ops") -> uCType:
        # Visit the left and right expression
        ltype = self.visit(node.left)
        rtype = self.visit(node.right)
        # Make sure left and right operands have the same type
        if ltype != rtype:
            raise OperationTypeDoesNotMatch(node)
        # Make sure the operation is supported
        if node.op not in getattr(ltype, kind, {}):
            raise UnsupportedOperation(node)
        # Assign the result type
        return ltype

    def visit_RelationOp(self, node: RelationOp) -> Literal[BoolType]:
        self.visit_BinaryOp(node, kind="rel_ops")
        # comparison results in boolean
        return BoolType

    def visit_Assignment(self, node: Assignment) -> uCType:
        ltype = self.visit_BinaryOp(node, kind="assign_ops")
        rtype = node.right.uc_type

        if isinstance(ltype, ArrayType) and ltype.size is not None and ltype.size != rtype.size:
            raise ArraySizeMismatch(node)

    def visit_UnaryOp(self, node: UnaryOp) -> uCType:
        uctype = self.visit(node.expr)
        # Make sure the operation is supported
        if node.op not in uctype.unary_ops:
            raise UnsupportedOperation(node)
        # Assign the result type
        return uctype

    def visit_ExprList(self, node: ExprList) -> uCType:
        self.generic_visit(node)
        # when used as the comma operator
        return node.as_comma_op().uc_type

    def visit_ArrayRef(self, node: ArrayRef) -> uCType:
        idx_type = self.visit(node.index)
        # index must be 'int'
        if idx_type != IntType:
            raise InvalidSubscriptType(node.index)
        # ltype must be an array type
        uc_type = self.visit(node.array)
        if not isinstance(uc_type, ArrayType):
            raise ExprIsNotAnArray(node.array)
        # check out bounds when possible
        if isinstance(node.index, Constant) and uc_type.out_of_bounds(node.index.value):
            raise IndexOutOfBounds(node.lvalue_name())
        return uc_type.elem_type

    def visit_FuncCall(self, node: FuncCall) -> uCType:
        # ltype must be a function type
        ltype = self.visit(node.callable)
        if not isinstance(ltype, FunctionType):
            raise ExprIsNotAFunction(node.callable)
        # check parameters types and length
        if node.params:
            self.visit(node.params)
        for param, value in zip_longest(ltype.params, node.parameters()):
            if param is None or value is None:
                raise FuncParamsLengthMismatch(node.callable, default=ltype.funcname)
            if value.uc_type != param.type:
                raise InvalidParameterType(param.name, value)

        return ltype.rettype

    # # # # # # # # #
    # BASIC SYMBOLS #

    def visit_ID(self, node: ID, uctype: Optional[uCType] = None) -> uCType:
        if uctype is None:
            # Look for its declaration in the symbol table
            definition = self.symtab.lookup(node.name)
            if definition is None:
                raise UndefinedIdentifier(node)
            # bind identifier to its associated symbol type
            return definition.type

        else:  # initialize the type
            node.uc_type = uctype
            if uctype == VoidType:
                raise InvalidVariableType(node, "a variable")
            if node.name in self.symtab.current_scope:
                raise NameAlreadyDefined(node)
            # add to table
            self.symtab.add(node)
            return uctype

    def _get_primary_type(self, typename: str, coord: Optional[Coord]) -> uCType:
        """Get primary type from type name."""
        uc_type = PrimaryType.get(typename)
        # check if type exists
        if uc_type is None:
            raise UnknownType(typename, coord)
        return uc_type

    def visit_Constant(self, node: Constant) -> Union[PrimaryType, ArrayType]:
        # Get the matching uCType
        if node.rawtype == "string":
            return ArrayType(CharType, len(node.value) + 1)
        else:
            return self._get_primary_type(node.rawtype, node.coord)

    def visit_Type(self, node: Type) -> PrimaryType:
        # Get the matching basic uCType
        return self._get_primary_type(node.name, node.coord)


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
