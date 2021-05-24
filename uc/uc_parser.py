import argparse
import pathlib
import sys
from typing import (
    List,
    NoReturn,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    overload,
)
from ply.yacc import yacc
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
    DeclList,
    ExprList,
    For,
    FuncCall,
    FuncDecl,
    FuncDef,
    GlobalDecl,
    If,
    InitList,
    Modifier,
    Node,
    ParamList,
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
from uc.uc_lexer import UCLexer


class Coord:
    """Coordinates of a syntactic element. Consists of:
    - Line number
    - (optional) column number, for the Lexer
    """

    __slots__ = ("line", "column")

    def __init__(self, line: int, column: Optional[int] = None):
        self.line = line
        self.column = column

    def __str__(self) -> str:
        if self.line and self.column is not None:
            coord_str = "@ %s:%s" % (self.line, self.column)
        elif self.line:
            coord_str = "@ %s" % (self.line)
        else:
            coord_str = ""
        return coord_str


T = TypeVar("T")
U = TypeVar("U")

# fmt: off
@overload
def getitem(seq: Sequence[T], index: int) -> Optional[T]: ...
@overload
def getitem(seq: Sequence[T], index: int, default: U) -> Union[T, U]: ...
# fmt: on
def getitem(seq: Sequence[T], index: int, default: U = None) -> Union[T, U]:
    """'getattr'-like helper for sequences"""
    if 0 <= index < len(seq):
        return seq[index]
    else:
        return default


Declaration = Union[VarDecl, Modifier]


class DeclSpec(TypedDict):
    decl: Declaration
    init: Optional[Node]


class UCParser:
    def __init__(self, debug: bool = True):
        """Create a new uCParser."""
        self.uclex = UCLexer(self._lexer_error)
        self.uclex.build()
        self.tokens = self.uclex.tokens

        self.ucparser = yacc(module=self, start="program", debug=debug)
        # Keeps track of the last token given to yacc (the lookahead token)
        self._last_yielded_token = None

    def parse(self, text: str, debuglevel: int = 0) -> Program:
        self.uclex.reset_lineno()
        self._last_yielded_token = None
        return self.ucparser.parse(input=text, lexer=self.uclex, debug=debuglevel)

    def _lexer_error(self, msg: str, line: int, column: int) -> NoReturn:
        # use stdout to match with the output in the .out test files
        print("LexerError: %s at %d:%d" % (msg, line, column), file=sys.stdout)
        sys.exit(1)

    def _parser_error(self, msg: str, coord: Optional[Coord] = None) -> NoReturn:
        # use stdout to match with the output in the .out test files
        if coord is None:
            print("ParserError: %s" % (msg), file=sys.stdout)
        else:
            print("ParserError: %s %s" % (msg, coord), file=sys.stdout)
        sys.exit(1)

    def _token_coord(self, p, token_idx: int) -> Coord:
        last_cr = p.lexer.lexer.lexdata.rfind("\n", 0, p.lexpos(token_idx))
        if last_cr < 0:
            last_cr = -1
        column = p.lexpos(token_idx) - (last_cr)
        return Coord(p.lineno(token_idx), column)

    def _build_declarations(self, spec: Optional[Type], decls: List[DeclSpec]) -> List[Decl]:
        """Builds a list of declarations all sharing the given specifiers."""
        declarations = []

        for decl in decls:
            declaration = Decl(decl["decl"], decl.get("init"))

            fixed_decl = self._fix_decl_name_type(declaration, spec)
            declarations.append(fixed_decl)

        return declarations

    def _fix_decl_name_type(self, decl: Decl, typename: Optional[Type]) -> Decl:
        """Fixes a declaration. Modifies decl."""
        # Reach the underlying basic type
        type = decl
        while not isinstance(type, VarDecl):
            type = type.type

        decl.name = type.declname
        if not typename:
            # Functions default to returning int
            if not isinstance(decl.type, FuncDecl):
                self._parser_error("Missing type in declaration", decl.coord)
            type.type = Type("int", coord=decl.coord)
        else:
            type.type = Type(typename.name, coord=typename.coord)

        return decl

    def _type_modify_decl(self, decl: Declaration, modifier: Modifier) -> Declaration:
        """Tacks a type modifier on a declarator, and returns
        the modified declarator.
        Note: the declarator and modifier may be modified
        """
        modifier_head = modifier
        modifier_tail = modifier

        # The modifier may be a nested list. Reach its tail.
        while modifier_tail.type:
            modifier_tail = modifier_tail.type

        # If the decl is a basic type, just tack the modifier onto it
        if isinstance(decl, VarDecl):
            modifier_tail.set_type(decl)
            return modifier
        else:
            # Otherwise, the decl is a list of modifiers. Reach
            # its tail and splice the modifier onto the tail,
            # pointing to the underlying basic type.
            decl_tail = decl

            while not isinstance(decl_tail.type, VarDecl):
                decl_tail = decl_tail.type

            modifier_tail.set_type(decl_tail.type)
            decl_tail.set_type(modifier_head)
            return decl

    # # # # # # # # #
    # DECLARATIONS  #

    def p_program(self, p):
        """program : global_declaration_list"""
        p[0] = Program(p[1])

    def p_global_declaration_list(self, p):
        """global_declaration_list : global_declaration
        | global_declaration_list global_declaration
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_global_declaration_1(self, p):
        """global_declaration    : declaration"""
        p[0] = GlobalDecl(p[1])

    def p_global_declaration_2(self, p):
        """global_declaration    : function_definition"""
        p[0] = p[1]

    def p_function_definition(self, p):
        """function_definition : type_specifier declarator declaration_list compound_statement"""
        decl = self._build_declarations(p[1], [{"decl": p[2]}])
        p[0] = FuncDef(p[1], decl[0], p[3], p[4])

    def p_declaration_list(self, p):
        """declaration_list :
        | declaration_list declaration
        """
        if len(p) == 1:
            p[0] = DeclList()
        else:
            p[1].extend(p[2])
            p[0] = p[1]

    def p_declarator(self, p):
        """declarator : identifier
        | LPAREN declarator RPAREN
        | declarator LBRACKET                     RBRACKET
        | declarator LBRACKET constant_expression RBRACKET
        | declarator LPAREN                RPAREN
        | declarator LPAREN parameter_list RPAREN
        """
        # simple identifier
        if len(p) == 2:
            p[0] = VarDecl(p[1])
        elif p[1] == "(":
            p[0] = p[2]
        # array declaration
        elif p[2] == "[":
            mod = ArrayDecl(p[3] if len(p) == 5 else None)
            p[0] = self._type_modify_decl(p[1], mod)
        # function declaration
        else:
            params = p[3] if len(p) == 5 else ParamList()
            params.coord = self._token_coord(p, 2)
            # use locate parameters at parenthesis
            p[0] = self._type_modify_decl(p[1], FuncDecl(params))

    def p_parameter_list(self, p):
        """parameter_list :    parameter_declaration
        | parameter_list COMMA parameter_declaration
        """
        if len(p) == 2:
            p[0] = ParamList(p[1])
        else:
            p[1].append(p[3])
            p[0] = p[1]

    def p_parameter_declaration(self, p):
        """parameter_declaration : type_specifier declarator"""
        decl = self._build_declarations(p[1], [{"decl": p[2]}])
        p[0] = decl[0]

    def p_declaration(self, p):
        """declaration  : type_specifier      SEMI
        | type_specifier init_declarator_list SEMI
        """
        decls = self._build_declarations(p[1], getitem(p, 2, []))
        p[0] = DeclList(decls)

    def p_init_declarator_list(self, p):
        """init_declarator_list :    init_declarator
        | init_declarator_list COMMA init_declarator
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[3]]

    def p_init_declarator(self, p):
        """init_declarator : declarator
        | declarator EQUALS initializer
        """
        p[0] = dict(decl=p[1], init=getitem(p, 3))

    def p_initializer(self, p):
        """initializer : assignment_expression
        | LBRACE                        RBRACE
        | LBRACE initializer_list       RBRACE
        | LBRACE initializer_list COMMA RBRACE
        """
        # simple initializer
        if len(p) == 2:
            p[0] = p[1]
        # array initializer
        else:
            p[0] = p[2] if len(p) > 3 else None

    def p_initializer_list(self, p):
        """initializer_list :    initializer
        | initializer_list COMMA initializer
        """
        if len(p) == 2:
            p[0] = InitList(p[1])
        else:
            p[1].append(p[3])
            p[0] = p[1]

    # # # # # # # #
    # STATEMENTS  #

    def p_compound_statement(self, p):
        """compound_statement : LBRACE declaration_list statement_list RBRACE"""
        coord = self._token_coord(p, 1)
        p[0] = Compound(p[2], p[3], coord)

    def p_statement_list(self, p):
        """statement_list :
        | statement_list statement
        """
        p[0] = p[1] + [p[2]] if len(p) > 1 else []

    def p_statement(self, p):
        """statement : expression_statement
        | compound_statement
        | selection_statement
        | iteration_statement
        | jump_statement
        | assert_statement
        | print_statement
        | read_statement
        """
        p[0] = p[1]

    def p_expression_statement(self, p):
        """expression_statement : maybe_expression SEMI"""
        p[0] = p[1]

    def p_selection_statement(self, p):
        """selection_statement : IF LPAREN expression RPAREN statement
        | IF LPAREN expression RPAREN statement ELSE statement
        """
        coord = self._token_coord(p, 1)
        p[0] = If(p[3], p[5], getitem(p, 7), coord)

    def p_iteration_statement_1(self, p):
        """iteration_statement : WHILE LPAREN expression RPAREN statement
        | FOR LPAREN maybe_expression SEMI maybe_expression SEMI maybe_expression RPAREN statement
        | FOR LPAREN declaration maybe_expression SEMI maybe_expression RPAREN statement
        """
        coord = self._token_coord(p, 1)
        if len(p) == 6:
            p[0] = While(p[3], p[5], coord)
        elif len(p) == 10:
            p[0] = For(p[3], p[5], p[7], p[9], coord)
        else:
            p[0] = For(p[3], p[4], p[6], p[8], coord)

    def p_jump_statement(self, p):
        """jump_statement : BREAK SEMI
        | RETURN maybe_expression SEMI
        """
        coord = self._token_coord(p, 1)
        if len(p) == 3:
            p[0] = Break(coord)
        else:
            p[0] = Return(p[2], coord)

    def p_assert_statement(self, p):
        """assert_statement : ASSERT expression SEMI"""
        coord = self._token_coord(p, 1)
        p[0] = Assert(p[2], coord)

    def p_print_statement(self, p):
        """print_statement : PRINT LPAREN maybe_expression RPAREN SEMI"""
        coord = self._token_coord(p, 1)
        p[0] = Print(p[3], coord)

    def p_read_statement(self, p):
        """read_statement : READ LPAREN argument_expression RPAREN SEMI"""
        coord = self._token_coord(p, 1)
        p[0] = Read(p[3], coord)

    # # # # # # # #
    # EXPRESSIONS #

    def p_maybe_expression(self, p):
        """maybe_expression :
        | expression
        """
        p[0] = getitem(p, 1)

    def p_expression(self, p):
        """expression  : assignment_expression
        | expression COMMA assignment_expression
        """
        # single expression
        if len(p) == 2:
            p[0] = ExprList(p[1])
        # multiple expressions
        else:
            p[1].append(p[3])
            p[0] = p[1]

    def p_assignment_expression(self, p):
        """assignment_expression : binary_expression
        | unary_expression EQUALS assignment_expression
        """
        if len(p) > 2:
            p[0] = Assignment(p[2], p[1], p[3])
        else:
            p[0] = p[1]

    def p_argument_expression(self, p):
        """argument_expression :    assignment_expression
        | argument_expression COMMA assignment_expression
        """
        # single expression
        if len(p) == 2:
            p[0] = ExprList(p[1])
        # multiple expressions
        else:
            p[1].append(p[3])
            p[0] = p[1]

    def p_binary_expression_1(self, p):
        """binary_expression : unary_expression
        | binary_expression TIMES  binary_expression
        | binary_expression DIVIDE binary_expression
        | binary_expression  MOD   binary_expression
        | binary_expression  PLUS  binary_expression
        | binary_expression MINUS  binary_expression
        | binary_expression  AND   binary_expression
        | binary_expression   OR   binary_expression
        """
        if len(p) > 2:
            p[0] = BinaryOp(p[2], p[1], p[3])
        else:
            p[0] = p[1]

    def p_binary_expression_2(self, p):
        """binary_expression : binary_expression LT binary_expression
        | binary_expression  LE  binary_expression
        | binary_expression  GT  binary_expression
        | binary_expression  GE  binary_expression
        | binary_expression  EQ  binary_expression
        | binary_expression  NE  binary_expression
        """
        p[0] = RelationOp(p[2], p[1], p[3])

    precedence = (
        ("left", "COMMA"),
        ("left", "EQUALS"),
        ("left", "OR"),
        ("left", "AND"),
        ("left", "EQ", "NE"),
        ("left", "LE", "LT", "GE", "GT"),
        ("left", "PLUS", "MINUS", "TIMES"),
        ("left", "TIMES", "DIVIDE", "MOD"),
        ("right", "NOT"),
    )

    def p_unary_expression(self, p):
        """unary_expression : postfix_expression
        | unary_operator unary_expression
        """
        if len(p) > 2:
            p[0] = UnaryOp(p[1], p[2])
        else:
            p[0] = p[1]

    def p_postfix_expression(self, p):
        """postfix_expression  : primary_expression
        | postfix_expression LBRACKET expression RBRACKET
        | postfix_expression LPAREN                     RPAREN
        | postfix_expression LPAREN argument_expression RPAREN
        """
        if len(p) == 2:
            p[0] = p[1]
        elif p[2] == "[":
            p[0] = ArrayRef(p[1], p[3])
        elif len(p) == 5:
            p[0] = FuncCall(p[1], p[3])
        else:
            p[0] = FuncCall(p[1])

    def p_primary_expression(self, p):
        """primary_expression : identifier
        | constant
        | string
        | LPAREN expression RPAREN
        """
        p[0] = getitem(p, 2, p[1])

    def p_constant_expression(self, p):
        """constant_expression : binary_expression"""
        p[0] = p[1]

    def p_constant(self, p):
        """constant : integer_constant
        | character_constant
        | boolean_constant
        | float_constant
        """
        p[0] = p[1]

    # # # # # # # # #
    # BASIC SYMBOLS #

    def p_type_specifier(self, p):
        """type_specifier : VOID
        | CHAR
        | INT
        | BOOL
        | FLOAT
        """
        coord = self._token_coord(p, 1)
        p[0] = Type(p[1], coord)

    def p_unary_operator(self, p):
        """unary_operator : PLUS
        | MINUS
        | NOT
        | TIMES
        """
        p[0] = p[1]

    def p_integer_constant(self, p):
        """integer_constant : INT_CONST"""
        coord = self._token_coord(p, 1)
        p[0] = Constant("int", int(p[1]), coord)

    def p_character_constant(self, p):
        """character_constant : CHAR_CONST"""
        coord = self._token_coord(p, 1)
        p[0] = Constant("char", p[1], coord)

    def p_boolean_constant(self, p):
        """boolean_constant : TRUE
        | FALSE
        """
        coord = self._token_coord(p, 1)
        p[0] = Constant("bool", p[1] == "true", coord)

    def p_float_constant(self, p):
        """float_constant : FLOAT_CONST"""
        coord = self._token_coord(p, 1)
        p[0] = Constant("float", float(p[1]), coord)

    def p_identifier(self, p):
        """identifier : ID"""
        coord = self._token_coord(p, 1)
        p[0] = ID(p[1], coord)

    def p_string(self, p):
        """string : STRING_LITERAL"""
        coord = self._token_coord(p, 1)
        p[0] = Constant("string", p[1], coord)

    # # # # # #
    # ERRORS  #

    def p_error(self, p) -> NoReturn:
        if p:
            self._parser_error(
                "Before %s" % p.value, Coord(p.lineno, self.uclex.find_tok_column(p))
            )
        else:
            self._parser_error("At the end of input (%s)" % self.uclex.filename)


if __name__ == "__main__":

    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to file to be parsed", type=str)
    args = parser.parse_args()

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("ERROR: Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    def print_error(msg, x, y):
        print("Lexical error: %s at %d:%d" % (msg, x, y), file=sys.stderr)

    # set error function
    p = UCParser()
    # open file and print tokens
    with open(input_path) as f:
        program = p.parse(f.read())
        program.show(attrnames=True, nodenames=True, showcoord=True)
