import argparse
import pathlib
import sys
from ply.yacc import yacc
from uc.asttypes import *
from uc.uc_lexer import UCLexer


class UCParser:
    def __init__(self, debug=True):
        """Create a new uCParser."""
        self.uclex = UCLexer(self._lexer_error)
        self.uclex.build()
        self.tokens = self.uclex.tokens

        self.ucparser = yacc(module=self, start="program", debug=debug)
        # Keeps track of the last token given to yacc (the lookahead token)
        self._last_yielded_token = None

    def show_parser_tree(self, text):
        print(self.parse(text))

    def parse(self, text, debuglevel=0):
        self.uclex.reset_lineno()
        self._last_yielded_token = None
        return self.ucparser.parse(input=text, lexer=self.uclex, debug=debuglevel)

    def _lexer_error(self, msg, line, column):
        # use stderr to match with the output in the .out test files
        print("LexerError: %s at %d:%d" % (msg, line, column), file=sys.stderr)
        sys.exit(1)

    def _parser_error(self, msg, line="", column=""):
        # use stderr to match with the output in the .out test files
        if line == "" and column == "":
            print("ParserError: %s" % (msg), file=sys.stderr)
        if column == "":
            print("ParserError: %s at %s" % (msg, line), file=sys.stderr)
        else:
            print("ParserError: %s at %s:%s" % (msg, line, column), file=sys.stderr)
        sys.exit(1)

    # # # # # # # # #
    # DECLARATIONS  #

    def p_program(self, p):
        """program  : global_declaration_list"""
        p[0] = Program(p[1])

    def p_global_declaration_list(self, p):
        """global_declaration_list  : global_declaration
        | global_declaration_list global_declaration
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[2]]

    def p_global_declaration(self, p):
        """global_declaration   : function_definition
        | declaration
        """
        p[0] = p[1]

    def p_function_definition(self, p):
        """function_definition  : type_specifier declarator declaration_list compound_statement"""
        p[0] = FunctionDef(p[1], p[2], p[3], p[4])

    def p_declaration_list(self, p):
        """declaration_list :
        | declaration_list declaration
        """
        p[0] = p[1] + [p[2]] if len(p) > 1 else []

    def p_declarator(self, p):
        """declarator   : identifier
        | LPAREN declarator RPAREN
        | declarator LBRACKET                     RBRACKET
        | declarator LBRACKET constant_expression RBRACKET
        | declarator LPAREN                RPAREN
        | declarator LPAREN parameter_list RPAREN
        """
        # simple identifier
        if len(p) == 2 or p[1] == "(":
            p[0] = p[1] if len(p) == 2 else p[2]
        # array declaration
        elif p[2] == "[":
            p[0] = ArrayDeclarator(p[1], p[3] if len(p) == 5 else None)
        # function declaration
        else:
            p[0] = FunctionDeclarator(p[1], p[3] if len(p) == 5 else [])

    def p_parameter_list(self, p):
        """parameter_list   : parameter_declaration
        | parameter_list COMMA parameter_declaration
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[3]]

    def p_parameter_declaration(self, p):
        """parameter_declaration    : type_specifier declarator"""
        p[0] = Parameter(p[1], p[2])

    def p_declaration(self, p):
        """declaration  : type_specifier                      SEMI
        | type_specifier init_declarator_list SEMI
        """
        p[0] = Declaration(p[1], p[2] if len(p) > 3 else [])

    def p_init_declarator_list(self, p):
        """init_declarator_list : init_declarator
        | init_declarator_list COMMA init_declarator
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[3]]

    def p_init_declarator(self, p):
        """init_declarator  : declarator
        | declarator EQUALS initializer
        """
        p[0] = InitDeclarator(p[1], p[3] if len(p) > 2 else None)

    def p_initializer(self, p):
        """initializer  : assignment_expression
        | LBRACE                        RBRACE
        | LBRACE initializer_list       RBRACE
        | LBRACE initializer_list COMMA RBRACE
        """
        # simple initializer
        if len(p) == 2:
            p[0] = p[1]
        # array initializer
        else:
            p[0] = ArrayInit(p[2] if len(p) > 3 else [])

    def p_initializer_list(self, p):
        """initializer_list : initializer
        | initializer_list COMMA initializer
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[3]]

    # # # # # # # #
    # STATEMENTS  #

    def p_compound_statement(self, p):
        """compound_statement   : LBRACE declaration_list statement_list RBRACE"""
        p[0] = CompoundStmt(p[2], p[3])

    def p_statement_list(self, p):
        """statement_list   :
        | statement_list statement
        """
        p[0] = p[1] + [p[2]] if len(p) > 1 else []

    def p_statement(self, p):
        """statement    : expression_statement
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
        """selection_statement  : IF LPAREN expression RPAREN statement
        | IF LPAREN expression RPAREN statement ELSE statement
        """
        p[0] = IfStmt(p[3], p[5], p[7] if len(p) == 8 else None).set_lineinfo(p)

    def p_iteration_statement(self, p):
        """iteration_statement  : WHILE LPAREN expression RPAREN statement
        | FOR LPAREN maybe_expression SEMI maybe_expression SEMI maybe_expression RPAREN statement
        | FOR LPAREN declaration           maybe_expression SEMI maybe_expression RPAREN statement
        """
        if len(p) == 6:
            p[0] = WhileStmt(p[3], p[5]).set_lineinfo(p)
        elif len(p) == 10:
            p[0] = ForStmt(p[3], p[5], p[7], p[9]).set_lineinfo(p)
        else:
            p[0] = ForStmt(p[3], p[4], p[6], p[8]).set_lineinfo(p)

    def p_jump_statement(self, p):
        """jump_statement   : BREAK                   SEMI
        | RETURN maybe_expression SEMI
        """
        if len(p) == 3:
            p[0] = BreakStmt().set_lineinfo(p)
        else:
            p[0] = ReturnStmt(p[2]).set_lineinfo(p)

    def p_assert_statement(self, p):
        """assert_statement : ASSERT expression SEMI"""
        p[0] = AssertStmt(p[2]).set_lineinfo(p)

    def p_print_statement(self, p):
        """print_statement  : PRINT LPAREN maybe_expression RPAREN SEMI"""
        p[0] = PrintStmt(p[3]).set_lineinfo(p)

    def p_read_statement(self, p):
        """read_statement   : READ LPAREN argument_expression RPAREN SEMI"""
        p[0] = ReadStmt(p[3]).set_lineinfo(p)

    # # # # # # # #
    # EXPRESSIONS #

    def p_maybe_expression(self, p):
        """maybe_expression :
        | expression
        """
        p[0] = p[1] if len(p) == 2 else None

    def p_expression(self, p):
        """expression   : assignment_expression
        | expression COMMA assignment_expression
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[3]]

    def p_assignment_expression(self, p):
        """assignment_expression    : binary_expression
        | unary_expression EQUALS assignment_expression
        """
        if len(p) > 2:
            p[0] = AssignExpr(p[1], p[3])
        else:
            p[0] = p[1]

    def p_argument_expression(self, p):
        """argument_expression  : assignment_expression
        | argument_expression COMMA assignment_expression
        """
        p[0] = [p[1]] if len(p) == 2 else p[1] + [p[3]]

    def p_binary_expression(self, p):
        """binary_expression    : unary_expression
        | binary_expression TIMES  binary_expression
        | binary_expression DIVIDE binary_expression
        | binary_expression  MOD   binary_expression
        | binary_expression  PLUS  binary_expression
        | binary_expression MINUS  binary_expression
        | binary_expression   LT   binary_expression
        | binary_expression   LE   binary_expression
        | binary_expression   GT   binary_expression
        | binary_expression   GE   binary_expression
        | binary_expression   EQ   binary_expression
        | binary_expression   NE   binary_expression
        | binary_expression  AND   binary_expression
        | binary_expression   OR   binary_expression
        """
        if len(p) > 2:
            op = Operator.from_token((None, p[2]), set_info=False)
            p[0] = BinOp(op, p[1], p[3])
        else:
            p[0] = p[1]

    precedence = (
        ("left", "COMMA"),
        ("left", "EQUALS"),
        ("left", "OR"),
        ("left", "AND"),
        ("left", "EQ", "NE"),
        ("left", "LE", "LT", "GE", "GT"),
        ("left", "PLUS", "MINUS"),
        ("left", "TIMES", "DIVIDE", "MOD"),
        ("right", "NOT"),
    )

    def p_unary_expression(self, p):
        """unary_expression : postfix_expression
        | unary_operator unary_expression
        """
        if len(p) > 2:
            p[0] = UnOp(p[1], p[2])
        else:
            p[0] = p[1]

    def p_postfix_expression(self, p):
        """postfix_expression   : primary_expression
        | postfix_expression LBRACKET expression RBRACKET
        | postfix_expression LPAREN                     RPAREN
        | postfix_expression LPAREN argument_expression RPAREN
        """
        if len(p) == 2:
            p[0] = p[1]
        elif p[2] == "(":
            p[0] = CallExpr(p[1], p[3] if len(p) == 5 else [])
        else:
            p[0] = AccessExpr(p[1], p[3])

    def p_primary_expression(self, p):
        """primary_expression   : identifier
        | constant
        | string
        | LPAREN expression RPAREN
        """
        p[0] = p[2] if len(p) == 4 else p[1]

    def p_constant_expression(self, p):
        """constant_expression  : binary_expression"""
        p[0] = p[1]

    def p_constant(self, p):
        """constant : integer_constant
        | character_constant
        """
        p[0] = p[1]

    # # # # # # # # # # #
    # TERMINAL  SYMBOLS #

    def p_type_specifier(self, p):
        """type_specifier   : VOID
                            | CHAR
                            | INT
        """
        p[0] = TypeSpec.from_token(p)

    def p_unary_operator(self, p):
        """unary_operator   : PLUS
                            | MINUS
                            | NOT
        """
        p[0] = Operator.from_token(p)

    def p_integer_constant(self, p):
        """integer_constant : INT_CONST"""
        p[0] = Int.from_token(p)

    def p_character_constant(self, p):
        """character_constant   : CHAR_CONST"""
        p[0] = Char.from_token(p)

    def p_identifier(self, p):
        """identifier   : ID"""
        p[0] = Ident.from_token(p)

    def p_string(self, p):
        """string   : STRING_LITERAL"""
        p[0] = String.from_token(p)

    # # # # # #
    # ERRORS  #

    def p_error(self, p):
        if p:
            self._parser_error(
                "Before: %s" % p.value, p.lineno, self.uclex.find_tok_column(p)
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
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and print tokens
    with open(input_path) as f:
        # p.parse(f.read())
        # use show_parser_tree instead of parser to print it
        p.show_parser_tree(f.read())
