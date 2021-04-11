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

    precedence = ()

    def p_program(self, p):
        """program  : global_declaration_list"""
        pass

    def p_global_declaration_list(self, p):
        """global_declaration_list : global_declaration
        | global_declaration_list global_declaration
        """
        pass
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
