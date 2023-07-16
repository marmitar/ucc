from argparse import ArgumentParser, FileType
from typing import TextIO

from result import Err, Ok

from ..common import results
from . import UCLexer

# create argument parser
parser = ArgumentParser(description="the uC lexer")
parser.add_argument("input_file", help="Path to file to be scanned", type=FileType("rt"))
args = parser.parse_args()

# get input text
input_file: TextIO = args.input_file
source_text = input_file.read()
input_file.close()

# start lexer
lexer = UCLexer()
tokens = lexer.tokenize(source_text)

# print tokens and errors
for result in results(tokens, ErrorType=UCLexer.Error):
    match result:
        case Ok(token):
            print(token)
        case Err(error):
            print("Lexical error:", error, flush=True)
