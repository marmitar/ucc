from argparse import ArgumentParser, FileType
from typing import TextIO

from . import UCLexer

# create argument parser
parser = ArgumentParser(description="the uC lexer")
parser.add_argument("input_file", help="Path to file to be scanned", type=FileType("rt"))
args = parser.parse_args()

# get input text
input_file: TextIO = args.input_file
source_text = input_file.read()
input_file.close()

# set error function
lexer = UCLexer(on_error=UCLexer.ABORT)
# open file and print tokens
for token in lexer.tokenize(source_text):
    print(token)
