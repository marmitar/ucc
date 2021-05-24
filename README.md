# Code Generation

The objective of this assignment is transform the AST into uCIR and to run the uCIR with the interpreter.
Further instructions can be seen in this
[link](https://github.com/MC921-1s21/notebooks-1s21/blob/master/P5-CodeGeneration.ipynb).

## Tasks

You should do the following tasks:

- [X] Paste your implementation of the lexer in `uc/uc_lexer.py`
- [ ] Paste your implementation of the parser in `uc/uc_parser.py`
- [ ] Paste your implementation of the ast in `uc/uc_ast.py`
- [ ] Paste your implementation of the semantic analysis in `uc/uc_sema.py`
- [ ] Paste your implementation of the type system in `uc/uc_type.py`
- [ ] Complete the implementation of the code generation in `uc/uc_code.py`
- [ ] Complete the implementation of basic blocks in `uc/uc_block.py`

Feel free to add modifications to files from previous projects such as `uc_type.py` or `uc_ast.py` to compliment your solution.

## Requirements

Use Python 3.5 or a newer version.
Required pip packages:
- ply, pytest, setuptools, graphviz

## Running

After you have accepted this assignment on the course's Github Classroom page,
clone it to your machine.

You can run `uc_code.py` directly with python. For more information, run:
```sh
    python3 uc/uc_code.py -h
```
You can use the inputs available inside
the `tests/in-out/` directory.

The `uc_compiler.py` script is also available, it can be run through its
symbolic link at the root of the repo (`ucc`). For more information, run:
```sh
    ./ucc -h
```

## Testing with Pytest

You can run all the tests in `tests/in-out/` automatically with `pytest`. For
that, you first need to make the source files visible to the tests. There are
two options:
- Install your project in editable mode using the `setup.py` file from the root
  of the repo
```sh
    pip install -e .
```
- Or, add the repo folder to the PYTHONPATH environment variable with `setup.sh`
```sh
    source setup.sh
```

Then you should be able to run all the tests by running `pytest` at the root
of the repo.

### Linting and Formatting

This step is **optional**. Required pip packages:
- flake8, black, isort

You can lint your code with two `flake8` commands from the root of the repo:
```sh
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 . --count --exit-zero --max-line-length=120 --statistics
```

The first command shows errors that need to be fixed and will cause your
commit to fail. The second shows only warnings that are suggestions for
a good coding style.

To format the code, you can use `isort` to manage the imports and `black`
for the rest of the code. Run both from the root of the repo:
```sh
    isort .
    black .
```

### Using Pre-Commit Hooks

Linting can be done automatically before every commit using
[git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks).
Required pip packages:
- pre-commit

Liniting hooks can be enabled with:
```sh
    pre-commit install --install-hooks
```

## Grading

Your assignment will be evaluated in terms of:

- Correctness: your program returns correct results for the tests;

Whenever you push your changes to Github, Github Actions will lint and run your
implementation with `pytest` to test all the inputs in `tests/in-out`.
Your grade will be automatically determined by the autograding job.

To check your grade online:
- Go to the `Actions` tab in your repo
- Click on the latest commit
- Click on `build` on the left panel
    - This will show all the steps on the Autograding CI
- Click on the `Run tests with autograding` job and scroll to the bottom

You **must not** modify the test files.

**Note:** The automatic grading system expects your program's output to be
formatted correctly. For that reason, you should not add `print()` or any other
functions that write to `stdout`, otherwise, your assignment will not be graded
correctly.

**Note:** The final grade for this assignment will be determined by the lastest
commit before the deadline, and it will not use Github's autograding.
An internal grading script will be run instead to prevent cheating.

## Questions

If you have any doubts or run into problems, please contact the TAs.
Happy coding! :smile: :keyboard:

## Contribute

Found a typo? Something is missing or broken? Have ideas for improvement? The
instructor and the TAs would love to hear from you!

## About

This repository is one of the assignments handed out to the students in the course
"MC921 - Compiler Construction" offered by the Institute of
Computing at Unicamp.
