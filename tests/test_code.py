from pathlib import Path
import pytest
from uc.uc_code import CodeGenerator
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import Visitor


def resolve_test_files(test_name):
    input_file = test_name + ".in"
    expected_file = test_name + ".out"

    # get current dir
    current_dir = Path(__file__).parent.absolute()

    # get absolute path to inputs folder
    test_folder = current_dir / Path("in-out")

    # get input path and check if exists
    input_path = test_folder / Path(input_file)
    assert input_path.exists()

    # get expected test file real path
    expected_path = test_folder / Path(expected_file)
    assert expected_path.exists()

    return input_path, expected_path


@pytest.mark.parametrize(
    "test_name",
    [
        "t01",
        "t02",
        "t03",
        "t04",
        "t05",
        "t06",
        "t07",
        "t08",
        "t09",
        "t10",
        "t11",
        "t12",
        "t13",
        "t14",
        "t15",
        "t16",
        "t17",
        "t18",
        "t19",
        "t20",
        "t21",
        "t22",
        "t23",
        "t24",
        "t25",
    ],
)
# capfd will capture the stdout/stderr outputs generated during the test
def test_code(test_name, capsys):
    input_path, expected_path = resolve_test_files(test_name)

    p = UCParser(debug=False)
    with open(input_path) as f_in, open(expected_path) as f_ex:
        ast = p.parse(f_in.read())
        sema = Visitor()
        sema.visit(ast)
        gen = CodeGenerator(False)
        gen.visit(ast)
        gencode = gen.code
        vm = Interpreter(False)
        with pytest.raises(SystemExit) as sys_error:
            vm.run(gencode)
        captured = capsys.readouterr()
        assert sys_error.value.code == 0
        expect = f_ex.read()
    assert captured.out == expect
    assert captured.err == ""
