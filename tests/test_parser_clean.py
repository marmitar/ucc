from pathlib import Path
import pytest
from uc.uc_parser_clean import UCParser


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


@pytest.mark.parametrize("test_name", ["t01"])
# capfd will capture the stdout/stderr outputs generated during the test
def test_parser_warning(test_name, capfd):
    input_path, expected_path = resolve_test_files(test_name)

    p = UCParser(debug=True)
    with open(input_path) as f_in, open(expected_path) as f_ex:
        p.show_parser_tree(f_in.read())
        captured = capfd.readouterr()
        expect = f_ex.read()
    assert captured.err == "" or captured.err == expect
    assert captured.out != "" and captured.out != "None\n"


@pytest.mark.parametrize(
    "test_name",
    [
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
        "t19",
        "t20",
        "t24",
        "t25",
        "t26",
        "t27",
        "t28",
        "t29",
        "t30",
        "t31",
        "t32",
        "t33",
        "t34",
        "t35",
        "t36",
        "t37",
        "t38",
        "t39",
        "t40",
    ],
)
# capfd will capture the stdout/stderr outputs generated during the test
def test_parser(test_name, capfd):
    input_path, expected_path = resolve_test_files(test_name)

    p = UCParser(debug=False)
    with open(input_path) as f_in, open(expected_path) as f_ex:
        p.show_parser_tree(f_in.read())
        captured = capfd.readouterr()
        expect = f_ex.read()
    assert captured.out != "" and captured.out != "None\n"
    assert captured.err == expect


@pytest.mark.parametrize("test_name", ["t15", "t16", "t17", "t18",  "t21", "t22", "t23"])
# capfd will capture the stdout/stderr outputs generated during the test
def test_parser_error(test_name, capfd):
    input_path, expected_path = resolve_test_files(test_name)

    p = UCParser(debug=False)
    with open(input_path) as f_in, open(expected_path) as f_ex:
        with pytest.raises(SystemExit) as sys_error:
            p.show_parser_tree(f_in.read())
        assert sys_error.value.code == 1
        captured = capfd.readouterr()
        expect = f_ex.read()
    assert captured.out == ""
    assert captured.err == expect
