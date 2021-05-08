from typing import Optional, Sequence, Set


class uCType:
    """
    Class that represents a type in the uC language.  Basic
    Types are declared as singleton instances of this type.
    """

    def __init__(
        self,
        name: Optional[str],
        binary_ops: Set[str] = set(),
        unary_ops: Set[str] = set(),
        rel_ops: Set[str] = set(),
        assign_ops: Set[str] = set(),
    ):
        """
        You must implement yourself and figure out what to store.
        """
        self.typename = name
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.rel_ops = rel_ops
        self.assign_ops = assign_ops


# # # # # # # #
# Basic Types #

IntType = uCType(
    "int",
    unary_ops={"-", "+"},
    binary_ops={"+", "-", "*", "/", "%"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

CharType = uCType(
    "char",
    rel_ops={"==", "!=", "&&", "||"},
    assign_ops={"="},
)


# # # # # # # # # #
# Compound Types  #


class ArrayType(uCType):
    def __init__(self, element_type: uCType, size: Optional[int] = None):
        """
        type: Any of the uCTypes can be used as the array's type. This
              means that there's support for nested types, like matrices.
        size: Integer with the length of the array.
        """
        super().__init__(None, unary_ops={"*", "&"}, rel_ops={"==", "!="})
        self.type = element_type
        self.size = size


class FunctionType(uCType):
    def __init__(self, return_type: Optional[uCType], params: Sequence[uCType]):
        """
        return_type: Only the basic uCTypes can be used here. None is
                     used to indicate the special 'void' type.
        params: Sequence of types for each of the function parameters.
        """
        super().__init__(None)  # only valid operation is call
        self.type = return_type
        self.params = tuple(params)
