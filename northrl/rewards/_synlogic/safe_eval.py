# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded parsers for model-controlled SynLogic answers."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from numbers import Real
from typing import Any, cast

_MAX_EXPRESSION_LENGTH = 4096
_MAX_LITERAL_LENGTH = 1_000_000
_MAX_AST_NODES = 4096

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Real, Real], Real]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}


def _unary_plus(value: Real) -> Real:
    """Apply unary plus while preserving the evaluator's ``Real`` contract."""
    return cast(Real, operator.pos(value))


def _unary_minus(value: Real) -> Real:
    """Apply unary minus while preserving the evaluator's ``Real`` contract."""
    return cast(Real, operator.neg(value))


_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Real], Real]] = {
    ast.UAdd: _unary_plus,
    ast.USub: _unary_minus,
}
_COMPARISON_OPERATORS: dict[type[ast.cmpop], Callable[[Real, Real], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def literal(value: str) -> Any:
    """Parse a bounded Python literal without executing model-provided code."""
    if not isinstance(value, str) or len(value) > _MAX_LITERAL_LENGTH:
        raise ValueError("SynLogic literal is missing or too large")
    parsed = ast.literal_eval(value)
    if sum(1 for _ in ast.walk(ast.parse(value, mode="eval"))) > _MAX_AST_NODES:
        raise ValueError("SynLogic literal is too complex")
    return parsed


def arithmetic(value: str) -> Real:
    """Evaluate bounded arithmetic using only numeric constants and five operators."""
    if not isinstance(value, str) or len(value) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("SynLogic expression is missing or too large")
    tree = ast.parse(value, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError("SynLogic expression is too complex")

    def visit(node: ast.AST) -> Real:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, Real)
            and not isinstance(node.value, bool)
        ):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        raise ValueError(f"Unsupported SynLogic expression node: {type(node).__name__}")

    return visit(tree)


def comparison(value: str) -> bool:
    """Evaluate a bounded arithmetic comparison without names, calls, or attributes."""
    if not isinstance(value, str) or len(value) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("SynLogic comparison is missing or too large")
    tree = ast.parse(value, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise ValueError("SynLogic comparison is too complex")
    body = tree.body
    if not isinstance(body, ast.Compare) or len(body.ops) != 1 or len(body.comparators) != 1:
        raise ValueError("SynLogic answer must contain one comparison")
    operation = _COMPARISON_OPERATORS.get(type(body.ops[0]))
    if operation is None:
        raise ValueError("Unsupported SynLogic comparison operator")

    def arithmetic_node(node: ast.AST) -> Real:
        return arithmetic(ast.unparse(node))

    return operation(arithmetic_node(body.left), arithmetic_node(body.comparators[0]))
