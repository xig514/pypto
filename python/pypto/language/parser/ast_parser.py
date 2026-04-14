# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST parsing for converting Python DSL to IR builder calls."""

import ast
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pypto.ir import IRBuilder
from pypto.ir import op as ir_op
from pypto.pypto_core import DataType, ir
from pypto.pypto_core.ir import MemorySpace, PipeType

_MEMORY_SPACE_MAP: dict[str, MemorySpace] = {
    "Left": MemorySpace.Left,
    "Right": MemorySpace.Right,
    "Vec": MemorySpace.Vec,
    "Mat": MemorySpace.Mat,
    "Acc": MemorySpace.Acc,
}

from .diagnostics import (
    InvalidOperationError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
)
from .expr_evaluator import ExprEvaluator
from .scope_manager import ScopeManager
from .span_tracker import SpanTracker
from .type_resolver import TypeResolver
from ..typing.tiling import ArrayFieldInfo, ScalarFieldInfo, get_tiling_fields, is_tiling_class

if TYPE_CHECKING:
    from .decorator import InlineFunction


def _is_const_int(value: object) -> bool:
    """Check if a value is a compile-time constant integer.

    Handles plain int, ir.ConstInt, and ir.Neg(ir.ConstInt) (negative literals).
    """
    if isinstance(value, (int, ir.ConstInt)):
        return True
    return isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt)


def _const_int_value(value: object) -> int | None:
    """Extract integer value from a compile-time constant, or None."""
    if isinstance(value, int):
        return value
    if isinstance(value, ir.ConstInt):
        return value.value
    if isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt):
        return -value.operand.value
    return None


def _arch_needs_same_pipe_sync(npu_arch: str | None) -> bool:
    """Return True if the architecture requires same-pipeline sync insertion.

    dav-2201 (a2 / a3): The V pipeline does not guarantee intra-pipe
    completion ordering in hardware; software must insert sync_src/sync_dst
    even between two consecutive V operations that share a tile.

    dav-3510 (a5): Hardware provides the ordering guarantee automatically.
    """
    if npu_arch is None:
        return False
    arch = npu_arch.lower()
    return "dav-2201" in arch or arch in ("a2", "a3")


def _loop_body_has_bar_all(body: list[ast.stmt]) -> bool:
    """Return True if the loop body contains a ``pl.system.bar_all()`` call.

    When a barrier is present at each iteration boundary, all pipelines are
    flushed — there are no cross-iteration tile dependencies and backward
    sync insertion can be skipped.
    """
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if (
                isinstance(func, ast.Attribute) and func.attr == "bar_all"
                and isinstance(func.value, ast.Attribute) and func.value.attr == "system"
            ):
                return True
    return False


def _loop_body_backward_sync_redundant(body: list[ast.stmt]) -> bool:
    """Return True if the outer loop's backward sync is redundant.

    This happens when:
    - The loop body contains a ``bar_all()`` (explicit barrier), OR
    - Every tile operation in the loop body is inside a nested ``for`` loop
      (the nested loop will have its own backward sync that covers cross-
      iteration deps; the outer loop doesn't need additional sync).
    """
    if _loop_body_has_bar_all(body):
        return True
    # Check if all tile ops are inside nested for-loops.
    # If the body has only for-loops, with/section blocks, and non-tile
    # assignments, the outer loop doesn't directly use tiles across iterations.
    for stmt in body:
        if isinstance(stmt, ast.For):
            continue  # nested loop handles its own sync
        if isinstance(stmt, ast.With):
            # Check inside with-body (e.g., section_cube)
            if not _loop_body_backward_sync_redundant(stmt.body):
                return False
            continue
        if isinstance(stmt, ast.Assign):
            # Tile ops in assignments: check if it's a plm.xxx() call
            if isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                if (isinstance(func, ast.Attribute)
                        and isinstance(func.value, ast.Name)
                        and func.value.id == "plm"):
                    return False  # tile op directly in outer loop body
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "plm"):
                return False  # tile op directly in outer loop body
            continue
    return True


class _StructVar:
    """Compile-time grouping of IR expressions, accessed via attribute syntax.

    Created by ``pl.struct(field1=val1, field2=val2, ...)``.
    """

    def __init__(self, fields: dict[str, Any], name: str = "") -> None:
        self.fields = fields
        self.name = name


class _StructArrayVar:
    """List/tuple of homogeneous structs supporting dynamic index access.

    Created by assigning a list or tuple of ``_StructVar`` objects that share
    the same field names::

        ctx_arr = [ctx_0, ctx_1, ctx_2]   # all created via pl.struct(...)

    Dynamic indexing ``ctx_arr[idx]`` produces a ``_DynamicStructView`` which
    can be used for field reads (``view.field``), field writes
    (``view.field = val``), or as a function argument.
    """

    def __init__(self, structs: list[_StructVar], name: str = "") -> None:
        self.structs = structs
        self.field_names = list(structs[0].fields.keys())
        self.name = name  # C++ array variable name


class _DynamicStructView:
    """Runtime view into a ``_StructArrayVar`` at a dynamic IR index.

    Field reads lower to ``struct.get`` IR calls; field writes to ``struct.set``.
    The CCE codegen translates these to direct C++ array access: ``arr[idx].field``.
    """

    def __init__(self, array: _StructArrayVar, index_expr: "ir.Expr", ref_name: str = "") -> None:
        self.array = array
        self.index_expr = index_expr
        self.ref_name = ref_name  # C++ reference variable name (empty = no ref)


class ASTParser:
    """Parses Python AST and builds IR using IRBuilder."""

    def __init__(
        self,
        source_file: str,
        source_lines: list[str],
        line_offset: int = 0,
        col_offset: int = 0,
        global_vars: dict[str, ir.GlobalVar] | None = None,
        gvar_to_func: dict[ir.GlobalVar, ir.Function] | None = None,
        strict_ssa: bool = False,
        closure_vars: dict[str, Any] | None = None,
        auto_sync: bool = False,
        npu_arch: str | None = None,
    ):
        """Initialize AST parser.

        Args:
            source_file: Path to source file
            source_lines: Lines of source code (dedented for parsing)
            line_offset: Line number offset to add to AST line numbers (for dedented code)
            col_offset: Column offset to add to AST column numbers (for dedented code)
            global_vars: Optional map of function names to GlobalVars for cross-function calls
            gvar_to_func: Optional map of GlobalVars to parsed Functions for type inference
            strict_ssa: If True, enforce SSA (single assignment). If False (default), allow reassignment.
            closure_vars: Optional variables from the enclosing scope for dynamic shape resolution
            auto_sync: If True, automatically insert sync_src/sync_dst for cross-pipeline deps.
            npu_arch: Target architecture string (e.g. ``"dav-2201"``, ``"a3"``, ``"dav-3510"``).
                Used to determine whether same-pipeline syncs are needed.
        """
        self.span_tracker = SpanTracker(source_file, source_lines, line_offset, col_offset)
        self.scope_manager = ScopeManager(strict_ssa=strict_ssa)
        self.expr_evaluator = ExprEvaluator(
            closure_vars=closure_vars or {},
            span_tracker=self.span_tracker,
        )
        self.type_resolver = TypeResolver(
            expr_evaluator=self.expr_evaluator,
            scope_lookup=self.scope_manager.lookup_var,
            span_tracker=self.span_tracker,
        )
        self.builder = IRBuilder()
        self.global_vars = global_vars or {}  # Track GlobalVars for cross-function calls
        self.gvar_to_func = gvar_to_func or {}  # Track parsed functions for type inference
        self.external_funcs: dict[str, ir.Function] = {}  # Track external functions referenced

        # Track context for handling yields and returns
        self.in_for_loop = False
        self.in_while_loop = False
        self.in_if_stmt = False
        self.current_if_builder = None
        self.current_loop_builder = None

        # Inline function expansion state
        self._inline_mode = False
        self._inline_return_expr: ir.Expr | None = None
        self._inline_prefix: str = ""  # unique prefix per inline expansion
        self._inline_counter: int = 0  # monotonic counter for unique prefixes

        # Cache for implicitly compiled functions (keyed by id(fn))
        self._implicit_func_cache: dict[int, Any] = {}

        # Registry mapping tiling param names to their flattened field vars.
        # Scalar fields map to a single ir.Var; array fields map to list[ir.Var].
        self.tiling_registry: dict[str, dict[str, ir.Var | list[ir.Var]]] = {}

        # Counter for generating unique names in variable tuple index lowering
        self._tuple_idx_counter: int = 0

        # Registry mapping variable names to their constant-integer-tuple values.
        # Populated when a simple assignment like `event_ids = (0, 1)` is parsed.
        # Used by the sync-op statement expander to generate per-branch IfStmt chains.
        self._const_tuple_registry: dict[str, list[int]] = {}

        # Registry mapping variable names to their tile-tuple contents.
        # Populated when a simple assignment like `tile_buf = (ping, pong)` is parsed
        # and all elements resolve to TileType variables in the current scope.
        # Used by auto-sync to resolve tile_buf[buf_idx] subscript accesses.
        self._tile_tuple_registry: dict[str, list[str]] = {}

        # Cache: (tuple_var_name, index_ssa_var_name) → phi ir.Var from _build_tuple_index_chain.
        # Applies to all tuple types (tile, tensor, event ID, etc.).
        # Prevents re-emitting an if-else chain when the same buf[idx] expression
        # appears multiple times in the same linear code region.
        self._tuple_select_cache: dict[tuple[str, str], ir.Var] = {}

        # Auto-sync: track per-tile pipeline state for automatic sync insertion
        if auto_sync:
            from pypto.frontend.sync_tracker import SyncTracker
            same_pipe_sync = _arch_needs_same_pipe_sync(npu_arch)
            self.sync_tracker: SyncTracker | None = SyncTracker(same_pipe_sync=same_pipe_sync)
        else:
            self.sync_tracker = None

    def parse_function(
        self,
        func_def: ast.FunctionDef,
        func_type: ir.FunctionType = ir.FunctionType.Opaque,
    ) -> ir.Function:
        """Parse function definition and build IR.

        Args:
            func_def: AST FunctionDef node
            func_type: Function type (default: Opaque)

        Returns:
            IR Function object
        """
        func_name = func_def.name
        func_span = self.span_tracker.get_span(func_def)

        # Enter function scope
        self.scope_manager.enter_scope("function")

        # Reset tiling registry for this function scope
        self.tiling_registry = {}

        # Collect args to process, filtering out bare 'self'
        args_to_process = [
            arg for arg in func_def.args.args
            if not (arg.arg == "self" and arg.annotation is None)
        ]

        # Pre-validate tiling constraints: at most 1 tiling param, must be last
        tiling_param_names = [
            arg.arg for arg in args_to_process
            if arg.annotation is not None and self._resolve_tiling_class(arg.annotation) is not None
        ]
        if len(tiling_param_names) > 1:
            raise ParserSyntaxError(
                f"Function '{func_def.name}' has {len(tiling_param_names)} tiling parameters "
                f"({', '.join(tiling_param_names)}), but at most 1 is allowed",
                span=self.span_tracker.get_span(func_def),
                hint="A kernel may have at most one tiling parameter",
            )
        if len(tiling_param_names) == 1:
            if not args_to_process or args_to_process[-1].arg != tiling_param_names[0]:
                tiling_arg = next(a for a in args_to_process if a.arg == tiling_param_names[0])
                raise ParserSyntaxError(
                    f"Tiling parameter '{tiling_param_names[0]}' must be the last parameter",
                    span=self.span_tracker.get_span(tiling_arg),
                    hint="Move the tiling parameter to the last position",
                )

        # Begin building function
        with self.builder.function(func_name, func_span, type=func_type) as f:
            # Parse parameters
            for arg in args_to_process:
                param_name = arg.arg

                if arg.annotation is None:
                    raise ParserTypeError(
                        f"Parameter '{param_name}' missing type annotation",
                        span=self.span_tracker.get_span(arg),
                        hint="Add a type annotation like: x: pl.Tensor[[64], pl.FP32]",
                    )

                tiling_cls = self._resolve_tiling_class(arg.annotation)
                if tiling_cls is not None:
                    param_span = self.span_tracker.get_span(arg)
                    field_vars: dict[str, ir.Var | list[ir.Var]] = {}
                    for field_name, field_info in get_tiling_fields(tiling_cls).items():
                        if isinstance(field_info, ScalarFieldInfo):
                            flat_name = f"{param_name}_{field_name}"
                            flat_var = f.param(flat_name, ir.ScalarType(field_info.dtype), param_span)
                            field_vars[field_name] = flat_var
                        else:  # ArrayFieldInfo
                            vars_list: list[ir.Var] = []
                            for i in range(field_info.size):
                                flat_name = f"{param_name}_{field_name}_{i}"
                                flat_var = f.param(flat_name, ir.ScalarType(field_info.dtype), param_span)
                                vars_list.append(flat_var)
                            field_vars[field_name] = vars_list
                    self.tiling_registry[param_name] = field_vars
                    continue  # do NOT register tiling name itself in scope
                param_type, param_direction = self.type_resolver.resolve_param_type(arg.annotation)
                param_span = self.span_tracker.get_span(arg)

                # Add parameter to function with direction
                param_var = f.param(param_name, param_type, param_span, direction=param_direction)

                # Register in scope
                self.scope_manager.define_var(param_name, param_var, allow_redef=True)

            # Parse return type
            if func_def.returns:
                return_type = self.type_resolver.resolve_type(func_def.returns)
                if isinstance(return_type, list):
                    # tuple[T1, T2, ...] -> multiple return types
                    for rt in return_type:
                        f.return_type(rt)
                else:
                    f.return_type(return_type)

            # Parse function body (skip docstrings)
            for i, stmt in enumerate(func_def.body):
                # Skip docstrings (string constants as first statement or after decorators)
                if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, str):
                        continue  # Skip docstring
                self.parse_statement(stmt)

        # Exit function scope
        self.scope_manager.exit_scope()

        return f.get_result()

    def _resolve_tiling_class(self, annotation: ast.expr) -> type | None:
        """Return the tiling class if annotation refers to one in closure_vars, else None.

        Args:
            annotation: AST expression node for the annotation

        Returns:
            The resolved tiling class, or None if the annotation is not a tiling class
        """
        if not isinstance(annotation, ast.Name):
            return None
        cls = self.expr_evaluator.closure_vars.get(annotation.id)
        return cls if is_tiling_class(cls) else None

    def parse_statement(self, stmt: ast.stmt) -> None:
        """Parse a statement node.

        Args:
            stmt: AST statement node
        """
        if isinstance(stmt, ast.AnnAssign):
            self.parse_annotated_assignment(stmt)
        elif isinstance(stmt, ast.Assign):
            self.parse_assignment(stmt)
        elif isinstance(stmt, ast.For):
            self.parse_for_loop(stmt)
        elif isinstance(stmt, ast.While):
            self.parse_while_loop(stmt)
        elif isinstance(stmt, ast.If):
            self.parse_if_statement(stmt)
        elif isinstance(stmt, ast.With):
            self.parse_with_statement(stmt)
        elif isinstance(stmt, ast.Return):
            self.parse_return(stmt)
        elif isinstance(stmt, ast.Break):
            self.parse_break(stmt)
        elif isinstance(stmt, ast.Continue):
            self.parse_continue(stmt)
        elif isinstance(stmt, ast.Expr):
            self.parse_evaluation_statement(stmt)
        elif isinstance(stmt, ast.Pass):
            pass  # No-op: pass statements are valid in DSL functions
        else:
            raise UnsupportedFeatureError(
                f"Unsupported statement type: {type(stmt).__name__}",
                span=self.span_tracker.get_span(stmt),
                hint="Only assignments, for loops, while loops, if statements, "
                "with statements, returns, break, and continue are supported in DSL functions",
            )

    def parse_annotated_assignment(self, stmt: ast.AnnAssign) -> None:
        """Parse annotated assignment: var: type = value.

        Args:
            stmt: AnnAssign AST node
        """
        if not isinstance(stmt.target, ast.Name):
            raise ParserSyntaxError(
                "Only simple variable assignments supported",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use a simple variable name for assignment targets",
            )

        var_name = stmt.target.id
        span = self.span_tracker.get_span(stmt)

        # Check if this is a yield assignment: var: type = pl.yield_(...)
        if isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == "yield_":
                # Handle yield assignment
                yield_exprs = []
                for arg in stmt.value.args:
                    expr = self.parse_expression(arg)
                    yield_exprs.append(expr)

                # Emit yield statement
                yield_span = self.span_tracker.get_span(stmt.value)
                self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))

                # Track variable name for if statement output registration
                if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
                    self._current_yield_vars.append(var_name)

                # Capture yield expression type for unannotated yield inference
                # Use setdefault so the then-branch type takes precedence over else
                if hasattr(self, "_current_yield_types") and self._current_yield_types is not None:
                    if len(yield_exprs) == 1:
                        self._current_yield_types.setdefault(var_name, yield_exprs[0].type)

                # Don't register in scope yet - will be done when if statement completes
                return

        # Parse value expression
        if stmt.value is None:
            raise UnsupportedFeatureError(
                "Yield assignment with no value is not supported",
                span=self.span_tracker.get_span(stmt),
                hint="Provide a value for the assignment",
            )
        value_expr = self.parse_expression(stmt.value)

        # Use annotation type as override when it carries memref info
        annotation_type = self.type_resolver.resolve_type_if_memref(stmt.annotation)
        var = self.builder.let(var_name, value_expr, type=annotation_type, span=span)

        # Register in scope
        self.scope_manager.define_var(var_name, var, span=span)

    def parse_assignment(self, stmt: ast.Assign) -> None:
        """Parse regular assignment: var = value or tuple unpacking.

        Args:
            stmt: Assign AST node
        """
        # Handle tuple unpacking for yields
        if len(stmt.targets) == 1:
            target = stmt.targets[0]

            # Handle tuple unpacking: (a, b, c) = pl.yield_(...) or self.func(...)
            if isinstance(target, ast.Tuple):
                # Check if value is a pl.yield_() call
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        # This is handled in yield parsing
                        self.parse_yield_assignment(target, stmt.value)
                        return

                # General tuple unpacking for function calls returning TupleType
                span = self.span_tracker.get_span(stmt)
                value_expr = self.parse_expression(stmt.value)

                # Bind the tuple result to a temporary variable
                tuple_var = self.builder.let("_tuple_tmp", value_expr, span=span)

                # Extract each element using TupleGetItemExpr
                for i, elt in enumerate(target.elts):
                    if not isinstance(elt, ast.Name):
                        raise ParserSyntaxError(
                            f"Tuple unpacking target must be a variable name, got {ast.unparse(elt)}",
                            span=self.span_tracker.get_span(elt),
                            hint="Use simple variable names in tuple unpacking: a, b, c = func()",
                        )
                    item_expr = ir.TupleGetItemExpr(tuple_var, i, span)
                    var = self.builder.let(elt.id, item_expr, span=span)
                    self.scope_manager.define_var(elt.id, var, span=span)
                return

            # Handle simple assignment
            if isinstance(target, ast.Name):
                var_name = target.id
                span = self.span_tracker.get_span(stmt)

                # Check if this is a TileType assignment
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    # TileType(...)
                    if isinstance(func, ast.Name) and func.id == "TileType":
                        tile_type = self._parse_tile_type_call(stmt.value)
                        self.scope_manager.define_python_var(var_name, tile_type, span=span)
                        return
                    # plm.TileType(...)
                    if isinstance(func, ast.Attribute) and func.attr == "TileType":
                        tile_type = self._parse_tile_type_call(stmt.value)
                        self.scope_manager.define_python_var(var_name, tile_type, span=span)
                        return
                    # pl.struct(field1=val1, field2=val2, ...)
                    if isinstance(func, ast.Attribute) and func.attr == "struct":
                        struct_var = self._parse_struct_call(stmt.value, var_name)
                        self.scope_manager.define_python_var(var_name, struct_var, span=span)
                        # Emit struct.declare for C++ struct codegen
                        fields_csv = ",".join(struct_var.fields.keys())
                        decl_call = ir.create_op_call(
                            "struct.declare", [],
                            {"array": var_name, "size": 1, "fields": fields_csv},
                            span,
                        )
                        self.builder.emit(ir.EvalStmt(decl_call, span))
                        # Emit struct.set for non-zero init values
                        idx_zero = ir.ConstInt(0, DataType.INDEX, span)
                        for fname, fval in struct_var.fields.items():
                            if isinstance(fval, ir.Expr):
                                # Skip trivial zero inits
                                if isinstance(fval, ir.ConstInt) and fval.value == 0:
                                    continue
                                init_call = ir.create_op_call(
                                    "struct.set", [idx_zero, fval],
                                    {"array": var_name, "field": fname}, span,
                                )
                                self.builder.emit(ir.EvalStmt(init_call, span))
                        return
                    # pl.StructArray(N, field1=val1, field2=val2, ...)
                    if isinstance(func, ast.Attribute) and func.attr == "StructArray":
                        sa_call = stmt.value
                        if not sa_call.args or not isinstance(sa_call.args[0], ast.Constant):
                            raise ParserSyntaxError(
                                "pl.StructArray() requires an integer size as first argument",
                                span=span,
                                hint="Use pl.StructArray(3, field1=0, field2=0, ...)",
                            )
                        arr_size = sa_call.args[0].value
                        if not isinstance(arr_size, int) or arr_size < 1:
                            raise ParserSyntaxError(
                                f"pl.StructArray() size must be a positive integer, got {arr_size}",
                                span=span,
                            )
                        # Parse field kwargs
                        fields: dict[str, Any] = {}
                        for kw in sa_call.keywords:
                            if kw.arg is None:
                                raise ParserSyntaxError("pl.StructArray() does not support **kwargs", span=span)
                            fields[kw.arg] = self.parse_expression(kw.value)
                        if not fields:
                            raise ParserSyntaxError("pl.StructArray() requires at least one field", span=span)

                        # Create _StructArrayVar with dummy _StructVar slots (for field validation)
                        structs = [_StructVar(dict(fields), name=f"{var_name}_{i}") for i in range(arr_size)]
                        struct_arr = _StructArrayVar(structs, name=var_name)
                        self.scope_manager.define_python_var(var_name, struct_arr, span=span)

                        # Emit struct.declare
                        fields_csv = ",".join(fields.keys())
                        decl_call = ir.create_op_call(
                            "struct.declare", [],
                            {"array": var_name, "size": arr_size, "fields": fields_csv},
                            span,
                        )
                        self.builder.emit(ir.EvalStmt(decl_call, span))

                        # Emit struct.set for non-zero init values (applied to all elements)
                        for slot in range(arr_size):
                            idx_expr = ir.ConstInt(slot, DataType.INDEX, span)
                            for fname, fval in fields.items():
                                if isinstance(fval, ir.Expr):
                                    if isinstance(fval, ir.ConstInt) and fval.value == 0:
                                        continue
                                    init_call = ir.create_op_call(
                                        "struct.set", [idx_expr, fval],
                                        {"array": var_name, "field": fname}, span,
                                    )
                                    self.builder.emit(ir.EvalStmt(init_call, span))
                        return
                # Check if RHS is a list/tuple containing struct variables → error (use StructArray)
                if isinstance(stmt.value, (ast.List, ast.Tuple)) and stmt.value.elts:
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Name):
                            obj = self.scope_manager.get_python_var(elt.id)
                            if isinstance(obj, _StructVar):
                                raise ParserSyntaxError(
                                    f"Putting pl.struct into a list/tuple is not supported. "
                                    f"Use pl.StructArray(N, field=val, ...) instead.",
                                    span=self.span_tracker.get_span(elt),
                                    hint=f"Example: {var_name} = pl.StructArray(N, field1=0, field2=0)",
                                )
                # Check if RHS is a struct array subscript: ctx_curr = ctx_arr[task_id]
                # Store as a _DynamicStructView alias and emit struct.ref for C++ reference
                if isinstance(stmt.value, ast.Subscript) and isinstance(stmt.value.value, ast.Name):
                    arr_obj = self.scope_manager.get_python_var(stmt.value.value.id)
                    if arr_obj is None:
                        arr_obj = self.scope_manager.lookup_var(stmt.value.value.id)
                    if isinstance(arr_obj, _StructArrayVar):
                        index_expr = self.parse_expression(stmt.value.slice)
                        view = _DynamicStructView(arr_obj, index_expr, ref_name=var_name)
                        self.scope_manager.define_python_var(var_name, view, span=span)
                        # Emit struct.ref for C++ codegen: auto& var = arr[idx];
                        ref_call = ir.create_op_call(
                            "struct.ref", [index_expr],
                            {"array": arr_obj.name, "var": var_name}, span,
                        )
                        self.builder.emit(ir.EvalStmt(ref_call, span))
                        return
                # Check if this is a yield assignment: var = pl.yield_(...)
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        # Handle yield assignment
                        yield_exprs = []
                        for arg in stmt.value.args:
                            expr = self.parse_expression(arg)
                            if not isinstance(expr, ir.Expr):
                                raise ParserSyntaxError(
                                    f"Yield argument must be an IR expression, got {type(expr)}",
                                    span=self.span_tracker.get_span(arg),
                                    hint="Ensure yield arguments are valid expressions",
                                )
                            yield_exprs.append(expr)

                        # Emit yield statement
                        yield_span = self.span_tracker.get_span(stmt.value)
                        self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))

                        # Track variable name for loop/if output registration
                        if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
                            self._current_yield_vars.append(var_name)

                        # Capture yield expression type for unannotated yield inference
                        # Use setdefault so the then-branch type takes precedence over else
                        if hasattr(self, "_current_yield_types") and self._current_yield_types is not None:
                            if len(yield_exprs) == 1:
                                self._current_yield_types.setdefault(var_name, yield_exprs[0].type)

                        # Don't register in scope yet - will be done when loop/if completes
                        return

                value_expr = self.parse_expression(stmt.value)
                if value_expr is None:
                    raise ParserTypeError(
                        f"Cannot assign void inline function result to '{var_name}'",
                        span=span,
                        hint="Inline functions used as expressions must return a value",
                    )
                ir_var_name = self._inline_prefix + var_name if self._inline_prefix else var_name
                var = self.builder.let(ir_var_name, value_expr, span=span)
                self.scope_manager.define_var(var_name, var, span=span)

                # Auto-sync: register tile region for overlap detection
                if self.sync_tracker is not None and isinstance(stmt.value, ast.Call):
                    call_op_name = self._extract_plm_or_block_op_name(stmt.value)
                    if call_op_name == "make_tile":
                        self._register_tile_region(var_name, var)

                # Register constant-integer tuples for sync-op event_id expansion
                if isinstance(stmt.value, ast.Tuple) and all(
                    isinstance(elt, ast.Constant) and isinstance(elt.value, int)
                    for elt in stmt.value.elts
                ):
                    self._const_tuple_registry[var_name] = [elt.value for elt in stmt.value.elts]  # type: ignore[union-attr]

                # Register tile-variable tuples for DB auto-sync subscript resolution
                if isinstance(stmt.value, ast.Tuple) and len(stmt.value.elts) >= 2:
                    tile_names: list[str] = []
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Name):
                            elt_var = self.scope_manager.lookup_var(elt.id)
                            if elt_var is not None and isinstance(getattr(elt_var, "type", None), ir.TileType):
                                tile_names.append(elt.id)
                    if len(tile_names) == len(stmt.value.elts):
                        self._tile_tuple_registry[var_name] = tile_names

                return

            # Handle struct field assignment: ctx.field = value
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                obj_name = target.value.id
                field_name = target.attr
                span = self.span_tracker.get_span(stmt)

                obj = self.scope_manager.get_python_var(obj_name)
                if obj is None:
                    obj = self.scope_manager.lookup_var(obj_name)
                if isinstance(obj, _StructVar):
                    if field_name not in obj.fields:
                        raise ParserTypeError(
                            f"Struct '{obj_name}' has no field '{field_name}'",
                            span=span,
                            hint=f"Available fields: {', '.join(obj.fields.keys())}",
                        )
                    value_expr = self.parse_expression(stmt.value)
                    if obj.name and isinstance(value_expr, ir.Expr):
                        # Named struct with C++ codegen — use struct.set
                        idx_zero = ir.ConstInt(0, DataType.INDEX, span)
                        call = ir.create_op_call(
                            "struct.set", [idx_zero, value_expr],
                            {"array": obj.name, "field": field_name}, span,
                        )
                        self.builder.emit(ir.EvalStmt(call, span))
                    elif isinstance(value_expr, ir.Expr):
                        # Unnamed struct — fallback to IR variable
                        ir_name = f"_{obj.name}_{field_name}" if obj.name else field_name
                        var = self.builder.let(ir_name, value_expr, span=span)
                        obj.fields[field_name] = var
                    else:
                        obj.fields[field_name] = value_expr
                    return
                # _DynamicStructView field write (view passed as function arg)
                if isinstance(obj, _DynamicStructView):
                    if field_name not in obj.array.field_names:
                        raise ParserTypeError(
                            f"Struct array view has no field '{field_name}'",
                            span=span,
                            hint=f"Available fields: {', '.join(obj.array.field_names)}",
                        )
                    value_expr = self.parse_expression(stmt.value)
                    self._struct_array_field_write(obj.array, obj.index_expr, field_name, value_expr, span, ref_name=obj.ref_name)
                    return

            # Handle struct array field assignment: ctx_arr[idx].field = value
            if (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Subscript)
                    and isinstance(target.value.value, ast.Name)):
                arr_name = target.value.value.id
                field_name = target.attr
                span = self.span_tracker.get_span(stmt)

                arr_obj = self.scope_manager.get_python_var(arr_name)
                if arr_obj is None:
                    arr_obj = self.scope_manager.lookup_var(arr_name)
                if isinstance(arr_obj, _StructArrayVar):
                    if field_name not in arr_obj.field_names:
                        raise ParserTypeError(
                            f"Struct array '{arr_name}' has no field '{field_name}'",
                            span=span,
                            hint=f"Available fields: {', '.join(arr_obj.field_names)}",
                        )
                    index_expr = self.parse_expression(target.value.slice)
                    value_expr = self.parse_expression(stmt.value)
                    self._struct_array_field_write(arr_obj, index_expr, field_name, value_expr, span)
                    return

        raise ParserSyntaxError(
            f"Unsupported assignment: {ast.unparse(stmt)}",
            span=self.span_tracker.get_span(stmt),
            hint="Use simple variable assignments or tuple unpacking with pl.yield_()",
        )

    def parse_yield_assignment(self, target: ast.Tuple, value: ast.Call) -> None:
        """Parse yield assignment: (a, b) = pl.yield_(x, y).

        Args:
            target: Tuple of target variable names
            value: Call to pl.yield_()
        """
        # Parse yield expressions
        yield_exprs = []
        for arg in value.args:
            expr = self.parse_expression(arg)
            # Ensure it's an IR Expr
            if not isinstance(expr, ir.Expr):
                raise ParserSyntaxError(
                    f"Yield argument must be an IR expression, got {type(expr)}",
                    span=self.span_tracker.get_span(arg),
                    hint="Ensure yield arguments are valid expressions",
                )
            yield_exprs.append(expr)

        # Emit yield statement
        span = self.span_tracker.get_span(value)
        self.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Track yielded variable names for if/for statement processing
        if hasattr(self, "_current_yield_vars") and self._current_yield_vars is not None:
            for elt in target.elts:
                if isinstance(elt, ast.Name):
                    self._current_yield_vars.append(elt.id)

        # Capture yield expression types for unannotated yield inference
        # Use setdefault so the then-branch type takes precedence over else
        if hasattr(self, "_current_yield_types") and self._current_yield_types is not None:
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name) and i < len(yield_exprs):
                    self._current_yield_types.setdefault(elt.id, yield_exprs[i].type)

        # For tuple yields at the for/while loop level, register the variables
        # (they'll be available as loop.get_result().return_vars)
        if (self.in_for_loop or self.in_while_loop) and not self.in_if_stmt:
            # Register yielded variable names in scope
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    # Will be resolved from loop outputs
                    self.scope_manager.define_var(var_name, f"loop_yield_{i}")

    _VALID_ITERATORS = {"range", "parallel", "unroll", "while_"}
    _ITERATOR_ERROR = "For loop must use pl.range(), pl.parallel(), pl.unroll(), or pl.while_()"
    _ITERATOR_HINT = "Use pl.range(), pl.parallel(), pl.unroll(), or pl.while_() as the iterator"

    def _validate_for_loop_iterator(self, stmt: ast.For) -> tuple[ast.Call, str]:
        """Validate that for loop uses pl.range(), pl.parallel(), pl.unroll(), or pl.while_().

        Returns:
            Tuple of (call_node, iterator_type) where iterator_type is
            "range", "parallel", "unroll", or "while_"
        """
        if not isinstance(stmt.iter, ast.Call):
            raise ParserSyntaxError(
                self._ITERATOR_ERROR,
                span=self.span_tracker.get_span(stmt.iter),
                hint=self._ITERATOR_HINT,
            )

        iter_call = stmt.iter
        func = iter_call.func
        if isinstance(func, ast.Attribute) and func.attr in self._VALID_ITERATORS:
            return iter_call, func.attr

        raise ParserSyntaxError(
            self._ITERATOR_ERROR,
            span=self.span_tracker.get_span(stmt.iter),
            hint=self._ITERATOR_HINT,
        )

    def _parse_for_loop_target(self, stmt: ast.For) -> tuple[str, ast.AST | None, bool]:
        """Parse for loop target, returning (loop_var_name, iter_args_node, is_simple_for)."""
        if isinstance(stmt.target, ast.Name):
            return stmt.target.id, None, True

        if isinstance(stmt.target, ast.Tuple) and len(stmt.target.elts) == 2:
            loop_var_node = stmt.target.elts[0]
            iter_args_node = stmt.target.elts[1]

            if not isinstance(loop_var_node, ast.Name):
                raise ParserSyntaxError(
                    "Loop variable must be a simple name",
                    span=self.span_tracker.get_span(loop_var_node),
                    hint="Use a simple variable name for the loop counter",
                )
            return loop_var_node.id, iter_args_node, False

        raise ParserSyntaxError(
            "For loop target must be a simple name or: (loop_var, (iter_args...))",
            span=self.span_tracker.get_span(stmt.target),
            hint="Use: for i in pl.range(n) or for i, (var1,) in pl.range(n, init_values=(...,))",
        )

    def _setup_iter_args(self, loop: Any, iter_args_node: ast.AST, init_values: list) -> None:
        """Set up iter_args and return_vars for Pattern A loops."""
        if not isinstance(iter_args_node, ast.Tuple):
            raise ParserSyntaxError(
                "Iter args must be a tuple",
                span=self.span_tracker.get_span(iter_args_node),
                hint="Wrap iteration variables in parentheses: (var1, var2)",
            )

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

        for iter_arg_node in iter_args_node.elts:
            assert isinstance(iter_arg_node, ast.Name)
            loop.return_var(f"{iter_arg_node.id}_out")

    def parse_for_loop(self, stmt: ast.For) -> None:
        """Parse for loop with pl.range(), pl.parallel(), pl.unroll(), or pl.while_().

        Supports patterns for range/parallel/unroll:
          Pattern A (explicit): for i, (vars,) in pl.range(..., init_values=(...,))
          Pattern B (simple):   for i in pl.range(n)

        Supports pattern for while-as-for:
          for (vars,) in pl.while_(init_values=(...,)):
              pl.cond(condition)

        Both patterns also work with pl.parallel() for parallel loops.
        pl.unroll() is for compile-time loop unrolling (no init_values).
        Pattern B produces a ForStmt without iter_args/return_vars/yield.
        The C++ ConvertToSSA pass handles converting to SSA form.
        """
        iter_call, iterator_type = self._validate_for_loop_iterator(stmt)

        # Handle pl.while_() case
        if iterator_type == "while_":
            self._parse_while_as_for(stmt, iter_call)
            return

        # Handle pl.range(), pl.parallel(), or pl.unroll()
        _ITERATOR_TO_KIND = {
            "range": ir.ForKind.Sequential,
            "parallel": ir.ForKind.Parallel,
            "unroll": ir.ForKind.Unroll,
        }
        loop_var_name, iter_args_node, is_simple_for = self._parse_for_loop_target(stmt)
        range_args = self._parse_range_call(iter_call)

        if is_simple_for and range_args["init_values"]:
            raise ParserSyntaxError(
                "For loop target must be a tuple when init_values is provided",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for i, (var1,) in pl.range(n, init_values=(val1,)) to include iter_args",
            )

        if iterator_type == "unroll" and range_args["init_values"]:
            raise ParserSyntaxError(
                "pl.unroll() cannot be combined with init_values",
                span=self.span_tracker.get_span(iter_call),
                hint="Unrolled loops do not support loop-carried values (init_values)",
            )

        # For pl.unroll(), require compile-time constant integer bounds
        # and reject step=0. Fail early with clear parser errors instead of
        # later generic failures in the UnrollLoops C++ pass.
        # Note: negative literals like -1 become ir.Neg(ir.ConstInt(1)).
        if iterator_type == "unroll":
            for _bound_name in ("start", "stop", "step"):
                _bound_value = range_args.get(_bound_name)
                if _bound_value is not None and not _is_const_int(_bound_value):
                    raise ParserSyntaxError(
                        "pl.unroll() requires compile-time constant integer bounds",
                        span=self.span_tracker.get_span(iter_call),
                        hint="Use integer literals for start, stop, and step in pl.unroll().",
                    )
            _step = range_args.get("step")
            if _const_int_value(_step) == 0:
                raise ParserSyntaxError(
                    "pl.unroll() step cannot be zero",
                    span=self.span_tracker.get_span(iter_call),
                    hint="Use a non-zero step in pl.unroll(start, stop, step).",
                )

        # Validate chunk arguments
        chunk_expr = range_args.get("chunk")
        chunk_policy_str = range_args.get("chunk_policy", "leading_full")
        if chunk_expr is not None:
            self._validate_chunk_args(chunk_expr, range_args["init_values"], iter_call)

        kind = _ITERATOR_TO_KIND[iterator_type]
        loop_var = self.builder.var(loop_var_name, ir.ScalarType(DataType.INDEX))
        span = self.span_tracker.get_span(stmt)
        loop_output_vars: list[str] = []

        # Auto-sync: pre-scan for backward (cross-iteration) dependencies
        backward_deps: list = []
        if self.sync_tracker is not None:
            from pypto.frontend.sync_tracker import (
                BackwardDep,
                emit_backward_sync_dst,
                emit_backward_sync_src,
                prescan_loop_backward_deps,
            )

            # Flush any outer loop's pending backward waits before this loop
            # starts.  Must happen BEFORE the for_loop builder context so the
            # waits are emitted at the outer loop level, not inside this loop.
            self._flush_all_pending_backward_waits(span)

            backward_deps = prescan_loop_backward_deps(
                stmt.body,
                self.scope_manager.lookup_var,
                self.sync_tracker._event_allocator,
                loop_depth=self.sync_tracker.get_loop_depth(),
                tile_tuple_registry=self._tile_tuple_registry,
            )
            # If the loop body's tile ops are all inside nested loops (which have
            # their own backward sync), or if bar_all is present, skip redundant
            # outer-loop backward sync.
            if _loop_body_backward_sync_redundant(stmt.body):
                backward_deps = []
            # 1. Priming: emit sync_src before the loop so the first
            #    iteration's wait_flag has a matching set_flag.
            #    For DB deps (n_slots>1), emit one set_flag per slot.
            for dep in backward_deps:
                if dep.n_slots > 1:
                    for slot in range(dep.n_slots):
                        eid = (dep.event_id + slot) % 8
                        slot_dep = BackwardDep(dep.first_pipe, dep.last_pipe, dep.tile_name, eid, dep.loop_depth)
                        emit_backward_sync_src(self.builder, slot_dep, span)
                else:
                    emit_backward_sync_src(self.builder, dep, span)
            # Save pre-loop buffer states
            self.sync_tracker.enter_loop()

        with self.builder.for_loop(
            loop_var,
            range_args["start"],
            range_args["stop"],
            range_args["step"],
            span,
            kind,
            chunk_size=chunk_expr,
            chunk_policy=chunk_policy_str,
        ) as loop:
            self.current_loop_builder = loop
            self.in_for_loop = True
            self.scope_manager.enter_scope("for")
            self.scope_manager.define_var(loop_var_name, loop_var, allow_redef=True)

            if not is_simple_for:
                assert iter_args_node is not None  # Guaranteed by _parse_for_loop_target
                self._setup_iter_args(loop, iter_args_node, range_args["init_values"])

            prev_yield_tracker = getattr(self, "_current_yield_vars", None)
            self._current_yield_vars = []
            prev_yield_types = getattr(self, "_current_yield_types", None)
            self._current_yield_types = {}

            # Auto-sync: register deferred backward waits.
            # Instead of emitting all backward waits at body start, defer
            # each wait until the first op on its first_pipe runs.  This
            # allows load (MTE2) to overlap with the previous matmul (M)
            # because the M→MTE1 wait is deferred to right before the move.
            if self.sync_tracker is not None:
                db_slot_expr = None
                if backward_deps and any(dep.n_slots > 1 for dep in backward_deps):
                    db_slot_expr = self._build_db_slot_expr(loop_var, range_args["step"],
                                                            backward_deps[0].n_slots, span)
                self._pending_backward_waits = {}
                self._pending_backward_wait_slot_expr = db_slot_expr
                for dep in backward_deps:
                    pipe = dep.first_pipe
                    self._pending_backward_waits.setdefault(pipe, []).append(dep)

            for body_stmt in stmt.body:
                self.parse_statement(body_stmt)

            # Auto-sync: backward set at loop body end
            if self.sync_tracker is not None:
                for dep in backward_deps:
                    if dep.n_slots > 1:
                        slot_expr = self._build_db_slot_expr(loop_var, range_args["step"], dep.n_slots, span)
                        self._emit_backward_db_sync_chain(dep, slot_expr, emit_backward_sync_src, span)
                    else:
                        emit_backward_sync_src(self.builder, dep, span)

            loop_output_vars = self._current_yield_vars[:]
            self._current_yield_vars = prev_yield_tracker
            self._current_yield_types = prev_yield_types

            should_leak = is_simple_for and not loop_output_vars
            self.scope_manager.exit_scope(leak_vars=should_leak)
            self.in_for_loop = False
            self.current_loop_builder = None

        # Auto-sync: restore pre-loop state and emit drain sync_dst
        if self.sync_tracker is not None:
            loop_ctx = self.sync_tracker.exit_loop()
            # Verify prescan results against actual loop body observations
            self._verify_backward_deps(backward_deps, loop_ctx)
            # Drain: emit wait_flag to consume the last iteration's body-end
            # set_flag, balancing the event flag counter.  Drain must NOT be
            # skipped even in nested loops — each (set_pipe, wait_pipe, event_id)
            # triple must have strictly paired set/wait counts.
            for dep in backward_deps:
                if dep.n_slots > 1:
                    for slot in range(dep.n_slots):
                        eid = (dep.event_id + slot) % 8
                        slot_dep = BackwardDep(dep.first_pipe, dep.last_pipe, dep.tile_name, eid, dep.loop_depth)
                        emit_backward_sync_dst(self.builder, slot_dep, span)
                else:
                    emit_backward_sync_dst(self.builder, dep, span)

        if not is_simple_for:
            loop_result = loop.get_result()
            if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
                for i, var_name in enumerate(loop_output_vars):
                    if i < len(loop_result.return_vars):
                        self.scope_manager.define_var(var_name, loop_result.return_vars[i])

    def _validate_chunk_args(self, chunk_expr: Any, init_values: list[Any], iter_call: ast.Call) -> None:
        """Validate chunk arguments for range/parallel/unroll loops."""
        if init_values:
            raise ParserSyntaxError(
                "chunk cannot be combined with init_values",
                span=self.span_tracker.get_span(iter_call),
                hint="Chunked loops do not support loop-carried values (init_values)",
            )
        if not _is_const_int(chunk_expr):
            raise ParserSyntaxError(
                "chunk must be a compile-time constant positive integer",
                span=self.span_tracker.get_span(iter_call),
                hint="Use an integer literal for chunk: chunk=5",
            )
        chunk_val = _const_int_value(chunk_expr)
        if chunk_val is not None and chunk_val <= 0:
            raise ParserSyntaxError(
                f"chunk must be a positive integer, got {chunk_val}",
                span=self.span_tracker.get_span(iter_call),
                hint="Use a positive integer for chunk: chunk=5",
            )

    def _parse_range_call(self, call: ast.Call) -> dict[str, Any]:
        """Parse pl.range() call arguments.

        Args:
            call: AST Call node for pl.range()

        Returns:
            Dictionary with start, stop, step, init_values
        """
        # Parse positional arguments
        if len(call.args) < 1:
            raise ParserSyntaxError(
                "pl.range() requires at least 1 argument (stop)",
                span=self.span_tracker.get_span(call),
                hint="Provide at least the stop value: pl.range(10) or pl.range(0, 10)",
            )

        # Default values
        start = 0
        step = 1

        if len(call.args) == 1:
            # range(stop)
            stop = self.parse_expression(call.args[0])
        elif len(call.args) == 2:
            # range(start, stop)
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
        elif len(call.args) >= 3:
            # range(start, stop, step)
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
            step = self.parse_expression(call.args[2])

        # Parse keyword arguments
        init_values = []
        chunk = None
        chunk_policy = "leading_full"
        for keyword in call.keywords:
            if keyword.arg == "init_values":
                # Parse list of init values
                if isinstance(keyword.value, (ast.List, ast.Tuple)):
                    for elt in keyword.value.elts:
                        init_values.append(self.parse_expression(elt))
                else:
                    raise ParserSyntaxError(
                        "init_values must be a list or tuple",
                        span=self.span_tracker.get_span(keyword.value),
                        hint="Use a tuple for init_values: init_values=(var1, var2)",
                    )
            elif keyword.arg == "chunk":
                chunk = self.parse_expression(keyword.value)
            elif keyword.arg == "chunk_policy":
                if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                    _VALID_CHUNK_POLICIES = {"leading_full"}
                    if keyword.value.value not in _VALID_CHUNK_POLICIES:
                        raise ParserSyntaxError(
                            f"Unsupported chunk_policy: {keyword.value.value!r}",
                            span=self.span_tracker.get_span(keyword.value),
                            hint=f"Supported values: {', '.join(sorted(_VALID_CHUNK_POLICIES))}",
                        )
                    chunk_policy = keyword.value.value
                else:
                    raise ParserSyntaxError(
                        "chunk_policy must be a string literal",
                        span=self.span_tracker.get_span(keyword.value),
                        hint='Use a string like chunk_policy="leading_full"',
                    )
            else:
                raise ParserSyntaxError(
                    f"Unknown keyword argument '{keyword.arg}' in range()",
                    span=self.span_tracker.get_span(keyword),
                    hint="Supported keywords: init_values, chunk, chunk_policy",
                )

        return {
            "start": start,
            "stop": stop,
            "step": step,
            "init_values": init_values,
            "chunk": chunk,
            "chunk_policy": chunk_policy,
        }

    def _is_cond_call(self, stmt: ast.stmt) -> bool:
        """Check if statement is a pl.cond() call (without parsing).

        Args:
            stmt: AST statement node

        Returns:
            True if statement is pl.cond() call, False otherwise
        """
        if not isinstance(stmt, ast.Expr):
            return False

        call = stmt.value
        if not isinstance(call, ast.Call):
            return False

        # Check if this is pl.cond() or cond()
        if isinstance(call.func, ast.Attribute):
            # pl.cond() form
            return call.func.attr == "cond"
        elif isinstance(call.func, ast.Name):
            # cond() form (if imported directly)
            return call.func.id == "cond"

        return False

    def _extract_cond_call(self, stmt: ast.stmt) -> ir.Expr | None:
        """Extract condition from pl.cond() call statement.

        Args:
            stmt: AST statement node

        Returns:
            Parsed condition expression if statement is pl.cond(), None otherwise
        """
        if not self._is_cond_call(stmt):
            return None

        call = stmt.value  # type: ignore[union-attr]

        # Parse the condition argument
        if len(call.args) != 1:  # type: ignore[attr-defined]
            raise ParserSyntaxError(
                "pl.cond() requires exactly 1 argument",
                span=self.span_tracker.get_span(call),
                hint="Use: pl.cond(condition)",
            )

        return self.parse_expression(call.args[0])  # type: ignore[attr-defined]

    def _validate_while_call_args(self, while_call: ast.Call) -> None:
        """Validate that pl.while_() has no positional arguments."""
        if len(while_call.args) > 0:
            raise ParserSyntaxError(
                "pl.while_() takes no positional arguments",
                span=self.span_tracker.get_span(while_call),
                hint="Use: pl.while_(init_values=(...,)) with pl.cond(condition) as first statement in body",
            )

    def _parse_while_init_values(self, while_call: ast.Call) -> list[ir.Expr]:
        """Parse init_values from pl.while_() keyword arguments."""
        init_values = []
        for keyword in while_call.keywords:
            if keyword.arg == "init_values":
                if isinstance(keyword.value, (ast.List, ast.Tuple)):
                    for elt in keyword.value.elts:
                        init_values.append(self.parse_expression(elt))
                else:
                    raise ParserSyntaxError(
                        "init_values must be a tuple or list",
                        span=self.span_tracker.get_span(keyword.value),
                        hint="Use a tuple for init_values (lists also accepted): init_values=(var1, var2)",
                    )

        if not init_values:
            raise ParserSyntaxError(
                "pl.while_() requires init_values",
                span=self.span_tracker.get_span(while_call),
                hint="Provide init_values: pl.while_(init_values=(val1, val2))",
            )

        return init_values

    def _validate_while_body(self, stmt: ast.For) -> None:
        """Validate pl.while_() body structure."""
        if not stmt.body:
            raise ParserSyntaxError(
                "pl.while_() body cannot be empty",
                span=self.span_tracker.get_span(stmt),
                hint="Add pl.cond(condition) as first statement",
            )

        if not self._is_cond_call(stmt.body[0]):
            raise ParserSyntaxError(
                "First statement in pl.while_() body must be pl.cond(condition)",
                span=self.span_tracker.get_span(stmt.body[0]),
                hint="Add pl.cond(condition) as first statement",
            )

    def _validate_while_target(self, stmt: ast.For, init_values: list[ir.Expr]) -> ast.Tuple:
        """Validate and return pl.while_() target tuple."""
        if not isinstance(stmt.target, ast.Tuple):
            raise ParserSyntaxError(
                "While loop target must be a tuple for pl.while_()",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for (var1, var2) in pl.while_(init_values=(...,))",
            )

        iter_args_node = stmt.target

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        return iter_args_node

    def _setup_while_iter_args(
        self, loop: Any, iter_args_node: ast.Tuple, init_values: list[ir.Expr]
    ) -> None:
        """Set up iter_args for pl.while_() loop."""
        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

    def _parse_while_body_statements(self, stmt: ast.For) -> list[str]:
        """Parse body statements for pl.while_() loop, return yielded vars."""
        prev_yield_tracker = getattr(self, "_current_yield_vars", None)
        self._current_yield_vars = []
        prev_yield_types = getattr(self, "_current_yield_types", None)
        self._current_yield_types = {}

        # Parse body (skip first statement which is pl.cond())
        for i, body_stmt in enumerate(stmt.body):
            if i == 0:
                continue  # Skip the pl.cond() statement

            # Check if pl.cond() appears anywhere else in body
            if self._is_cond_call(body_stmt):
                raise ParserSyntaxError(
                    "pl.cond() can only be the first statement in a pl.while_() loop body",
                    span=self.span_tracker.get_span(body_stmt),
                    hint="Remove this pl.cond() - condition is already specified at the start",
                )

            self.parse_statement(body_stmt)

        loop_output_vars = self._current_yield_vars[:]
        self._current_yield_vars = prev_yield_tracker
        self._current_yield_types = prev_yield_types
        return loop_output_vars

    def _register_while_outputs(self, loop: Any, loop_output_vars: list[str]) -> None:
        """Register output variables from pl.while_() loop."""
        loop_result = loop.get_result()
        if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
            for i, var_name in enumerate(loop_output_vars):
                if i < len(loop_result.return_vars):
                    self.scope_manager.define_var(var_name, loop_result.return_vars[i])

    def _parse_while_as_for(self, stmt: ast.For, while_call: ast.Call) -> None:
        """Parse while loop using for...in pl.while_() pattern.

        Pattern: for (var1, var2) in pl.while_(init_values=(val1, val2)):
                     pl.cond(condition)
                     ...

        Args:
            stmt: For AST node
            while_call: Call to pl.while_()
        """
        # Validate and parse arguments
        self._validate_while_call_args(while_call)
        init_values = self._parse_while_init_values(while_call)
        self._validate_while_body(stmt)
        iter_args_node = self._validate_while_target(stmt, init_values)

        span = self.span_tracker.get_span(stmt)
        placeholder_condition = ir.ConstBool(True, span)

        with self.builder.while_loop(placeholder_condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self.scope_manager.enter_scope("while")

            # Set up iter_args
            self._setup_while_iter_args(loop, iter_args_node, init_values)

            # Parse and set the condition (now that iter_args are in scope)
            condition = self._extract_cond_call(stmt.body[0])
            if condition is None:
                raise ParserSyntaxError(
                    "First statement in pl.while_() body must be pl.cond(condition)",
                    span=self.span_tracker.get_span(stmt.body[0]),
                    hint="Add pl.cond(condition) as first statement",
                )
            loop.set_condition(condition)

            # Add return_vars
            for iter_arg_node in iter_args_node.elts:
                assert isinstance(iter_arg_node, ast.Name)
                loop.return_var(f"{iter_arg_node.id}_out")

            # Parse body statements
            loop_output_vars = self._parse_while_body_statements(stmt)

            self.scope_manager.exit_scope(leak_vars=False)
            self.in_while_loop = False
            self.current_loop_builder = None

        # Register output variables
        self._register_while_outputs(loop, loop_output_vars)

    def parse_while_loop(self, stmt: ast.While) -> None:
        """Parse natural while loop syntax.

        Natural while syntax: while condition: body

        This creates a WhileStmt without iter_args (non-SSA form).
        The C++ ConvertToSSA pass will convert it to SSA form if needed.

        Args:
            stmt: While AST node
        """
        # Parse natural while syntax: while condition:
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)

        with self.builder.while_loop(condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self.scope_manager.enter_scope("while")

            # Parse body statements
            for body_stmt in stmt.body:
                self.parse_statement(body_stmt)

            # Variables leak to outer scope (ConvertToSSA will handle)
            self.scope_manager.exit_scope(leak_vars=True)
            self.in_while_loop = False
            self.current_loop_builder = None

    def parse_if_statement(self, stmt: ast.If) -> None:
        """Parse if statement with phi nodes.

        When pl.yield_() is used, phi nodes are created via return_vars.
        When no yields are used (plain syntax), variables leak to outer scope
        and the C++ ConvertToSSA pass handles creating phi nodes.

        Args:
            stmt: If AST node
        """
        # Parse condition
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)

        # Auto-sync: save pre-if buffer states and pending backward waits
        if self.sync_tracker is not None:
            self.sync_tracker.enter_if_branch()
            # Save pending backward waits so both branches can flush them
            import copy
            self._pre_if_pending_backward_waits = copy.deepcopy(
                getattr(self, "_pending_backward_waits", None)
            )

        # Track yield output variable names from both branches
        then_yield_vars = []

        # Begin if statement
        with self.builder.if_stmt(condition, span) as if_builder:
            self.current_if_builder = if_builder
            self.in_if_stmt = True

            # Save and initialize yield trackers
            prev_yield_tracker = getattr(self, "_current_yield_vars", None)
            self._current_yield_vars = []
            prev_yield_types = getattr(self, "_current_yield_types", None)
            self._current_yield_types = {}

            # Scan for yield variable names (without executing)
            then_yield_vars = self._scan_for_yields(stmt.body)

            # Also scan else branch to handle yields in both branches
            if stmt.orelse:
                else_yield_vars = self._scan_for_yields(stmt.orelse)
                # Merge with then branch yields (then branch takes precedence for type)
                then_names = {name for name, _ in then_yield_vars}
                # Add else-only yields
                for name, annotation in else_yield_vars:
                    if name not in then_names:
                        then_yield_vars.append((name, annotation))

            # Determine if we should leak variables (no explicit yields)
            should_leak = not bool(then_yield_vars)

            # Parse then branch (yield types captured via _current_yield_types)
            # Save tuple-select cache so entries added in the then-branch
            # don't leak into the else-branch (the phi vars are scoped to
            # the branch where they were emitted).
            saved_tuple_cache = dict(self._tuple_select_cache)
            self.scope_manager.enter_scope("if")
            for then_stmt in stmt.body:
                self.parse_statement(then_stmt)
            self.scope_manager.exit_scope(leak_vars=should_leak)

            # Parse else branch if present
            if stmt.orelse:
                # Auto-sync: save then-states, restore pre-if for else
                if self.sync_tracker is not None:
                    self.sync_tracker.enter_else_branch()
                    # Restore pending backward waits so else branch can flush them too
                    import copy
                    saved = getattr(self, "_pre_if_pending_backward_waits", None)
                    if saved is not None:
                        self._pending_backward_waits = copy.deepcopy(saved)

                # Restore tuple-select cache to pre-then state so the
                # else branch generates its own phi vars instead of
                # referencing ones defined only inside the then branch.
                self._tuple_select_cache = saved_tuple_cache

                if_builder.else_()
                self.scope_manager.enter_scope("else")
                for else_stmt in stmt.orelse:
                    self.parse_statement(else_stmt)
                self.scope_manager.exit_scope(leak_vars=should_leak)

            # Declare return vars AFTER parsing branches so captured yield types
            # are available for unannotated yields (fixes issue #233 / #234)
            for var_name, annotation in then_yield_vars:
                if annotation is not None:
                    var_type = self._resolve_yield_var_type(annotation)
                elif var_name in self._current_yield_types:
                    var_type = self._current_yield_types[var_name]
                else:
                    var_type = self._resolve_yield_var_type(None)
                if_builder.return_var(var_name, var_type)

            # Restore previous yield trackers
            self._current_yield_vars = prev_yield_tracker
            self._current_yield_types = prev_yield_types

        # Restore tuple-select cache: entries added inside either branch
        # are not valid in the outer scope.
        self._tuple_select_cache = saved_tuple_cache

        # Auto-sync: merge branch states conservatively
        if self.sync_tracker is not None:
            self.sync_tracker.exit_if()

        # After if statement completes, register the output variables in the outer scope
        if then_yield_vars:
            # Get the output variables from the if statement
            if_result = if_builder.get_result()
            if hasattr(if_result, "return_vars") and if_result.return_vars:
                # Register each output variable with its name (extract name from tuple)
                for i, (var_name, _) in enumerate(then_yield_vars):
                    if i < len(if_result.return_vars):
                        output_var = if_result.return_vars[i]
                        self.scope_manager.define_var(var_name, output_var)

        self.in_if_stmt = False
        self.current_if_builder = None

    def parse_with_statement(self, stmt: ast.With) -> None:
        """Parse with statement for scope contexts.

        Currently supports:
        - with pl.incore(): ... (creates ScopeStmt with InCore scope)
        - with pl.section_vector(): ... (creates SectionStmt with Vector section)
        - with pl.section_cube(): ... (creates SectionStmt with Cube section)

        Args:
            stmt: With AST node
        """
        # Check that we have exactly one context manager
        if len(stmt.items) != 1:
            raise ParserSyntaxError(
                "Only single context manager supported in with statement",
                span=self.span_tracker.get_span(stmt),
                hint="Use 'with pl.incore():' or 'with pl.section_vector():' without multiple context managers",
            )

        item = stmt.items[0]
        context_expr = item.context_expr

        # Check if this is pl.incore(), pl.section_vector(), or pl.section_cube()
        if isinstance(context_expr, ast.Call):
            func = context_expr.func
            if isinstance(func, ast.Attribute):
                span = self.span_tracker.get_span(stmt)
                
                # Handle pl.incore() - creates ScopeStmt
                if func.attr == "incore":
                    with self.builder.scope(ir.ScopeKind.InCore, span):
                        self.scope_manager.enter_scope("scope")
                        for body_stmt in stmt.body:
                            self.parse_statement(body_stmt)
                        self.scope_manager.exit_scope(leak_vars=False)
                    return
                
                # Handle pl.section_vector() - creates SectionStmt
                if func.attr == "section_vector":
                    with self.builder.section(ir.SectionKind.Vector, span):
                        self.scope_manager.enter_scope("section")
                        for body_stmt in stmt.body:
                            self.parse_statement(body_stmt)
                        self.scope_manager.exit_scope(leak_vars=False)
                    return
                
                # Handle pl.section_cube() - creates SectionStmt
                if func.attr == "section_cube":
                    with self.builder.section(ir.SectionKind.Cube, span):
                        self.scope_manager.enter_scope("section")
                        for body_stmt in stmt.body:
                            self.parse_statement(body_stmt)
                        self.scope_manager.exit_scope(leak_vars=False)
                    return

        # Unsupported context manager
        raise UnsupportedFeatureError(
            "Unsupported context manager in with statement",
            span=self.span_tracker.get_span(stmt),
            hint="Only 'with pl.incore():', 'with pl.section_vector():', or 'with pl.section_cube():' are currently supported",
        )

    def parse_return(self, stmt: ast.Return) -> None:
        """Parse return statement.

        In inline mode, captures return expression instead of emitting ReturnStmt.

        Args:
            stmt: Return AST node
        """
        if self._inline_mode:
            if stmt.value is None:
                return  # void inline, no return value
            if isinstance(stmt.value, ast.Tuple):
                exprs = [self.parse_expression(elt) for elt in stmt.value.elts]
                self._inline_return_expr = ir.MakeTuple(exprs, self.span_tracker.get_span(stmt))
            else:
                self._inline_return_expr = self.parse_expression(stmt.value)
            return

        span = self.span_tracker.get_span(stmt)

        if stmt.value is None:
            self.builder.return_stmt(None, span)
            return

        # Handle tuple return
        if isinstance(stmt.value, ast.Tuple):
            return_exprs = []
            for elt in stmt.value.elts:
                return_exprs.append(self.parse_expression(elt))
            self.builder.return_stmt(return_exprs, span)
        else:
            # Single return value
            return_expr = self.parse_expression(stmt.value)
            self.builder.return_stmt([return_expr], span)

    def parse_break(self, stmt: ast.Break) -> None:
        """Parse break statement.

        Args:
            stmt: Break AST node
        """
        span = self.span_tracker.get_span(stmt)
        self.builder.break_stmt(span)

    def parse_continue(self, stmt: ast.Continue) -> None:
        """Parse continue statement.

        Args:
            stmt: Continue AST node
        """
        span = self.span_tracker.get_span(stmt)
        self.builder.continue_stmt(span)

    # -----------------------------------------------------------------------
    # Sync-op statement expander — event_id=tuple[index] support
    # -----------------------------------------------------------------------

    _SYNC_OP_NAMES: frozenset[str] = frozenset({"sync_src", "sync_dst"})

    def _expand_stmt_level(self, node: ast.expr) -> bool:
        """Try to expand a system sync-op call with a subscript event_id at statement level.

        When a sync op is written as:
            pl.system.sync_src(..., event_id=event_ids[buf_idx])
        where ``event_ids`` is a tuple of integer constants, this expands it
        into an if-else chain so each branch contains a sync call with a
        static constant event_id.

        Returns True if expansion was performed (caller should skip normal handling).

        Note: kept for potential future statement-level expansion use cases.
        """
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in self._SYNC_OP_NAMES
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "system"
        ):
            return False
        # Find event_id kwarg with a subscript value
        event_id_kw = next(
            (kw for kw in node.keywords if kw.arg == "event_id" and isinstance(kw.value, ast.Subscript)),
            None,
        )
        if event_id_kw is None:
            return False

        span = self.span_tracker.get_span(node)
        subscript = event_id_kw.value
        assert isinstance(subscript, ast.Subscript)

        # Resolve the tuple of integer constants
        constants = self._resolve_const_event_id_tuple(subscript.value, span)
        if constants is None:
            return False  # fall through to normal handling, which will error on ConvertKwargsDict

        # Parse index expression
        index_expr = self.parse_expression(subscript.slice)

        # Parse all other kwargs (excluding event_id)
        other_kwargs: dict[str, Any] = {
            kw.arg: self._resolve_single_kwarg(kw.arg, kw.value)
            for kw in node.keywords
            if kw.arg is not None and kw.arg != "event_id"
        }

        op_name = func.attr
        op_func = getattr(ir_op.system, op_name)
        self._build_sync_event_chain(op_func, other_kwargs, index_expr, constants, 0, span)
        return True

    def _resolve_const_event_id_tuple(self, node: ast.expr, span: ir.Span) -> list[int] | None:
        """Return the list of integer constants for a tuple AST node, or None if not resolvable."""
        # Case 1: literal tuple, e.g. (0, 1)
        if isinstance(node, ast.Tuple):
            if all(isinstance(elt, ast.Constant) and isinstance(elt.value, int) for elt in node.elts):
                return [elt.value for elt in node.elts]  # type: ignore[union-attr]
        # Case 2: name in constant registry, e.g. event_ids = (0, 1) assigned earlier
        if isinstance(node, ast.Name) and node.id in self._const_tuple_registry:
            return self._const_tuple_registry[node.id]
        return None

    def _build_sync_event_chain(
        self,
        op_func: Any,
        other_kwargs: dict[str, Any],
        index_expr: ir.Expr,
        constants: list[int],
        level: int,
        span: ir.Span,
    ) -> None:
        """Recursively build if-else chain of sync calls with constant event_ids.

        Generates:
            if index == 0: sync_op(event_id=constants[0])
            else:
              if index == 1: sync_op(event_id=constants[1])
              else: sync_op(event_id=constants[n-1])   # leaf
        """
        n = len(constants)
        if level == n - 1:
            # Leaf: always emit this variant
            call_expr = op_func(**other_kwargs, event_id=constants[level], span=span)
            self.builder.eval_stmt(call_expr, span)
            return

        cond = index_expr == level
        with self.builder.if_stmt(cond, span) as if_b:
            # Then branch
            call_expr = op_func(**other_kwargs, event_id=constants[level], span=span)
            self.builder.eval_stmt(call_expr, span)
            if_b.else_()
            # Else branch: recurse
            self._build_sync_event_chain(op_func, other_kwargs, index_expr, constants, level + 1, span)
            # No return_var — this is a non-value-returning scf.if

    def parse_evaluation_statement(self, stmt: ast.Expr) -> None:
        """Parse evaluation statement (EvalStmt).

        Evaluation statements represent operations executed for their side effects,
        with the return value discarded (e.g., synchronization barriers).

        Args:
            stmt: Expr AST node
        """
        expr = self.parse_expression(stmt.value)
        span = self.span_tracker.get_span(stmt)

        # Void inline functions return None — nothing to emit
        if expr is None:
            return

        # Validate that we got an IR expression (not a list literal, etc.)
        if not isinstance(expr, ir.Expr):
            raise ParserSyntaxError(
                f"Evaluation statement must be an IR expression, got {type(expr).__name__}",
                span=span,
                hint="Only function calls and operations can be used as standalone statements",
            )

        # Emit EvalStmt using builder method
        self.builder.eval_stmt(expr, span)

    def parse_expression(self, expr: ast.expr) -> ir.Expr:
        """Parse expression and return IR Expr.

        Args:
            expr: AST expression node

        Returns:
            IR expression
        """
        if isinstance(expr, ast.Name):
            return self.parse_name(expr)
        elif isinstance(expr, ast.Constant):
            return self.parse_constant(expr)
        elif isinstance(expr, ast.BinOp):
            return self.parse_binop(expr)
        elif isinstance(expr, ast.Compare):
            return self.parse_compare(expr)
        elif isinstance(expr, ast.Call):
            return self.parse_call(expr)
        elif isinstance(expr, ast.Attribute):
            return self.parse_attribute(expr)
        elif isinstance(expr, ast.UnaryOp):
            return self.parse_unaryop(expr)
        elif isinstance(expr, ast.List):
            return self.parse_list(expr)
        elif isinstance(expr, ast.Tuple):
            return self.parse_tuple_literal(expr)
        elif isinstance(expr, ast.Subscript):
            return self.parse_subscript(expr)
        else:
            raise UnsupportedFeatureError(
                f"Unsupported expression type: {type(expr).__name__}",
                span=self.span_tracker.get_span(expr),
                hint="Use supported expressions like variables, constants, operations, or function calls",
            )

    def parse_name(self, name: ast.Name) -> ir.Expr | Any:
        """Parse variable name reference.

        Resolves names by checking the DSL scope first, then falling back
        to closure variables from the enclosing Python scope.

        Args:
            name: Name AST node

        Returns:
            IR expression (Var from scope, or constant/tuple from closure)
        """
        var_name = name.id
        # Check if it's a Python variable (non-IR value like TileType)
        python_var = self.scope_manager.get_python_var(var_name)
        if python_var is not None:
            return python_var

        # In inline mode, restrict IR variable lookup to the inline scope only.
        # Variables from the caller's scope must be passed as function arguments.
        # Exception: tile buffers (TileType, TupleType, TensorType) are allowed
        # from outer scope since they represent hardware resources, not scalars.
        if self._inline_mode:
            var = self.scope_manager.lookup_var_bounded(var_name, barrier="inline")
            if var is None:
                # Check if the variable exists in outer scope and is a tile/tensor/tuple
                outer_var = self.scope_manager.lookup_var(var_name)
                if outer_var is not None and hasattr(outer_var, "type"):
                    vtype = outer_var.type
                    if (isinstance(vtype, ir.TileType)
                            or isinstance(vtype, ir.TupleType)
                            or isinstance(vtype, ir.TensorType)):
                        var = outer_var
        else:
            var = self.scope_manager.lookup_var(var_name)

        if var is not None:
            return var

        # Fall back to closure variables
        result = self.expr_evaluator.try_eval_as_ir(name)
        if result is not None:
            return result

        raise UndefinedVariableError(
            f"Undefined variable '{var_name}'",
            span=self.span_tracker.get_span(name),
            hint="Check if the variable is defined before using it or is available in the enclosing scope",
        )

    def parse_constant(self, const: ast.Constant) -> ir.Expr:
        """Parse constant value.

        Args:
            const: Constant AST node

        Returns:
            IR constant expression
        """
        span = self.span_tracker.get_span(const)
        value = const.value

        if isinstance(value, bool):
            return ir.ConstBool(value, span)
        elif isinstance(value, int):
            return ir.ConstInt(value, DataType.INDEX, span)
        elif isinstance(value, float):
            return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, span)
        else:
            raise ParserTypeError(
                f"Unsupported constant type: {type(value)}",
                span=self.span_tracker.get_span(const),
                hint="Use int, float, or bool constants",
            )

    def parse_binop(self, binop: ast.BinOp) -> ir.Expr:
        """Parse binary operation.

        Args:
            binop: BinOp AST node

        Returns:
            IR binary expression
        """
        span = self.span_tracker.get_span(binop)
        left = self.parse_expression(binop.left)
        right = self.parse_expression(binop.right)

        op_map = {
            ast.Add: ir.add,
            ast.Sub: ir.sub,
            ast.Mult: ir.mul,
            ast.Div: ir.truediv,
            ast.FloorDiv: ir.floordiv,
            ast.Mod: ir.mod,
        }

        op_type = type(binop.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported binary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(binop),
                hint="Use supported operators: +, -, *, /, //, %",
            )

        return op_map[op_type](left, right, span)

    def parse_compare(self, compare: ast.Compare) -> ir.Expr:
        """Parse comparison operation.

        Args:
            compare: Compare AST node

        Returns:
            IR comparison expression
        """
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise ParserSyntaxError(
                "Only simple comparisons supported",
                span=self.span_tracker.get_span(compare),
                hint="Use single comparison operators like: a < b, not chained comparisons",
            )

        span = self.span_tracker.get_span(compare)
        left = self.parse_expression(compare.left)
        right = self.parse_expression(compare.comparators[0])

        op_map = {
            ast.Eq: ir.eq,
            ast.NotEq: ir.ne,
            ast.Lt: ir.lt,
            ast.LtE: ir.le,
            ast.Gt: ir.gt,
            ast.GtE: ir.ge,
        }

        op_type = type(compare.ops[0])
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported comparison: {op_type.__name__}",
                span=self.span_tracker.get_span(compare),
                hint="Use supported comparisons: ==, !=, <, <=, >, >=",
            )

        return op_map[op_type](left, right, span)

    def parse_unaryop(self, unary: ast.UnaryOp) -> ir.Expr:
        """Parse unary operation.

        Args:
            unary: UnaryOp AST node

        Returns:
            IR unary expression
        """
        span = self.span_tracker.get_span(unary)
        operand = self.parse_expression(unary.operand)

        op_map = {
            ast.USub: ir.neg,
            ast.Not: ir.bit_not,
        }

        op_type = type(unary.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported unary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(unary),
                hint="Use supported unary operators: -, not",
            )

        return op_map[op_type](operand, span)

    def parse_call(self, call: ast.Call) -> ir.Expr:
        """Parse function call.

        Args:
            call: Call AST node

        Returns:
            IR expression from call
        """
        func = call.func
        # Handle TileType(...) - type descriptor, not an operation
        if isinstance(func, ast.Name) and func.id == "TileType":
            return self._parse_tile_type_call(call)

        # Handle pl.yield_() specially
        if isinstance(func, ast.Attribute) and func.attr == "yield_":
            return self.parse_yield_call(call)

        # Handle cross-function calls via self.method_name() in @pl.program classes
        if isinstance(func, ast.Attribute):
            # Check for self.method_name pattern
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                method_name = func.attr
                if method_name in self.global_vars:
                    gvar = self.global_vars[method_name]
                    args = [self.parse_expression(arg) for arg in call.args]
                    span = self.span_tracker.get_span(call)

                    # Use return type from the parsed function if available
                    func_obj = self.gvar_to_func.get(gvar)
                    return_types = func_obj.return_types if func_obj else []
                    return self._make_call_with_return_type(gvar, args, return_types, span)
                else:
                    raise UndefinedVariableError(
                        f"Function '{method_name}' not defined in program",
                        span=self.span_tracker.get_span(call),
                        hint=f"Available functions: {list(self.global_vars.keys())}",
                    )

            # Handle pl.tensor.*, pl.block.*, and pl.* operation calls
            return self.parse_op_call(call)

        # Handle bare-name calls to external ir.Function or InlineFunction
        if isinstance(func, ast.Name):
            from .decorator import InlineFunction, KernelFunction  # noqa: PLC0415 (circular import)

            func_name = func.id
            resolved = self.expr_evaluator.closure_vars.get(func_name)
            if isinstance(resolved, ir.Function):
                return self._parse_external_function_call(func_name, resolved, call)
            if isinstance(resolved, InlineFunction):
                return self._parse_inline_call(func_name, resolved, call)
            if isinstance(resolved, KernelFunction):
                return self._parse_func_call(func_name, resolved, call)
            # Implicit func: annotated → func.call, unannotated → auto-inline
            if callable(resolved) and not isinstance(resolved, type):
                import inspect as _inspect  # noqa: PLC0415
                hints = getattr(resolved, "__annotations__", {})
                params = [p for p in _inspect.signature(resolved).parameters if p != "self"]
                # Annotated: all params have hints, OR no params but has return hint
                fully_annotated = (
                    (params and all(p in hints for p in params))
                    or (not params and "return" in hints)
                )
                if fully_annotated:
                    return self._implicit_func_call(func_name, resolved, call)
                return self._auto_inline_call(func_name, resolved, call)

        raise UnsupportedFeatureError(
            f"Unsupported function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.* operations, pl.yield_(), self.method() for cross-function calls, "
            "or call an external @pl.function / @pl.inline by name",
        )

    def parse_yield_call(self, call: ast.Call) -> ir.Expr:
        """Parse pl.yield_() call.

        Args:
            call: Call to pl.yield_() or pl.yield_()

        Returns:
            IR expression (first yielded value for single yield)
        """
        span = self.span_tracker.get_span(call)
        yield_exprs = []

        for arg in call.args:
            expr = self.parse_expression(arg)
            yield_exprs.append(expr)

        # Emit yield statement
        self.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Track yielded variables for if statement processing
        # This is for single assignment like: var = pl.yield_(expr)
        # We'll return a placeholder that gets resolved when if statement completes

        # Return first expression as the "value" of the yield
        # This handles: var = pl.yield_(expr)
        if len(yield_exprs) == 1:
            return yield_exprs[0]

        # For multiple yields, this should be handled as tuple assignment
        raise ParserSyntaxError(
            "Multiple yields should use tuple unpacking assignment",
            span=self.span_tracker.get_span(call),
            hint="Use tuple unpacking: (a, b) = pl.yield_(x, y)",
        )

    def parse_op_call(self, call: ast.Call) -> ir.Expr:
        """Parse operation call like pl.tensor.create_tensor() or pl.add().

        Args:
            call: Call AST node

        Returns:
            IR expression from operation
        """
        func = call.func

        # Navigate through attribute chain to find operation
        # e.g., pl.tensor.create_tensor -> ["pl", "tensor", "create_tensor"]
        # e.g., pl.add -> ["pl", "add"]
        attrs = []
        node = func
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)

        # pl.tensor.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] in ("pl", "plm") and attrs[1] == "tensor":
            op_name = attrs[2]
            return self._parse_tensor_op(op_name, call)

        # pl.block.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] in ("pl", "plm") and attrs[1] == "block":
            op_name = attrs[2]
            return self._parse_block_op(op_name, call)

        # pl.system.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "system":
            op_name = attrs[2]
            return self._parse_system_op(op_name, call)

        # pl.const(value, dtype) — typed constant literal
        if len(attrs) >= 2 and attrs[0] in ("pl", "plm") and attrs[1] == "const":
            return self._parse_typed_constant(call)
        
        # plm.TileType(...) - type descriptor, not an operation
        if len(attrs) == 2 and attrs[0] == "plm" and attrs[1] == "TileType":
            return self._parse_tile_type_call(call)

        # plm.{operation} (2-segment) — manual (non-SSA) ops
        if len(attrs) == 2 and attrs[0] == "plm" and attrs[1] != "const":
            return self._parse_manual_op(attrs[1], call)

        # pl.{operation} (2-segment, unified dispatch or promoted ops)
        if len(attrs) >= 2 and attrs[0] in ("pl", "plm") and attrs[1] not in ("tensor", "block", "system", "TileType"):
            op_name = attrs[1]
            return self._parse_unified_op(op_name, call)

        raise UnsupportedFeatureError(
            f"Unsupported operation call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.*, pl.tensor.*, pl.block.*, or pl.system.* operations",
        )

    def _make_call_with_return_type(
        self,
        gvar: ir.GlobalVar,
        args: list[ir.Expr],
        return_types: list[ir.Type],
        span: ir.Span,
    ) -> ir.Expr:
        """Create an ir.Call, attaching the return type when known.

        Args:
            gvar: GlobalVar identifying the callee
            args: Parsed argument expressions
            return_types: The callee's return type list (may be empty)
            span: Source span for the call
        """
        if not return_types:
            return ir.Call(gvar, args, span)
        if len(return_types) == 1:
            return ir.Call(gvar, args, return_types[0], span)
        return ir.Call(gvar, args, ir.TupleType(return_types), span)

    def _parse_external_function_call(
        self, _local_name: str, ext_func: ir.Function, call: ast.Call
    ) -> ir.Expr:
        """Parse a call to an externally-defined ir.Function.

        Args:
            _local_name: The name used in the caller's scope (may be aliased)
            ext_func: The external ir.Function object
            call: The AST Call node
        """
        func_name = ext_func.name
        span = self.span_tracker.get_span(call)

        # Validate no naming conflict with internal program functions
        if func_name in self.global_vars:
            raise ParserSyntaxError(
                f"External function '{func_name}' conflicts with program function '{func_name}'",
                span=span,
                hint="Rename either the external or program function to avoid the name conflict",
            )

        # Check for conflicting externals with same .name but different objects
        if func_name in self.external_funcs and self.external_funcs[func_name] is not ext_func:
            raise ParserSyntaxError(
                f"Conflicting external functions with name '{func_name}'",
                span=span,
                hint="External functions must have unique names; rename one of the functions",
            )

        # Track the external function
        self.external_funcs[func_name] = ext_func

        args = [self.parse_expression(arg) for arg in call.args]
        gvar = ir.GlobalVar(func_name)
        return self._make_call_with_return_type(gvar, args, ext_func.return_types, span)

    @staticmethod
    def _is_docstring(stmt: ast.stmt) -> bool:
        """Check if an AST statement is a docstring (string constant expression)."""
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    def _parse_inline_call(self, _local_name: str, inline_func: "InlineFunction", call: ast.Call) -> ir.Expr | None:
        """Parse a call to an InlineFunction, expanding its body in-place.

        Args:
            _local_name: The name used in the caller's scope
            inline_func: The InlineFunction object
            call: The AST Call node
        """
        span = self.span_tracker.get_span(call)

        expected = len(inline_func.param_names)
        got = len(call.args)
        if got < expected:
            raise ParserTypeError(
                f"Inline function '{inline_func.name}' expects {expected} argument(s), got {got}",
                span=span,
                hint=f"Check the inline function's parameter list: {inline_func.param_names}",
            )

        # Parse call arguments in the caller's context before entering inline scope
        arg_exprs = [self.parse_expression(arg) for arg in call.args]

        self.scope_manager.enter_scope("inline")
        for param_name, arg_expr in zip(inline_func.param_names, arg_exprs):
            self.scope_manager.define_var(param_name, arg_expr, allow_redef=True)

        # Handle extra arguments: inject them as closure variables
        extra_closure_vars = {}
        if got > expected:
            for i in range(expected, got):
                arg_node = call.args[i]
                if isinstance(arg_node, ast.Name):
                    var_name = arg_node.id
                    extra_closure_vars[var_name] = arg_exprs[i]

        # Save parser state and switch to the inline function's context
        prev_inline_state = (self._inline_mode, self._inline_return_expr, self._inline_prefix)
        self._inline_mode = True
        self._inline_return_expr = None
        self._inline_prefix = f"_il{self._inline_counter}_"
        self._inline_counter += 1

        prev_closure_vars = self.expr_evaluator.closure_vars
        self.expr_evaluator.closure_vars = {**inline_func.closure_vars, **prev_closure_vars, **extra_closure_vars}

        prev_span_state = (
            self.span_tracker.source_file,
            self.span_tracker.source_lines,
            self.span_tracker.line_offset,
            self.span_tracker.col_offset,
        )
        self.span_tracker.source_file = inline_func.source_file
        self.span_tracker.source_lines = inline_func.source_lines
        self.span_tracker.line_offset = inline_func.line_offset
        self.span_tracker.col_offset = inline_func.col_offset

        try:
            for i, stmt in enumerate(inline_func.func_def.body):
                if i == 0 and self._is_docstring(stmt):
                    continue
                self.parse_statement(stmt)
        finally:
            # Restore parser state
            (
                self.span_tracker.source_file,
                self.span_tracker.source_lines,
                self.span_tracker.line_offset,
                self.span_tracker.col_offset,
            ) = prev_span_state
            self.expr_evaluator.closure_vars = prev_closure_vars
            return_expr = self._inline_return_expr
            self._inline_mode, self._inline_return_expr, self._inline_prefix = prev_inline_state
            # Leak vars so inlined definitions are visible to the caller
            self.scope_manager.exit_scope(leak_vars=True)

        return return_expr

    @staticmethod
    def _check_no_nested_calls(func_def: ast.FunctionDef, func_name: str, span: Any) -> None:
        """Raise UnsupportedFeatureError if the function body contains bare-name function calls."""
        for node in ast.walk(func_def):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                raise UnsupportedFeatureError(
                    f"Auto-inlined function '{func_name}' cannot call other functions "
                    f"(found call to '{node.func.id}'). "
                    f"Add type annotations to use func.call, or use @pl.inline.",
                    span=span,
                )

    def _auto_inline_call(self, func_name: str, fn: Callable, call: ast.Call) -> ir.Expr | None:
        """Inline an unannotated plain Python function at the call site.

        The function body is expanded in-place (like @pl.inline). Nested bare-name
        function calls are forbidden; raise UnsupportedFeatureError if found.

        Args:
            func_name: Name used at the call site
            fn: The callable Python function (no DSL annotations)
            call: AST Call node

        Returns:
            IR expression (inlined return value)
        """
        import textwrap as _tw  # noqa: PLC0415

        from .decorator import InlineFunction, _get_source_info  # noqa: PLC0415

        span = self.span_tracker.get_span(call)

        try:
            source_file, source_lines_raw, starting_line = _get_source_info(fn, "function")
        except Exception as e:
            raise UnsupportedFeatureError(
                f"Cannot auto-inline '{func_name}': unable to retrieve source — {e}",
                span=span,
                hint=f"Define '{func_name}' in a .py file, or use @pl.inline",
            ) from e

        source_code = _tw.dedent("".join(source_lines_raw))
        col_offset = len(source_lines_raw[0]) - len(source_lines_raw[0].lstrip()) if source_lines_raw else 0
        line_offset = starting_line - 1
        source_lines = source_code.split("\n")

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise UnsupportedFeatureError(
                f"Cannot parse '{func_name}': {e}",
                span=span,
                hint=f"Use @pl.inline to explicitly mark '{func_name}' as an inline helper",
            ) from e

        func_def = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn.__name__),
            None,
        )
        if func_def is None:
            raise UnsupportedFeatureError(
                f"Cannot find function definition for '{func_name}' in source",
                span=span,
                hint=f"Use @pl.inline to explicitly mark '{func_name}' as an inline helper",
            )

        # Constraint: no nested bare-name function calls
        self._check_no_nested_calls(func_def, func_name, span)

        fn_closure: dict[str, Any] = {**fn.__globals__}
        if fn.__closure__ and fn.__code__.co_freevars:
            fn_closure.update(
                dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
            )

        param_names = [a.arg for a in func_def.args.args if a.arg != "self"]
        inline_func = InlineFunction(
            name=fn.__name__,
            func_def=func_def,
            param_names=param_names,
            source_file=source_file,
            source_lines=source_lines,
            line_offset=line_offset,
            col_offset=col_offset,
            closure_vars={**fn_closure, **self.expr_evaluator.closure_vars},
        )
        return self._parse_inline_call(func_name, inline_func, call)

    def _implicit_func_call(self, func_name: str, fn: Callable, call: ast.Call) -> ir.Expr:
        """Compile an annotated plain Python function as a KernelFunction and emit func.call.

        Functions with complete DSL type annotations are compiled on first encounter and
        cached by id(fn). Subsequent calls reuse the cached KernelFunction.

        Functions without annotations raise UnsupportedFeatureError with a helpful hint.

        Args:
            func_name: Name used at the call site
            fn: The callable Python function
            call: AST Call node

        Returns:
            IR expression (func.call result)
        """
        import textwrap as _tw

        from .decorator import KernelFunction, _get_source_info  # noqa: PLC0415
        from .diagnostics import ParserError, ParserTypeError as _ParserTypeError  # noqa: PLC0415

        span = self.span_tracker.get_span(call)

        fn_id = id(fn)
        if fn_id in self._implicit_func_cache:
            return self._parse_func_call(func_name, self._implicit_func_cache[fn_id], call)

        # Retrieve source
        try:
            source_file, source_lines_raw, starting_line = _get_source_info(fn, "function")
        except Exception as e:
            raise UnsupportedFeatureError(
                f"Cannot compile '{func_name}': unable to retrieve source — {e}",
                span=span,
                hint=f"Define '{func_name}' in a .py file, or use @pl.func",
            ) from e

        source_code = _tw.dedent("".join(source_lines_raw))
        col_offset = len(source_lines_raw[0]) - len(source_lines_raw[0].lstrip()) if source_lines_raw else 0
        line_offset = starting_line - 1
        source_lines = source_code.split("\n")

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise UnsupportedFeatureError(
                f"Cannot parse '{func_name}': {e}",
                span=span,
                hint=f"Use @pl.func to explicitly mark '{func_name}' as a DSL helper",
            ) from e

        func_def = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn.__name__),
            None,
        )
        if func_def is None:
            raise UnsupportedFeatureError(
                f"Cannot find function definition for '{func_name}' in source",
                span=span,
                hint=f"Use @pl.func to explicitly mark '{func_name}' as a DSL helper",
            )

        # Build closure for the function
        fn_closure: dict[str, Any] = {**fn.__globals__}
        if fn.__closure__ and fn.__code__.co_freevars:
            fn_closure.update(
                dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
            )

        sub_parser = ASTParser(
            source_file,
            source_lines,
            line_offset,
            col_offset,
            closure_vars={**fn_closure, **self.expr_evaluator.closure_vars},
        )
        # Share cache so nested implicit calls are deduplicated
        sub_parser._implicit_func_cache = self._implicit_func_cache

        try:
            ir_func = sub_parser.parse_function(func_def, func_type=ir.FunctionType.Helper)
        except _ParserTypeError as e:
            raise UnsupportedFeatureError(
                f"'{func_name}' called from kernel but has no DSL type annotations. "
                f"Add annotations or use @pl.func.",
                span=span,
                hint=f"Example: def {func_name}(x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]: ...",
            ) from e
        except ParserError:
            raise

        gvar = ir.GlobalVar(fn.__name__)
        param_names = [a.arg for a in func_def.args.args if a.arg != "self"]
        kfunc = KernelFunction(
            name=fn.__name__,
            ir_function=ir_func,
            gvar=gvar,
            param_names=param_names,
        )
        self._implicit_func_cache[fn_id] = kfunc
        # Merge nested implicit functions discovered by the sub-parser
        self.external_funcs.update(sub_parser.external_funcs)
        # Register this function so its definition is included in the program
        self.external_funcs[fn.__name__] = ir_func
        return self._parse_func_call(func_name, kfunc, call)

    def _parse_func_call(self, func_name: str, kfunc: "KernelFunction", call: ast.Call) -> ir.Expr:
        """Parse a call to a @pl.func function, emitting an ir.Call for func.call generation.

        Args:
            func_name: Name used at the call site
            kfunc: KernelFunction holding the compiled ir.Function
            call: AST Call node

        Returns:
            ir.Call expression with the function's GlobalVar and parsed arguments
        """
        from .decorator import KernelFunction  # noqa: PLC0415

        span = self.span_tracker.get_span(call)
        expected = len(kfunc.param_names)
        got = len(call.args)
        if got != expected:
            raise ParserTypeError(
                f"Function '{func_name}' expects {expected} argument(s), got {got}",
                span=span,
                hint=f"Parameters: {kfunc.param_names}",
            )

        arg_exprs = [self.parse_expression(arg) for arg in call.args]
        return_types = list(kfunc.ir_function.return_types)
        return self._make_call_with_return_type(kfunc.gvar, arg_exprs, return_types, span)

    def _parse_tile_type_call(self, call: ast.Call) -> Any:
        """Parse TileType(...) as a dataclass instantiation."""
        from pypto.language.op.manual.op.manual_ops import TileType

        kwargs = {}
        for kw in call.keywords:
            kwargs[kw.arg] = self._resolve_single_kwarg(kw.arg, kw.value)

        return TileType(**kwargs)

    def _parse_struct_call(self, call: ast.Call, struct_name: str = "") -> _StructVar:
        """Parse pl.struct(field1=val1, field2=val2, ...) into a _StructVar.

        Each field is emitted as an IR variable via ``builder.let`` so that
        subsequent mutations produce proper IR ``AssignStmt`` nodes.  This is
        critical for loop-carried variables: the codegen must see the
        reassignment inside the loop body to generate correct MLIR.

        IR variable names are prefixed with ``_{struct_name}_`` to avoid
        collisions with standalone variables of the same name in other scopes
        (e.g. cube vs. vector section both using ``q_count``).

        Args:
            call: The AST Call node for pl.struct(...)
            struct_name: LHS variable name (e.g. "ctx") used to prefix IR names

        Returns:
            _StructVar with fields mapping to IR Vars
        """
        span = self.span_tracker.get_span(call)
        if call.args:
            raise ParserSyntaxError(
                "pl.struct() only accepts keyword arguments",
                span=span,
                hint="Use pl.struct(name1=val1, name2=val2, ...)",
            )
        fields: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError(
                    "pl.struct() does not support **kwargs",
                    span=span,
                )
            value_expr = self.parse_expression(kw.value)
            if struct_name:
                # Named struct: fields will use struct.get/set via C++ struct.
                # Don't emit IR variables — just record field names for validation.
                fields[kw.arg] = value_expr  # keep for validation only
            elif isinstance(value_expr, ir.Expr):
                # Unnamed struct: emit IR variable for codegen tracking
                ir_name = kw.arg
                var = self.builder.let(ir_name, value_expr, span=span)
                fields[kw.arg] = var
            else:
                fields[kw.arg] = value_expr
        return _StructVar(fields, name=struct_name)

    def _parse_op_kwargs(self, call: ast.Call) -> dict[str, Any]:
        """Parse keyword arguments for an operation call.

        Shared helper for tensor, block, system, and unified op parsing.

        Args:
            call: Call AST node

        Returns:
            Dictionary of keyword argument names to values
        """
        return {kw.arg: self._resolve_single_kwarg(kw.arg, kw.value) for kw in call.keywords}

    def _resolve_single_kwarg(self, key: str, value: ast.expr) -> Any:
        """Resolve a single keyword argument value to a Python or IR value.

        Args:
            key: Keyword argument name
            value: AST expression for the value

        Returns:
            Resolved Python or IR value
        """
        if key == "dtype":
            return self.type_resolver.resolve_dtype(value)
        elif isinstance(value, ast.Constant):
            return value.value
        elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
            return self._resolve_unary_kwarg(value)
        elif isinstance(value, ast.Name):
            return self._resolve_name_kwarg(value)
        elif isinstance(value, ast.Attribute):
            return self._resolve_attribute_kwarg(value)
        elif isinstance(value, ast.List):
            return self._resolve_list_kwarg(value)
        elif isinstance(value, ast.Subscript):
            return self._resolve_subscript_kwarg(value)
        else:
            return self.parse_expression(value)

    def _resolve_unary_kwarg(self, value: ast.UnaryOp) -> Any:
        """Resolve a unary op kwarg value (e.g., -1)."""
        if isinstance(value.operand, ast.Constant) and isinstance(value.operand.value, (int, float)):
            return -value.operand.value
        return self.parse_expression(value)

    def _resolve_name_kwarg(self, value: ast.Name) -> Any:
        """Resolve a Name kwarg value via scope lookup or closure eval."""
        if value.id in ["True", "False"]:
            return value.id == "True"
        if self.scope_manager.lookup_var(value.id) is not None:
            return self.parse_expression(value)  # IR var from scope
        # Not in IR scope — evaluate from closure (raises ParserTypeError if undefined)
        return self.expr_evaluator.eval_expr(value)

    def _resolve_attribute_kwarg(self, value: ast.Attribute) -> Any:
        """Resolve an Attribute kwarg value (e.g., pl.FP32, config.field)."""
        try:
            return self.type_resolver.resolve_dtype(value)
        except ParserTypeError:
            # Not a dtype — evaluate as a general expression from closure.
            # Use eval_expr (not try_eval_expr) so failures surface expression-specific
            # errors instead of the misleading dtype error from above.
            return self.expr_evaluator.eval_expr(value)

    def _resolve_list_kwarg(self, value: ast.List) -> Any:
        """Resolve a List kwarg value, trying closure eval first."""
        # If any element refers to a name in IR scope, parse as IR expressions
        # (mirrors _resolve_name_kwarg: IR scope takes priority over closure)
        if any(
            isinstance(elt, ast.Name) and self.scope_manager.lookup_var(elt.id) is not None
            for elt in value.elts
        ):
            return self.parse_list(value)
        success, result = self.expr_evaluator.try_eval_expr(value)
        if success and isinstance(result, list):
            return result
        return self.parse_list(value)

    def _resolve_subscript_kwarg(self, value: ast.Subscript) -> Any:
        """Resolve a Subscript kwarg value (e.g., MY_LIST[0]), trying closure eval first."""
        success, result = self.expr_evaluator.try_eval_expr(value)
        if success:
            return result
        return self.parse_expression(value)

    def _parse_tensor_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tensor operation.

        Args:
            op_name: Name of tensor operation
            call: Call AST node

        Returns:
            IR expression from tensor operation
        """
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)

        # Map language-level operation name to IR-level name if needed
        ir_op_name = self._TENSOR_OP_NAME_MAP.get(op_name, op_name)

        # Call the appropriate tensor operation
        if hasattr(ir_op.tensor, ir_op_name):
            op_func = getattr(ir_op.tensor, ir_op_name)
            call_span = self.span_tracker.get_span(call)
            return op_func(*args, **kwargs, span=call_span)

        raise InvalidOperationError(
            f"Unknown tensor operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid tensor operation",
        )

    def _parse_block_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse block operation.

        Args:
            op_name: Name of block operation
            call: Call AST node

        Returns:
            IR expression from block operation
        """
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)

        # Special handling for make_tile with TileType
        if op_name == "make_tile":
            from pypto.language.op.manual.op.manual_ops import TileType
            if len(args) >= 1 and isinstance(args[0], TileType):
                tile_type = args[0]
                # Extract parameters from TileType
                kwargs.setdefault("shape", tile_type.shape)
                kwargs.setdefault("dtype", tile_type.dtype)
                kwargs.setdefault("target_memory", tile_type.target_memory)
                if tile_type.valid_shape is not None:
                    kwargs.setdefault("valid_shape", tile_type.valid_shape)
                if tile_type.blayout is not None:
                    kwargs.setdefault("blayout", tile_type.blayout)
                if tile_type.slayout is not None:
                    kwargs.setdefault("slayout", tile_type.slayout)
                if tile_type.fractal is not None:
                    kwargs.setdefault("fractal", tile_type.fractal)
                if tile_type.pad is not None:
                    kwargs.setdefault("pad", tile_type.pad)
                if tile_type.compact is not None:
                    kwargs.setdefault("compact", tile_type.compact)
                # Remove TileType from args, keep addr and size
                args = args[1:]

        # Call the appropriate block operation
        if hasattr(ir_op.block, op_name):
            op_func = getattr(ir_op.block, op_name)
            call_span = self.span_tracker.get_span(call)
            return op_func(*args, **kwargs, span=call_span)

        raise InvalidOperationError(
            f"Unknown block operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid block operation",
        )

    def _parse_system_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse system operation.

        Args:
            op_name: Name of system operation
            call: Call AST node

        Returns:
            IR expression from system operation
        """
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)

        if hasattr(ir_op.system, op_name):
            op_func = getattr(ir_op.system, op_name)
            call_span = self.span_tracker.get_span(call)
            result = op_func(*args, **kwargs, span=call_span)
            # wait_cross_core acts as a pipeline fence: all local pipes
            # complete while the core blocks waiting for the signal.
            if op_name == "wait_cross_core" and self.sync_tracker is not None:
                self.sync_tracker.pipeline_fence()
            return result

        raise InvalidOperationError(
            f"Unknown system operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid system operation",
        )

    def _parse_debug_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse debug operation."""
        call_span = self.span_tracker.get_span(call)

        if op_name in {"dump_tensor", "dump_tile"}:
            args = [self.parse_expression(arg) for arg in call.args]
            kwargs = self._parse_op_kwargs(call)
            if hasattr(ir_op.debug, op_name):
                op_func = getattr(ir_op.debug, op_name)
                return op_func(*args, **kwargs, span=call_span)

            raise InvalidOperationError(
                f"Unknown debug operation: {op_name}",
                span=call_span,
                hint=f"Check if '{op_name}' is a valid debug operation",
            )

        if op_name == "assert_":
            if call.keywords:
                raise ParserSyntaxError(
                    "assert_ does not accept keyword arguments",
                    span=call_span,
                )
            if len(call.args) < 1:
                raise ParserSyntaxError(
                    f"assert_ requires at least 1 argument (condition), got {len(call.args)}",
                    span=call_span,
                )

            condition = self.parse_expression(call.args[0])
            condition_text = self.span_tracker.get_source_text(call.args[0])

            if len(call.args) == 1:
                return ir_op.debug.assert_(condition, condition_text=condition_text, span=call_span)

            format_node = call.args[1]
            if not isinstance(format_node, ast.Constant) or not isinstance(format_node.value, str):
                raise ParserTypeError(
                    "assert_ message must be a string literal",
                    span=self.span_tracker.get_span(format_node),
                    hint='Use a literal like plm.assert_(cond, "bad state") or plm.assert_(cond, "x=%d", x)',
                )

            args = [self.parse_expression(arg) for arg in call.args[2:]]
            return ir_op.debug.assert_(
                condition,
                format_node.value,
                *args,
                condition_text=condition_text,
                span=call_span,
            )

        if op_name == "trap":
            if call.keywords:
                raise ParserSyntaxError(
                    "trap does not accept keyword arguments",
                    span=call_span,
                )
            if call.args:
                raise ParserSyntaxError(
                    f"trap takes no arguments, got {len(call.args)}",
                    span=call_span,
                )

            return ir_op.debug.trap(span=call_span)

        if op_name == "printf":
            if call.keywords:
                raise ParserSyntaxError(
                    "printf does not accept keyword arguments",
                    span=call_span,
                )
            if len(call.args) < 1:
                raise ParserSyntaxError(
                    f"printf requires at least a format string, got {len(call.args)} arguments",
                    span=call_span,
                )

            format_node = call.args[0]
            if not isinstance(format_node, ast.Constant) or not isinstance(format_node.value, str):
                raise ParserTypeError(
                    "printf format must be a string literal",
                    span=self.span_tracker.get_span(format_node),
                    hint='Use a literal like plm.printf("hello\\n") or plm.printf("x=%d\\n", value)',
                )

            args = [self.parse_expression(arg) for arg in call.args[1:]]
            return ir_op.debug.printf(format_node.value, *args, span=call_span)

        raise InvalidOperationError(
            f"Unknown debug operation: {op_name}",
            span=call_span,
            hint=f"Check if '{op_name}' is a valid debug operation",
        )

    def _parse_ptr_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse pointer operation (ptoas scene: make_tensor, addptr).

        Args:
            op_name: Name of pointer operation
            call: Call AST node

        Returns:
            IR expression from pointer operation
        """
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)
        if hasattr(ir_op.ptr, op_name):
            op_func = getattr(ir_op.ptr, op_name)
            call_span = self.span_tracker.get_span(call)
            return op_func(*args, **kwargs, span=call_span)

        raise InvalidOperationError(
            f"Unknown ptr operation: {op_name}",
            span=self.span_tracker.get_span(call),
            hint=f"Check if '{op_name}' is a valid ptr operation",
        )

    # Manual ops that share block SSA semantics (no explicit output tile arg).
    # These are routed to _parse_block_op directly.
    _MANUAL_AS_BLOCK_OPS: frozenset[str] = frozenset({
        "make_tile",  # allocation — same IR op as SSA
    })

    _MANUAL_AS_DEBUG_OPS: frozenset[str] = frozenset({
        "assert_",
        "dump_tensor",
        "dump_tile",
        "printf",
        "trap",
    })

    def _parse_manual_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse a manual (non-SSA) operation call: plm.{op_name}(..., dst=tile).

        Manual ops differ from block ops in two ways:
          1. They receive a pre-allocated output tile via ``dst=`` or ``out=`` kwarg.
          2. The dst variable is rebound in scope to the returned SSA value so that
             subsequent reads of that variable see the updated data.

        Args:
            op_name: Name of the manual operation (without ``manual.`` prefix).
            call: Call AST node.

        Returns:
            IR expression for the manual op call.
        """
        span = self.span_tracker.get_span(call)

        # Ops with SSA block semantics — no explicit output tile needed.
        if op_name in self._MANUAL_AS_BLOCK_OPS:
            # Auto-sync: emit forward sync before block ops too
            if self.sync_tracker is not None:
                self._emit_forward_syncs_for_manual_op(op_name, call, span)
            return self._parse_block_op(op_name, call)
        if op_name in self._MANUAL_AS_DEBUG_OPS:
            return self._parse_debug_op(op_name, call)

        # Auto-sync: emit forward sync_src/sync_dst before this op
        if self.sync_tracker is not None:
            self._emit_forward_syncs_for_manual_op(op_name, call, span)

        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)
        # Dispatch to ir_op.manual.<op_name> when a handler exists.
        if hasattr(ir_op.manual, op_name):
            op_func = getattr(ir_op.manual, op_name)
            return op_func(*args, **kwargs, span=span)

        # first args is out, we need push out from first to last when create op
        result_expr = ir.create_op_call(
            f"manual.{op_name}", args[1:] + args[0:1], kwargs, span
        )

        # Do NOT rebind the dst variable in scope.  The tile Var created by
        # make_tile remains the canonical buffer handle for the whole kernel;
        # manual ops (load, add, …) are side-effects on that buffer.  Re-binding
        # to the Call node would cause subsequent uses of the variable to resolve
        # to a Call rather than a Var, breaking GetExprAsCode lookups in the
        # PTO backend (which expects Var nodes for tile arguments).

        return result_expr

    # -- auto-sync helpers ------------------------------------------------

    def _extract_plm_or_block_op_name(self, call: ast.Call) -> str | None:
        """Extract op name from a ``plm.xxx(...)`` call, or None."""
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "plm"
        ):
            return func.attr
        return None

    def _emit_forward_syncs_for_manual_op(
        self, op_name: str, call: ast.Call, span: ir.Span,
    ) -> None:
        """Check for cross-pipeline deps and emit sync_src/sync_dst before a manual op."""
        from pypto.frontend.sync_tracker import (
            _OP_TILE_ACCESS,
            _OP_TO_PIPE,
            emit_sync_pair,
            get_move_pipe,
        )

        # Determine pipeline for this op
        if op_name == "move":
            pipe = self._resolve_move_pipe(call)
        elif op_name in ("store", "store_tile"):
            pipe = self._resolve_store_pipe(call, op_name)
        else:
            pipe = _OP_TO_PIPE.get(op_name)
        if pipe is None:
            return

        # Flush deferred backward waits for this pipe.
        # Backward waits are deferred from loop body-start to the first op
        # on each pipe, enabling pipeline overlap (e.g., load overlaps with
        # previous matmul because M→MTE1 wait is deferred to before move).
        self._flush_pending_backward_waits(pipe, span)

        access = _OP_TILE_ACCESS.get(op_name)
        if access is None:
            return

        # Collect tile names from positional args (may be str or list[str])
        read_raw: list[str | list[str] | None] = [
            self._extract_tile_name_from_ast(call, i) for i in access.read_indices
        ]
        write_raw: list[str | list[str] | None] = [
            self._extract_tile_name_from_ast(call, i) for i in access.write_indices
        ]

        has_db = any(isinstance(n, list) for n in read_raw + write_raw)

        assert self.sync_tracker is not None

        if not has_db:
            # Non-DB path (existing logic)
            read_names = [n for n in read_raw if isinstance(n, str)]
            write_names = [n for n in write_raw if isinstance(n, str)]
            pairs = self.sync_tracker.record_op(pipe, read_names, write_names)
            for pair in pairs:
                emit_sync_pair(self.builder, pair, span)
        else:
            # DB path: per-slot analysis + if-else sync chain
            self._emit_db_forward_syncs(pipe, read_raw, write_raw, call, span)

    def _emit_db_forward_syncs(
        self,
        pipe: PipeType,
        read_raw: list[str | list[str] | None],
        write_raw: list[str | list[str] | None],
        call: ast.Call,
        span: ir.Span,
    ) -> None:
        """Emit forward sync for double-buffer tile tuples.

        Analyzes dependencies per buffer slot independently, then emits an
        if-else chain per (set_pipe, wait_pipe) pair so the runtime selects
        the correct per-slot event_id.
        """
        import copy
        from pypto.frontend.sync_tracker import SyncPair

        assert self.sync_tracker is not None
        tracker = self.sync_tracker

        # Determine n_slots from the first list[str] encountered
        n_slots = 2
        for n in read_raw + write_raw:
            if isinstance(n, list):
                n_slots = len(n)
                break

        # Extract the buf_idx IR expression from the first Subscript arg
        index_expr = self._extract_db_index_expr(call)
        if index_expr is None:
            return  # cannot resolve index; skip (conservative: no sync)

        # Per-slot dependency analysis with state snapshots
        saved_states = copy.deepcopy(tracker._buffer_states)
        slot_pairs: list[list[SyncPair]] = []

        for slot in range(n_slots):
            # Restore to same starting state for each slot
            tracker._buffer_states = copy.deepcopy(saved_states)
            read_names = [self._resolve_slot_name(n, slot) for n in read_raw]
            write_names = [self._resolve_slot_name(n, slot) for n in write_raw]
            read_names = [n for n in read_names if n is not None]
            write_names = [n for n in write_names if n is not None]
            pairs = tracker.record_op(pipe, read_names, write_names)
            slot_pairs.append(pairs)

        # Restore and update state: union of all slots' final state
        # After the loop, buffer states from the last slot's record_op are
        # in tracker._buffer_states.  We need to merge all slots' tile states.
        merged = copy.deepcopy(saved_states)
        for slot in range(n_slots):
            # Re-run with correct starting state to get final per-slot state
            tracker._buffer_states = copy.deepcopy(saved_states)
            read_names = [self._resolve_slot_name(n, slot) for n in read_raw]
            write_names = [self._resolve_slot_name(n, slot) for n in write_raw]
            read_names = [n for n in read_names if n is not None]
            write_names = [n for n in write_names if n is not None]
            # Just update state (ignore returned pairs — we already have them)
            tracker.record_op(pipe, read_names, write_names)
            for tile_name, state in tracker._buffer_states.items():
                if tile_name not in saved_states or state != saved_states.get(tile_name):
                    merged[tile_name] = copy.deepcopy(state)
        tracker._buffer_states = merged

        # Collect unique (set_pipe, wait_pipe) pairs across all slots
        all_pipe_keys: list[tuple[PipeType, PipeType]] = []
        seen: set[tuple[PipeType, PipeType]] = set()
        for pairs in slot_pairs:
            for p in pairs:
                key = (p.set_pipe, p.wait_pipe)
                if key not in seen:
                    seen.add(key)
                    all_pipe_keys.append(key)

        # Emit if-else chain for each unique pipe pair
        from pypto.ir.op import system_ops as ir_sys_ops
        for set_p, wait_p in all_pipe_keys:
            # Collect per-slot event IDs
            slot_event_ids: list[int] = []
            for slot, pairs in enumerate(slot_pairs):
                matching = [p for p in pairs if p.set_pipe == set_p and p.wait_pipe == wait_p]
                if matching:
                    base_eid = tracker._event_allocator.forward_event_id(set_p, wait_p, n_slots=n_slots)
                    slot_event_ids.append((base_eid + slot) % tracker._event_allocator.MAX_EVENTS)
                else:
                    slot_event_ids.append(-1)  # sentinel: this slot has no dep

            # Build if-else chain emitting sync_src + sync_dst per slot
            self._build_db_sync_chain(set_p, wait_p, slot_event_ids, index_expr, 0, span)

    def _extract_db_index_expr(self, call: ast.Call) -> ir.Expr | None:
        """Extract the buffer-index IR expression from the first Subscript arg."""
        for arg in call.args:
            if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
                if arg.value.id in self._tile_tuple_registry:
                    return self.parse_expression(arg.slice)
        return None

    @staticmethod
    def _resolve_slot_name(raw: str | list[str] | None, slot: int) -> str | None:
        """Resolve a raw tile name to a specific slot's tile name."""
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw
        # list[str] from tile tuple: pick the slot-th element
        if slot < len(raw):
            return raw[slot]
        return None

    def _build_db_sync_chain(
        self,
        set_pipe: PipeType,
        wait_pipe: PipeType,
        slot_event_ids: list[int],
        index_expr: ir.Expr,
        level: int,
        span: ir.Span,
    ) -> None:
        """Recursively build if-else chain emitting sync_src+sync_dst per slot.

        Generates IR like:
            if buf_idx == 0:
                sync_src(set_pipe, wait_pipe, event_id=slot_event_ids[0])
                sync_dst(set_pipe, wait_pipe, event_id=slot_event_ids[0])
            else:
                if buf_idx == 1:
                    sync_src(set_pipe, wait_pipe, event_id=slot_event_ids[1])
                    sync_dst(set_pipe, wait_pipe, event_id=slot_event_ids[1])
                else: ...
        """
        from pypto.frontend.sync_tracker import emit_sync_pair
        from pypto.frontend.sync_tracker.data_structures import SyncPair

        n = len(slot_event_ids)
        eid = slot_event_ids[level]

        if level == n - 1:
            # Leaf: emit unconditionally
            if eid >= 0:
                pair = SyncPair(set_pipe, wait_pipe, "raw", eid)
                emit_sync_pair(self.builder, pair, span)
            return

        if eid < 0:
            # This slot has no dependency — skip to next
            cond = index_expr == level
            with self.builder.if_stmt(cond, span) as if_b:
                pass  # empty then-branch
                if_b.else_()
                self._build_db_sync_chain(set_pipe, wait_pipe, slot_event_ids, index_expr, level + 1, span)
            return

        cond = index_expr == level
        with self.builder.if_stmt(cond, span) as if_b:
            pair = SyncPair(set_pipe, wait_pipe, "raw", eid)
            emit_sync_pair(self.builder, pair, span)
            if_b.else_()
            self._build_db_sync_chain(set_pipe, wait_pipe, slot_event_ids, index_expr, level + 1, span)

    def _build_db_slot_expr(
        self, loop_var: ir.Var, step: ir.Expr, n_slots: int, span: ir.Span,
    ) -> ir.Expr:
        """Build ``(loop_var / step) % n_slots`` as an IR expression."""
        div_expr = loop_var // step
        mod_expr = div_expr % n_slots
        return mod_expr

    def _flush_pending_backward_waits(self, pipe: PipeType, span: ir.Span) -> None:
        """Emit deferred backward waits for *pipe* and remove them from pending.

        Called by ``_emit_forward_syncs_for_manual_op`` right before an op on
        *pipe* is processed, so that the backward wait happens at the correct
        time — e.g., ``M→MTE1`` wait is emitted before the first move (MTE1),
        not before the first load (MTE2).
        """
        pending = getattr(self, "_pending_backward_waits", None)
        if pending is None or pipe not in pending:
            return
        from pypto.frontend.sync_tracker import emit_backward_sync_dst
        slot_expr = getattr(self, "_pending_backward_wait_slot_expr", None)
        for dep in pending.pop(pipe):
            if dep.n_slots > 1 and slot_expr is not None:
                self._emit_backward_db_sync_chain(dep, slot_expr, emit_backward_sync_dst, span)
            else:
                emit_backward_sync_dst(self.builder, dep, span)

    def _flush_all_pending_backward_waits(self, span: ir.Span) -> None:
        """Emit ALL remaining deferred backward waits.

        Called before entering a nested for-loop to ensure outer-loop backward
        waits are emitted before the inner loop overwrites ``_pending_backward_waits``.
        """
        pending = getattr(self, "_pending_backward_waits", None)
        if not pending:
            return
        from pypto.frontend.sync_tracker import emit_backward_sync_dst
        slot_expr = getattr(self, "_pending_backward_wait_slot_expr", None)
        for pipe in list(pending.keys()):
            for dep in pending.pop(pipe):
                if dep.n_slots > 1 and slot_expr is not None:
                    self._emit_backward_db_sync_chain(dep, slot_expr, emit_backward_sync_dst, span)
                else:
                    emit_backward_sync_dst(self.builder, dep, span)

    def _emit_backward_db_sync_chain(
        self,
        dep: "BackwardDep",
        slot_expr: ir.Expr,
        emit_fn: "Callable",
        span: ir.Span,
    ) -> None:
        """Emit per-slot backward sync via if-else chain on *slot_expr*."""
        self._build_backward_db_chain(dep, slot_expr, emit_fn, 0, span)

    def _build_backward_db_chain(
        self,
        dep: "BackwardDep",
        slot_expr: ir.Expr,
        emit_fn: "Callable",
        level: int,
        span: ir.Span,
    ) -> None:
        from pypto.frontend.sync_tracker.data_structures import BackwardDep as BD
        n = dep.n_slots
        eid = (dep.event_id + level) % 8
        slot_dep = BD(dep.first_pipe, dep.last_pipe, dep.tile_name, eid, dep.loop_depth)

        if level == n - 1:
            emit_fn(self.builder, slot_dep, span)
            return

        cond = slot_expr == level
        with self.builder.if_stmt(cond, span) as if_b:
            emit_fn(self.builder, slot_dep, span)
            if_b.else_()
            self._build_backward_db_chain(dep, slot_expr, emit_fn, level + 1, span)

    def _extract_tile_name_from_ast(self, call: ast.Call, idx: int) -> str | list[str] | None:
        """Extract tile variable name(s) from a positional arg at *idx*.

        Returns:
            ``str`` for a simple tile variable (e.g. ``tile_a``).
            ``list[str]`` for a tile-tuple subscript (e.g. ``tile_buf[buf_idx]``),
            containing all tile names in the tuple.
            ``None`` if the arg cannot be resolved to a tile.
        """
        if idx >= len(call.args):
            return None
        arg = call.args[idx]

        # Path 1: simple variable name → single tile
        if isinstance(arg, ast.Name):
            var = self.scope_manager.lookup_var(arg.id)
            if var is None:
                return None
            var_type = getattr(var, "type", None)
            if var_type is None or not isinstance(var_type, ir.TileType):
                return None
            return arg.id

        # Path 2: tile_buf[buf_idx] subscript → all tiles in the tuple
        if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
            tuple_name = arg.value.id
            if tuple_name in self._tile_tuple_registry:
                return self._tile_tuple_registry[tuple_name]

        return None

    def _resolve_move_pipe(self, call: ast.Call) -> PipeType:
        """Resolve the pipeline for a ``move`` op.

        DSL signature: ``plm.move(out, tile)`` where arg0=out (target),
        arg1=tile (source).  The pipe is determined by the source and target
        memory spaces (e.g. Mat→Left = MTE1).
        """
        from pypto.frontend.sync_tracker import get_move_pipe

        src_memory: MemorySpace | None = None
        target_memory: MemorySpace | None = None
        for kw in call.keywords:
            if kw.arg == "src_memory" and isinstance(kw.value, ast.Attribute):
                src_memory = _MEMORY_SPACE_MAP.get(kw.value.attr)
            elif kw.arg == "target_memory" and isinstance(kw.value, ast.Attribute):
                target_memory = _MEMORY_SPACE_MAP.get(kw.value.attr)
        # Resolve target_memory from arg0 (out tile)
        if target_memory is None and len(call.args) >= 1:
            target_memory = self._resolve_tile_arg_memory_space(call.args[0])
        # Resolve src_memory from arg1 (source tile)
        if src_memory is None and len(call.args) >= 2:
            src_memory = self._resolve_tile_arg_memory_space(call.args[1])
        return get_move_pipe(src_memory, target_memory)

    def _resolve_store_pipe(self, call: ast.Call, op_name: str) -> PipeType:
        """Resolve the pipeline for a ``store`` / ``store_tile`` op.

        PTOAS rules:
        - Store from Vec (UB) → PIPE_MTE3 (TSTORE_VEC)
        - Store from Acc (L0C) → PIPE_FIX (TSTORE_ACC)

        The source tile is DSL arg[1]: ``plm.store(tensor, tile, ...)``.
        """
        from pypto.frontend.sync_tracker import get_store_pipe

        # DSL convention: arg[1] is the source tile
        src_memory: MemorySpace | None = None
        if 1 < len(call.args):
            src_memory = self._resolve_tile_arg_memory_space(call.args[1])
        return get_store_pipe(src_memory)

    def _resolve_tile_arg_memory_space(self, arg: ast.expr) -> MemorySpace | None:
        """Resolve the memory space of a tile argument (Name or Subscript).

        For ``ast.Name``: looks up the variable in scope and reads its
        ``type.memref.memory_space_``.
        For ``ast.Subscript`` (e.g. ``tile_buf[buf_idx]``): looks up the first
        tile in the tuple from ``_tile_tuple_registry`` — all tiles in a tuple
        share the same TileType, so any element gives the correct memory space.
        """
        def _get_memory_space_from_var(var_name: str) -> MemorySpace | None:
            var = self.scope_manager.lookup_var(var_name)
            if var is None:
                return None
            var_type = getattr(var, "type", None)
            if var_type is None:
                return None
            # TileType stores memory space in memref.memory_space_
            memref = getattr(var_type, "memref", None)
            if memref is not None:
                return getattr(memref, "memory_space_", None)
            # Fallback: direct memory_space attribute
            return getattr(var_type, "memory_space", None)

        # Path 1: simple variable name
        if isinstance(arg, ast.Name):
            return _get_memory_space_from_var(arg.id)

        # Path 2: tile_buf[buf_idx] — resolve from first tuple element
        if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
            tuple_name = arg.value.id
            if tuple_name in self._tile_tuple_registry:
                first_tile_name = self._tile_tuple_registry[tuple_name][0]
                return _get_memory_space_from_var(first_tile_name)

        return None

    def _register_tile_region(self, var_name: str, var: ir.Var) -> None:
        """Extract MemRef from a tile Var and register with sync tracker."""
        from pypto.frontend.sync_tracker import TileRegion

        var_type = var.type
        if not isinstance(var_type, ir.TileType):
            return
        memref = var_type.memref
        if memref is None:
            return
        addr_offset: int | None = None
        if isinstance(memref.addr_, ir.ConstInt):
            addr_offset = memref.addr_.value
        region = TileRegion(
            memory_space=memref.memory_space_,
            addr_offset=addr_offset,
            byte_size=memref.size_,
        )
        assert self.sync_tracker is not None
        self.sync_tracker.register_tile(var_name, region)

    def _verify_backward_deps(
        self,
        prescan_deps: list,
        loop_ctx: object,
    ) -> None:
        """Warn if prescan backward deps differ from actual loop body access."""
        import warnings
        from pypto.frontend.sync_tracker import LoopContext

        if not isinstance(loop_ctx, LoopContext):
            return
        actual_deps: set[tuple] = set()
        for tile_name in loop_ctx.first_access:
            first = loop_ctx.first_access[tile_name]
            last = loop_ctx.last_access.get(tile_name, first)
            if first != last:
                actual_deps.add((first, last, tile_name))

        prescan_set = {(d.first_pipe, d.last_pipe, d.tile_name) for d in prescan_deps}
        missed = actual_deps - prescan_set
        if missed:
            for first, last, name in missed:
                warnings.warn(
                    f"Auto-sync: prescan missed backward dep for tile '{name}' "
                    f"(first={first.name}, last={last.name}). "
                    f"Consider adding manual sync.",
                    stacklevel=2,
                )

    # Maps unified op names to the scalar variant for block ops.
    # Only binary arithmetic ops have scalar auto-dispatch.
    _BLOCK_SCALAR_OPS: dict[str, str] = {
        "add": "adds",
        "sub": "subs",
        "mul": "muls",
        "div": "divs",
    }

    # Maps unified op names to ir scalar expression functions.
    _SCALAR_BINARY_OPS: dict[str, str] = {
        "min": "min_",
        "max": "max_",
    }

    _SCALAR_UNARY_OPS: dict[str, str] = {}

    _SCALAR_BLOCK_OPS = {"index_cast"}

    # Maps language-level tensor operation names to IR-level names.
    _TENSOR_OP_NAME_MAP: dict[str, str] = {
        "create_tensor": "create",
    }

    # Ops that exist only in one module (no dispatch needed).
    _TENSOR_ONLY_OPS = {
        "create_tensor",
        "dim",
        "assemble",
        "add_scalar",
        "sub_scalar",
        "mul_scalar",
        "div_scalar",
    }

    # Ops that only exist in the ptr module (ptoas scene).
    _PTR_ONLY_OPS = {"make_tensor", "addptr"}
    _BLOCK_ONLY_OPS = {
        "load",
        "store",
        "move",
        "neg",
        "sqrt",
        "rsqrt",
        "recip",
        "log",
        "relu",
        "matmul_acc",
        "minimum",
        "cmp",
        "cmps",
        "adds",
        "subs",
        "muls",
        "divs",
        "sum",
        "row_min",
        "row_expand",
        "row_expand_add",
        "row_expand_sub",
        "row_expand_mul",
        "row_expand_div",
        "col_expand",
        "col_expand_mul",
        "col_expand_div",
        "col_expand_sub",
        "col_max",
        "col_sum",
        "expands",
        "matmul_bias",
        "gemv",
        "gemv_acc",
        "gemv_bias",
        "abs",
        "make_tile",
    }

    def _parse_unified_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse unified operation call (pl.{op_name}).

        Dispatches to tensor or block IR op based on the first argument's type.

        Args:
            op_name: Name of the operation
            call: Call AST node

        Returns:
            IR expression from the dispatched operation
        """
        # Short-circuit for ops that only exist in one module
        if op_name in self._PTR_ONLY_OPS:
            return self._parse_ptr_op(op_name, call)
        if op_name in self._TENSOR_ONLY_OPS:
            return self._parse_tensor_op(op_name, call)
        if op_name in self._BLOCK_ONLY_OPS:
            return self._parse_block_op(op_name, call)

        call_span = self.span_tracker.get_span(call)

        if not call.args:
            raise InvalidOperationError(
                f"Unified operation '{op_name}' requires at least one argument for type dispatch",
                span=call_span,
                hint="Provide a Tensor or Tile as the first argument",
            )

        # Parse only the first arg to determine dispatch target
        first_arg = self.parse_expression(call.args[0])
        first_type = first_arg.type

        if isinstance(first_type, ir.TensorType):
            return self._parse_tensor_op(op_name, call)

        if isinstance(first_type, ir.TileType):
            # For binary arithmetic ops, check if rhs is scalar → use scalar variant
            scalar_op = self._BLOCK_SCALAR_OPS.get(op_name)
            if scalar_op and len(call.args) >= 2:
                rhs_arg = self.parse_expression(call.args[1])
                if isinstance(rhs_arg.type, ir.ScalarType):
                    return self._parse_block_op(scalar_op, call)

            return self._parse_block_op(op_name, call)

        if isinstance(first_type, ir.ScalarType):
            return self._parse_scalar_op(op_name, call, call_span)

        raise InvalidOperationError(
            f"Cannot dispatch '{op_name}': first argument has type {type(first_type).__name__}, "
            f"expected TensorType, TileType, or ScalarType",
            span=call_span,
            hint="Use pl.tensor.* or pl.block.* for explicit dispatch; for pointer ops use pl.make_tensor or pl.addptr",
        )

    def _parse_typed_constant(self, call: ast.Call) -> ir.Expr:
        """Parse pl.const(value, dtype) → ConstInt or ConstFloat.

        Args:
            call: Call AST node for pl.const(value, dtype)

        Returns:
            ConstInt or ConstFloat with the specified dtype
        """
        span = self.span_tracker.get_span(call)

        if len(call.args) != 2:
            raise ParserSyntaxError(
                "pl.const() requires exactly 2 arguments: value and dtype",
                span=span,
                hint="Use pl.const(42, pl.INT32) or pl.const(1.0, pl.FP16)",
            )

        # Extract numeric value from first argument (handles Constant and -Constant)
        value_node = call.args[0]
        negate = False
        if isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.USub):
            negate = True
            value_node = value_node.operand

        if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, (int, float)):
            raise ParserSyntaxError(
                "pl.const() first argument must be a numeric literal",
                span=span,
                hint="Use an int or float literal: pl.const(42, pl.INT32)",
            )

        value = value_node.value
        if negate:
            value = -value

        # Resolve dtype from second argument
        dtype = self.type_resolver.resolve_dtype(call.args[1])

        if isinstance(value, float):
            return ir.ConstFloat(value, dtype, span)
        else:
            return ir.ConstInt(value, dtype, span)

    def _parse_scalar_op(self, op_name: str, call: ast.Call, call_span: ir.Span) -> ir.Expr:
        """Parse scalar operation (e.g. pl.min(s1, s2) where s1, s2 are scalars).

        Args:
            op_name: Name of the operation
            call: Call AST node
            call_span: Source span for error reporting

        Returns:
            IR scalar expression
        """
        if call.keywords:
            raise InvalidOperationError(
                f"Scalar operation '{op_name}' does not accept keyword arguments",
                span=call_span,
            )

        if op_name in self._SCALAR_BINARY_OPS:
            if len(call.args) != 2:
                raise InvalidOperationError(
                    f"Scalar binary operation '{op_name}' requires exactly 2 arguments, got {len(call.args)}",
                    span=call_span,
                )
            lhs = self.parse_expression(call.args[0])
            rhs = self.parse_expression(call.args[1])
            ir_func_name = self._SCALAR_BINARY_OPS[op_name]
            ir_func = getattr(ir, ir_func_name)
            return ir_func(lhs, rhs, call_span)

        if op_name in self._SCALAR_UNARY_OPS:
            if len(call.args) != 1:
                raise InvalidOperationError(
                    f"Scalar unary operation '{op_name}' requires exactly 1 argument, got {len(call.args)}",
                    span=call_span,
                )
            arg = self.parse_expression(call.args[0])
            ir_func_name = self._SCALAR_UNARY_OPS[op_name]
            ir_func = getattr(ir, ir_func_name)
            return ir_func(arg, call_span)

        if op_name in self._SCALAR_BLOCK_OPS:
            if len(call.args) != 1:
                raise InvalidOperationError(
                    f"Scalar block operation '{op_name}' requires exactly 1 argument, got {len(call.args)}",
                    span=call_span,
                )
            arg = self.parse_expression(call.args[0])
            ir_func = getattr(ir_op.block, op_name)
            return ir_func(arg, call_span)

        raise InvalidOperationError(
            f"Operation '{op_name}' is not supported for scalar arguments",
            span=call_span,
            hint="Supported scalar ops: min, max, index_cast",
        )

    def parse_attribute(self, attr: ast.Attribute) -> ir.Expr:
        """Parse attribute access.

        Args:
            attr: Attribute AST node

        Returns:
            IR expression
        """
        span = self.span_tracker.get_span(attr)
        if isinstance(attr.value, ast.Name):
            obj_name = attr.value.id
            field_name = attr.attr

            # Check struct vars (python var or regular scope var)
            obj = self.scope_manager.get_python_var(obj_name)
            if obj is None:
                obj = self.scope_manager.lookup_var(obj_name)
            if isinstance(obj, _StructVar):
                if field_name not in obj.fields:
                    raise ParserTypeError(
                        f"Struct '{obj_name}' has no field '{field_name}'",
                        span=span,
                        hint=f"Available fields: {', '.join(obj.fields.keys())}",
                    )
                if obj.name:
                    # Named struct with C++ codegen — use struct.get
                    idx_zero = ir.ConstInt(0, DataType.INDEX, span)
                    return ir.create_op_call(
                        "struct.get", [idx_zero],
                        {"array": obj.name, "field": field_name}, span,
                    )
                return obj.fields[field_name]
            # _DynamicStructView: struct array view passed as function argument
            if isinstance(obj, _DynamicStructView):
                if field_name not in obj.array.field_names:
                    raise ParserTypeError(
                        f"Struct array view has no field '{field_name}'",
                        span=span,
                        hint=f"Available fields: {', '.join(obj.array.field_names)}",
                    )
                return self._struct_array_field_read(obj, field_name, span)
            if obj_name in self.tiling_registry:
                field_vars = self.tiling_registry[obj_name]
                if field_name in field_vars:
                    val = field_vars[field_name]
                    if isinstance(val, list):
                        raise ParserTypeError(
                            f"Array field '{field_name}' must be accessed with an integer index",
                            span=span,
                            hint=f"Use {obj_name}.{field_name}[0] through "
                                         f"{obj_name}.{field_name}[{len(val) - 1}]",
                        )
                    return val  # scalar ir.Var
                raise ParserTypeError(
                    f"Tiling parameter '{obj_name}' has no field '{field_name}'",
                    span=span,
                    hint=f"Valid fields are: {', '.join(field_vars.keys())}",
                )
            # Check for pl.MemorySpace.* attribute access (nested: pl.MemorySpace.Left)
            if obj_name == "pl" and field_name in _MEMORY_SPACE_MAP:
                return ir.ConstInt(_MEMORY_SPACE_MAP[field_name].value, DataType.INT64, span)
        # Check for nested attribute access like pl.MemorySpace.Left
        if isinstance(attr.value, ast.Attribute):
            inner_attr = attr.value
            if isinstance(inner_attr.value, ast.Name):
                inner_obj_name = inner_attr.value.id
                inner_field_name = inner_attr.attr
                outer_field_name = attr.attr
                # Handle pl.MemorySpace.Left, pl.MemorySpace.Right, etc.
                if inner_obj_name == "pl" and inner_field_name == "MemorySpace":
                    if outer_field_name in _MEMORY_SPACE_MAP:
                        return ir.ConstInt(_MEMORY_SPACE_MAP[outer_field_name].value, DataType.INT64, span)
        # Check for struct_array[idx].field compound pattern
        if isinstance(attr.value, ast.Subscript) and isinstance(attr.value.value, ast.Name):
            arr_name = attr.value.value.id
            arr_obj = self.scope_manager.get_python_var(arr_name)
            if arr_obj is None:
                arr_obj = self.scope_manager.lookup_var(arr_name)
            if isinstance(arr_obj, _StructArrayVar):
                field_name = attr.attr
                if field_name not in arr_obj.field_names:
                    raise ParserTypeError(
                        f"Struct array '{arr_name}' has no field '{field_name}'",
                        span=span,
                        hint=f"Available fields: {', '.join(arr_obj.field_names)}",
                    )
                index_expr = self.parse_expression(attr.value.slice)
                view = _DynamicStructView(arr_obj, index_expr)
                return self._struct_array_field_read(view, field_name, span)
        raise UnsupportedFeatureError(
            f"Standalone attribute access not supported: {ast.unparse(attr)}",
            span=span,
            hint="Attribute access is only supported for tiling parameters (e.g., tiling.x) "
                         "or within function calls",
        )

    def parse_list(self, list_node: ast.List) -> ir.MakeTuple:
        """Parse list literal into MakeTuple IR expression.

        Args:
            list_node: List AST node

        Returns:
            MakeTuple IR expression
        """
        span = self.span_tracker.get_span(list_node)
        elements = [self.parse_expression(elt) for elt in list_node.elts]
        return ir.MakeTuple(elements, span)

    def parse_tuple_literal(self, tuple_node: ast.Tuple) -> ir.MakeTuple:
        """Parse tuple literal like (x, y, z).

        Args:
            tuple_node: Tuple AST node

        Returns:
            MakeTuple IR expression
        """
        span = self.span_tracker.get_span(tuple_node)
        elements = [self.parse_expression(elt) for elt in tuple_node.elts]
        return ir.MakeTuple(elements, span)

    def _struct_array_field_read(
        self,
        view: "_DynamicStructView",
        field_name: str,
        span: "ir.Span",
    ) -> ir.Expr:
        """Read a field from a struct array at a dynamic index.

        If the view has a ref_name (from ``ctx = ctx_arr[idx]``), the codegen
        uses the C++ reference (``ctx.field``) instead of ``ctx_arr[idx].field``.
        """
        kwargs: dict[str, Any] = {"array": view.array.name, "field": field_name}
        if view.ref_name:
            kwargs["ref"] = view.ref_name
        return ir.create_op_call("struct.get", [view.index_expr], kwargs, span)

    def _struct_array_field_write(
        self,
        arr: "_StructArrayVar",
        index_expr: ir.Expr,
        field_name: str,
        value_expr: ir.Expr,
        span: "ir.Span",
        ref_name: str = "",
    ) -> None:
        """Write a field to one slot of a struct array at a dynamic index.

        If ref_name is set, the codegen uses the C++ reference (``ctx.field = val``)
        instead of ``ctx_arr[idx].field = val``.
        """
        kwargs: dict[str, Any] = {"array": arr.name, "field": field_name}
        if ref_name:
            kwargs["ref"] = ref_name
        call = ir.create_op_call("struct.set", [index_expr, value_expr], kwargs, span)
        self.builder.emit(ir.EvalStmt(call, span))

    def _build_tuple_index_chain(
        self,
        value_expr: ir.Expr,
        index_expr: ir.Expr,
        elem_type: ir.Type,
        n: int,
        level: int,
        span: ir.Span,
    ) -> ir.Var | None:
        """Recursively build nested if-else chain for variable tuple index access.

        Lowers `tuple[idx]` into a nested if-else structure at the IR level:
          if idx == 0: yield tuple[0]
          else:
            if idx == 1: yield tuple[1]
            else: yield tuple[2]  # leaf

        Args:
            value_expr: The tuple expression being indexed
            index_expr: Variable index expression
            elem_type: Type of each tuple element (must be homogeneous)
            n: Total number of tuple elements
            level: Current element index being tested (0-based)
            span: Source span for IR nodes

        Returns:
            The phi var from the outermost IfStmt, or None for the leaf case.
        """
        if level == n - 1:
            # Leaf: emit yield unconditionally (already in innermost else branch)
            self.builder.emit(ir.YieldStmt([ir.TupleGetItemExpr(value_expr, level, span)], span))
            return None

        cond = index_expr == level  # Expr.__eq__ produces a comparison Expr
        with self.builder.if_stmt(cond, span) as if_b:
            # Then branch: yield the element at this level
            self.builder.emit(ir.YieldStmt([ir.TupleGetItemExpr(value_expr, level, span)], span))
            if_b.else_()
            # Else branch: recurse to the next level
            inner_result = self._build_tuple_index_chain(
                value_expr, index_expr, elem_type, n, level + 1, span
            )
            if inner_result is not None:
                # Forward the inner phi var as this branch's yield
                self.builder.emit(ir.YieldStmt([inner_result], span))
            # Declare phi variable AFTER both branches, still inside the with block
            result_name = f"_tidx_{self._tuple_idx_counter}"
            self._tuple_idx_counter += 1
            if_b.return_var(result_name, elem_type, span)
        return if_b.output(0)

    def parse_subscript(self, subscript: ast.Subscript) -> ir.Expr:
        """Parse subscript expression like tuple[0].

        Args:
            subscript: Subscript AST node

        Returns:
            IR expression (TupleGetItemExpr for tuple access)

        Example Python syntax:
            first = my_tuple[0]      # Creates TupleGetItemExpr(my_tuple, 0)
            nested = my_tuple[1][2]  # Creates nested TupleGetItemExpr
        """
        span = self.span_tracker.get_span(subscript)

        # Check for tiling array field access: tiling.arr[i]
        if isinstance(subscript.value, ast.Attribute):
            attr = subscript.value
            if (isinstance(attr.value, ast.Name)
                    and attr.value.id in self.tiling_registry):
                obj_name = attr.value.id
                field_name = attr.attr
                field_val = self.tiling_registry[obj_name].get(field_name)
                if isinstance(field_val, list):
                    if (not isinstance(subscript.slice, ast.Constant)
                            or not isinstance(subscript.slice.value, int)):
                        raise UnsupportedFeatureError(
                            "Tiling array fields only support literal integer indices",
                            span=span,
                            hint=f"Use a constant index like tiling.{field_name}[0]",
                        )
                    idx = subscript.slice.value
                    if idx < 0 or idx >= len(field_val):
                        raise ParserTypeError(
                            f"Index {idx} out of bounds for array field '{field_name}' "
                            f"(size {len(field_val)})",
                            span=span,
                            hint=f"Valid indices are 0 to {len(field_val) - 1}",
                        )
                    return field_val[idx]
                elif field_val is not None:
                    # Scalar field accessed with subscript — helpful error
                    raise ParserTypeError(
                        f"Scalar field '{field_name}' does not support subscript access",
                        span=span,
                        hint=f"Use tiling.{field_name} directly (no index needed)",
                    )

        # Check for struct array subscript: ctx_arr[idx] → _DynamicStructView
        if isinstance(subscript.value, ast.Name):
            _sa_obj = self.scope_manager.get_python_var(subscript.value.id)
            if _sa_obj is None:
                _sa_obj = self.scope_manager.lookup_var(subscript.value.id)
            if isinstance(_sa_obj, _StructArrayVar):
                index_expr = self.parse_expression(subscript.slice)
                return _DynamicStructView(_sa_obj, index_expr)

        value_expr = self.parse_expression(subscript.value)

        # Parse index from slice
        if isinstance(subscript.slice, ast.Constant):
            index = subscript.slice.value
            if not isinstance(index, int):
                raise ParserSyntaxError(
                    "Tuple index must be an integer",
                    span=span,
                    hint="Use integer index like tuple[0]",
                )
        else:
            # Variable index: parse as IR expression and lower to an if-else chain
            index_expr = self.parse_expression(subscript.slice)

            value_type = value_expr.type
            if not isinstance(value_type, ir.TupleType):
                raise ParserTypeError(
                    f"Subscript requires tuple type, got {type(value_type).__name__}",
                    span=span,
                    hint="Only tuple types support subscript access in this context",
                )

            elem_types = list(value_type.types)
            if not elem_types:
                raise ParserTypeError(
                    "Cannot index into empty tuple",
                    span=span,
                )

            # Variable indexing requires all elements to share the same type
            first_type = elem_types[0]
            for i, t in enumerate(elem_types[1:], 1):
                if not ir.structural_equal(t, first_type, enable_auto_mapping=False):
                    raise ParserTypeError(
                        f"Variable tuple index requires all elements to have the same type, "
                        f"but element 0 has type {first_type} and element {i} has type {t}",
                        span=span,
                        hint="Use a constant index to access elements of different types",
                    )

            # Cache lookup: same (tuple_var_name, index_ssa_var_name) → reuse existing phi var.
            # Applies to all tuple element types (tile, tensor, event ID, etc.).
            cache_key: tuple[str, str] | None = None
            if isinstance(subscript.value, ast.Name) and isinstance(index_expr, ir.Var):
                cache_key = (subscript.value.id, index_expr.name)
                if cache_key in self._tuple_select_cache:
                    return self._tuple_select_cache[cache_key]

            result = self._build_tuple_index_chain(
                value_expr, index_expr, first_type, len(elem_types), 0, span
            )
            if result is None:
                # Single-element tuple: leaf emits directly, return TupleGetItemExpr
                return ir.TupleGetItemExpr(value_expr, 0, span)

            # Store in cache for subsequent uses of the same buf[idx]
            if cache_key is not None:
                self._tuple_select_cache[cache_key] = result
            return result

        # Check if value is tuple type (runtime check)
        value_type = value_expr.type
        if not isinstance(value_type, ir.TupleType):
            raise ParserTypeError(
                f"Subscript requires tuple type, got {type(value_type).__name__}",
                span=span,
                hint="Only tuple types support subscript access in this context",
            )

        # Create TupleGetItemExpr
        return ir.TupleGetItemExpr(value_expr, index, span)

    def _resolve_yield_var_type(self, annotation: ast.expr | None) -> ir.Type:
        """Resolve type annotation for a yield variable.

        Args:
            annotation: Type annotation AST node, or None if not annotated

        Returns:
            Resolved IR type
        """
        if annotation is None:
            # Fallback to generic tensor type when no annotation present
            return ir.TensorType([1], DataType.INT32)

        resolved = self.type_resolver.resolve_type(annotation)
        # resolve_type can return list[Type] for tuple[...] annotations
        if isinstance(resolved, list):
            if len(resolved) == 0:
                # Empty tuple type - use fallback
                return ir.TensorType([1], DataType.INT32)
            if len(resolved) == 1:
                # Single element - unwrap
                return resolved[0]
            # Multiple elements - create TupleType
            return ir.TupleType(resolved)
        # Single type
        return resolved

    def _scan_for_yields(self, stmts: list[ast.stmt]) -> list[tuple[str, ast.expr | None]]:
        """Scan statements for yield assignments to determine output variable names and types.

        Args:
            stmts: List of statements to scan

        Returns:
            List of tuples (variable_name, type_annotation) where type_annotation is None if not annotated
        """
        yield_vars = []

        for stmt in stmts:
            # Check for annotated assignment with yield_: var: type = pl.yield_(...)
            if isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        yield_vars.append((stmt.target.id, stmt.annotation))

            # Check for regular assignment with yield_: var = pl.yield_(...)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    # Single variable assignment
                    if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yield_":
                            yield_vars.append((target.id, None))
                    # Tuple unpacking: (a, b) = pl.yield_(...)
                    elif isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yield_":
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    yield_vars.append((elt.id, None))

            # Recursively scan nested if statements
            elif isinstance(stmt, ast.If):
                yield_vars.extend(self._scan_for_yields(stmt.body))
                if stmt.orelse:
                    # Only take yields from else if they match then branch
                    # For simplicity, just take from then branch
                    pass

        return yield_vars


__all__ = ["ASTParser"]
