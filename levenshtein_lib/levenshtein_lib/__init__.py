import polars as pl
from polars.type_aliases import IntoExpr
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)

__version__ = "0.1.0"

@pl.api.register_expr_namespace("dist")
class LevenshteinDistance:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def get_levenshtein_ratio(self, word: IntoExpr) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            args=[word],
            symbol="get_levenshtein_ratio",
            is_elementwise=True,
        )
