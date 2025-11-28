import functools
from typing import Any, Sequence, Iterable
from types import EllipsisType

# TODO
import numpy
import numpy.typing
import bracex
import wcmatch.fnmatch


# TODO allow ellipsis??
class PathExpression:
    __slots__ = ["_expr"]

    # _expr: ...

    def __init__(self, expr: "PathExpressionLike"):
        # TODO validate
        if isinstance(expr, self.__class__):
            expr = expr._expr
        self._expr = expr
    
    def __repr__(self):
        return f"{self.__class__.__qualname__}({self._expr!r})"

    # TODO
    # @functools.lru_cache(maxsize=1)
    def _cached_is_concrete_single(self, expr: ...):
        match expr:
            case str():
                return not wcmatch.fnmatch.is_magic(
                    expr, 
                    flags=wcmatch.fnmatch.BRACE,
                )
        return False

    # TODO
    # @functools.lru_cache(maxsize=1)
    def _cached_matcher(self, expr: ...):
        return wcmatch.fnmatch.compile(
            expr, 
            flags=wcmatch.fnmatch.BRACE,
            limit=0,
        )

    @property
    def is_concrete_single(self):
        return self._cached_is_concrete_single(self._expr)

    # TODO keep order
    def resolve(self, paths: Iterable[str]):
        return self._cached_matcher(self._expr).filter(paths)

    def match(self, path: str):
        return self._cached_matcher(self._expr).match(path)

    # TODO with paths??
    def expand(self, paths: Iterable[str] | None = None):
        if paths is not None:
            raise NotImplementedError("TODO")

        match self._expr:
            case str():
                return bracex.expand(self._expr, limit=0)
            case _ if isinstance(self._expr, (Sequence, numpy.ndarray)):
                # TODO
                return numpy.reshape(
                    [bracex.expand(expr, limit=0) for expr in numpy.reshape(self._expr, -1)],
                    -1,
                )
            
        raise ValueError("TODO")


# TODO
PathExpressionLike = PathExpression | str | Sequence[str] | numpy.typing.ArrayLike # | EllipsisType


def is_path_expression_like(expr: PathExpressionLike | Any):
    match expr:
        case PathExpression():
            return True
        case str():
            return True
        case _ if isinstance(expr, Sequence):
            return True
        case numpy.ndarray():
            return True
        case _:
            pass
    return False