"""
TODO

"""


# TODO
import bracex

import wcmatch.fnmatch



class PathExpression:
    # TODO expr: list[str | Path]
    def __init__(self, expr: "PathExpressionLike"):
        self._expr = str(expr)

    def __str__(self):
        return self._expr
    
    def __repr__(self):
        return f"{PathExpression.__qualname__}({str(self._expr)!r})"

    # TODO
    def resolve(self, paths: list[str]):
        # TODO cache?
        # TODO brace expansion?
        matcher = wcmatch.fnmatch.compile(self._expr, limit=0)
        return matcher.filter(paths)

    # TODO
    def expand(self):
        return bracex.expand(self._expr)
    
    def match(self, path: str):
        # TODO
        matcher = wcmatch.fnmatch.compile(self._expr, limit=0)
        return matcher.match(path)


# TODO
PathExpressionLike = str | PathExpression