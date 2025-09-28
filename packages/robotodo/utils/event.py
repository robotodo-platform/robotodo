
import abc
from contextlib import AbstractContextManager
from typing import Callable, Generic, TypeVar, Awaitable, AsyncIterator

_T = TypeVar("_T")


# TODO read https://github.com/python/cpython/issues/119154
class BaseAsyncEventStream(
    abc.ABC,
    Generic[_T],
):
    # TODO ergonometry: dont expect users to call __enter__ everytime!!!!
    @abc.abstractmethod
    def subscribe(
        self, 
        callable: Callable[[_T], None | Awaitable[None]],
    ) -> AbstractContextManager[None]:
        ...

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[_T]:
        ...

    @abc.abstractmethod
    def __anext__(self) -> Awaitable[_T]:
        ...