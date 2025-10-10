
import abc
from contextlib import AbstractContextManager
from typing import Callable, Generic, TypeVar, Awaitable, AsyncIterator

_T = TypeVar("_T")


# TODO read https://github.com/python/cpython/issues/119154
class BaseAsyncEventStream(
    Generic[_T],
    abc.ABC,
):
    """
    Async event stream base class.
    """

    # TODO ergonometry: dont expect users to call __enter__ everytime!!!!
    @abc.abstractmethod
    def subscribe(
        self, 
        callable: Callable[[_T], None | Awaitable[None]],
    ) -> AbstractContextManager[None]:
        """
        TODO doc
        """

        ...

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[_T]:
        ...

    @abc.abstractmethod
    def __anext__(self) -> Awaitable[_T]:
        ...


import asyncio

class BaseSubscriptionPartialAsyncEventStream(
    BaseAsyncEventStream[_T],
    Generic[_T],
    abc.ABC,
):
    """
    Async event stream helper class backed by the implementation of 
    :meth:`subscribe`. Unlike in :class:`BaseAsyncEventStream`, users are not
    required to implement methods other than :meth:`subscribe`.
    """

    def __aiter__(self):
        queue = asyncio.Queue()

        loop = asyncio.get_running_loop()
        def listener(event: None):
            loop.call_soon_threadsafe(queue.put_nowait, event)

        # TODO !!!!
        async def agenerator():
            with self.subscribe(listener):
                while True:
                    yield await queue.get()

        return aiter(agenerator())

    def __anext__(self):
        return self.__aiter__().__anext__()
