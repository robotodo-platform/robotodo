import abc
import enum
from typing import Iterable

from tensorspecs import TensorLike
from robotodo.utils.pose import Pose


class ProtoScene(abc.ABC):
    """
    TODO doc
    """

    # TODO
    @classmethod
    # @abc.abstractmethod
    def load(
        cls, 
        source: str, 
    ):
        raise NotImplementedError
        ...

    # TODO
    def save(
        self,
        source: str | None = None
    ):
        raise NotImplementedError

    def __repr__(self):
        # TODO
        return (
            f"""{self.__class__.__qualname__}()"""
        )

    @abc.abstractmethod
    def traverse(self) -> Iterable[str]:
        ...

    @property
    @abc.abstractmethod
    def autostepping(self) -> bool:
        r"""
        TODO doc
        """
        ...

    @abc.abstractmethod
    async def step(self, timestep: float | None = None) -> float:
        r"""
        TODO doc
        """
        ...

    @property
    @abc.abstractmethod
    def gravity(self) -> TensorLike["xyz:3"]:
        r"""
        Gravity vector.
        Unit: :math:`\mathrm{m / s^2}`.
        TODO doc clarity: xyz direction
        """
        ...

    # TODO
    @property
    @abc.abstractmethod
    def viewer(self) -> ...:
        ...

