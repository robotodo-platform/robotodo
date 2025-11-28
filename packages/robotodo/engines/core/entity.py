import abc
import enum
from typing import Set, Type, TypeVar

from tensorspecs import TensorLike
from robotodo.engines.core.path import PathExpressionLike
# TODO
from robotodo.engines.core.scene import ProtoScene


_T = TypeVar("_T")


class ProtoEntity(abc.ABC):
    r"""
    TODO doc
    """

    @classmethod
    @abc.abstractmethod
    def load_usd(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: ProtoScene,
    ):
        ...
    
    @classmethod
    @abc.abstractmethod
    def load(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: ProtoScene,
    ):
        ...

    @classmethod
    def __class_getitem__(cls, label: TensorLike["*", str] | None):
        # TODO
        label
        return cls

    # TODO
    @abc.abstractmethod
    def __init__(
        self, 
        ref: "ProtoEntity | PathExpressionLike", 
        scene: ProtoScene | None = None,
    ):
        ...

    # TODO necesito?
    @property
    # @abc.abstractmethod
    def prototypes(self) -> Set[Type]:
        ...
        raise NotImplementedError

    # TODO necesito?
    # @abc.abstractmethod
    def astype(self, prototype: Type[_T]) -> _T:
        ...
        raise NotImplementedError

    def __repr__(self):
        return (
            f"""{self.__class__.__qualname__}"""
            f"""{f"[{self.label!r}]" if self.label is not None else ""}"""
            # TODO
            f"""({self.path}, scene={self.scene})"""
        )

    @property
    @abc.abstractmethod
    def path(self) -> TensorLike["*", str]:
        ...

    @property
    @abc.abstractmethod
    def scene(self) -> ProtoScene:
        ...

    # TODO necesito??
    # parent: "ProtoEntity"
    # children: list["ProtoEntity"]

    # TODO necesito??
    @property
    def label(self) -> TensorLike["*", str] | None:
        return None
