
import abc

from tensorspecs import TensorLike
from robotodo.engines.core.entity import ProtoEntity


class ProtoCamera(ProtoEntity, abc.ABC):
    r"""
    TODO doc convention
    TODO https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/reference_conventions.html#world-axes
    TODO https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/reference_conventions.html#default-camera-axes
    """

    @abc.abstractmethod
    async def read_rgba(
        self, 
        resolution: TensorLike["xy:2"] | None = None,
    ) -> TensorLike["* x y rgba:4", "float"]:
        ...

    # TODO
    @property
    @abc.abstractmethod
    def viewer(self) -> ...:
        ...