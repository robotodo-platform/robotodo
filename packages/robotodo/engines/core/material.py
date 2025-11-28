import abc
import enum

from tensorspecs import TensorLike
from robotodo.engines.core.entity import ProtoEntity


class ProtoMaterial(ProtoEntity, abc.ABC):
    # TODO
    static_friction: TensorLike["* value"]
    dynamic_friction: TensorLike["* value"]
    density: TensorLike["* value"]

    young: TensorLike["* value"] | None
    poisson: TensorLike["* value"] | None

    surface_thickness: TensorLike["* value"] | None