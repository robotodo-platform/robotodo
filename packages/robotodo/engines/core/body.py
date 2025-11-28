import abc
import enum
from typing import NotRequired, TypedDict

from tensorspecs import TensorLike
from robotodo.utils.pose import Pose
from robotodo.utils.geometry import Plane, Box, Sphere, PolygonMesh
from robotodo.engines.core.path import PathExpressionLike
from robotodo.engines.core.scene import ProtoScene
from robotodo.engines.core.entity import ProtoEntity
from robotodo.engines.core.material import ProtoMaterial


# TODO deprecate
# class MotionKind(enum.IntEnum):
#     """
#     TODO doc
#     https://maniskill.readthedocs.io/en/latest/user_guide/concepts/simulation_101.html#actor-types-dynamic-kinematic-static
#     """
#     NONE = -1
#     STATIC = 0
#     KINEMATIC = 1
#     DYNAMIC = 2
# TODO deprecate


class BodyKind(enum.IntEnum):
    NONE = -1
    RIGID = 0
    DEFORMABLE_VOLUME = 1
    DEFORMABLE_SURFACE = 2


class BodySpec(TypedDict):
    # TODO
    kind: NotRequired[BodyKind]
    geometry: NotRequired[Plane | Box | Sphere | PolygonMesh]
    # material: NotRequired[ProtoMaterial]


class ProtoBody(ProtoEntity, abc.ABC):

    @classmethod
    @abc.abstractmethod
    def create(
        cls, 
        ref: PathExpressionLike, 
        scene: ProtoScene, 
        spec: BodySpec = BodySpec(),
    ):
        ...

    @classmethod
    @abc.abstractmethod
    def load_usd(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: ProtoScene,
        spec_overrides: BodySpec = BodySpec(),
    ):
        ...
    
    @classmethod
    @abc.abstractmethod
    def load(
        cls, 
        ref: PathExpressionLike, 
        source: str, 
        scene: ProtoScene,
        spec_overrides: BodySpec = BodySpec(),
    ):
        ...

    kind: TensorLike["* value", BodyKind]

    pose: Pose
    # TODO
    velocity: Pose
    
    # parent: ProtoEntity
    # pose_in_parent: Pose

    fixed: TensorLike["* value", bool]

    # TODO 
    # kinematic: TensorLike["* value", bool]
    # TODO deprecate
    # motion_kind: TensorLike["* value", EntityMotionKind]

    collision: "ProtoCollision"

    inertia: ...

    # TODO dedicated BodyGeometry??
    geometry: ...
    material: ProtoMaterial | None
    mass: TensorLike["* value"] | None
    mass_center: Pose | None


# TODO
class ProtoRigidBody(ProtoBody, abc.ABC):
    ...


# TODO
class ProtoDeformableBody(ProtoBody, abc.ABC):
    ...


class ProtoCollision(abc.ABC):
    enabled: TensorLike["* value", bool]
    on_contact: ...
