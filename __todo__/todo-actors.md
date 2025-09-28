- Manipulator API
    - see `import isaacsim.robot.manipulators.examples.franka`

- Planner API

overly complicated???

```python
class DOFController:

    stiffness: ...
    damping: ...

    action_spec: ...
    
    def exec(self, action, **options):
        ...


class PoseController:
    action_spec: ...
    
    def exec(self, action, **options):
        # await self._articulation.dof_controller.exec(...)

        ...



# TODO
"""

panda.dof_driver


await panda.dof_controller.exec(
    panda.dof_controller.action_spec.sample(),
    error_threshold=1.e-1,
    # done_condition=...,
)

await panda.pose_controller.exec(
    {"target_pose": ...},
    link="panda_hand",
)

"""
```


```
import os

# TODO NOTE seealso https://curobo.org/notes/07_environment_variables.html
os.environ.setdefault("CUROBO_TORCH_CUDA_GRAPH_RESET", "1")

from curobo.types.robot import RobotConfig
from curobo.geom.types import WorldConfig, Mesh

from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)


from curobo.util.torch_utils import is_cuda_graph_available, is_cuda_graph_reset_available



from robotodo.engines.core import Pose
from tensorspecs import BoxSpec


# TODO NOTE high-level control ux
class FrankaPanda:

    @property
    def motions_spec(self):
        ...

    @property
    def motions(self):
        ...

    def drive(self, motions: ...):
        ...

    # TODO
    def reach(self, local_pose: Pose, link: str | None = None):
        ...

    # TODO
    def grasp(self):
        ...

    def release(self):
        ...


# panda = FrankaPanda()
# panda.reach(panda.root_pose.inv() * Pose(...))
# panda.grasp()
# panda.motions
# panda.drive(...)
# panda.gripper.aperture
# panda.gripper.open()
# panda.gripper.close()
panda.dof_positions

from robotodo.engines.core import Pose
from tensorspecs import BoxSpec

from robotodo.engines.isaac import Articulation


class PandaManipulator:
    # TODO !!!
    def __init__(self, articulation: Articulation):
        self._articulation = articulation
        ...

    @property
    def states_spec(self):
        # TODO 
        return BoxSpec(
            "n? dof",
            bounds=(
                panda.dof_position_limits[..., 0],
                panda.dof_position_limits[..., 1],
            ),
        )

    @property
    def states(self):
        return self._articulation.dof_positions

    def drive(self, states):
        # TODO
        raise NotImplementedError
panda_manipulator = PandaManipulator(panda)


panda_manipulator.states_spec.random()



panda.driver.dof_drive_types
from tensorspecs import TensorSpec


# TensorSpec(...)

```