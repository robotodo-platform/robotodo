- TODO: license:
    ```
    # SPDX-License-Identifier: Apache-2.0
    ```

- TODO: doc:
    ```
    .conda/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/sysadmin/lab/robotodo/.conda/lib/python3.11/site-packages/omni/libcarb.so)
    ```
    ```
    conda install -c conda-forge gcc=12 -y
    ```

- TODO:
    ```
    import omni.kit.app

    # extension enablement
    app = omni.kit.app.get_app()
    em = app.get_extension_manager()
    em.set_extension_enabled_immediate("isaacsim.app.about", True)
    em.set_extension_enabled_immediate("omni.services.livestream.nvcf", True)

    # extension settings
    import carb
    settings = carb.settings.get_settings()
    settings.get("exts/omni.services.transport.server.http/port")
    # settings.set("/app/runLoops/main/rateLimitEnabled", False)          # or True + raise Hz
    settings.get("/app/runLoops/main/rateLimitEnabled")
    settings.get_settings_dictionary()
    settings.get("/app/python/logSysStdOutput")
    settings.get("/app/enableStdoutOutput")
    ```

- TODO: agilex curobo demo
    0. `Path`
        - `Path("/some/expression/**")`
    0. `Scene`
        - `.copy("/**", target=["/a", "/b"])`
    0. `Group`
        - `.pose`
        - `.bounding_box`
    1. `Camera("/**")`
        - `.pose`: read/write
    2. `Articulation("/**")`
        - ensure homogenous
        - `.dof_positions`: read/write
    
```python
    scene = Scene()

    camera = Camera(
        "/World/Camera{1..128}", 
        resolution=(32, 16),
        scene=scene,
    )

    scene.copy(
        "/World/Camera1",
        target="/World/Camera_0_{1..128}",
    )
```


```
# def todo_get_physx():
#     s = kernel.omni.physx.get_physx_simulation_interface()
#     # TODO NOTE valid stage is REQUIRED!!!
#     s.attach_stage(kernel.omni.usd.get_context().get_stage_id())

# def todo_sim():
#     s.simulate(0, 1)
#     s.fetch_results()

# def todo_new_stage():
#     kernel.omni.usd.get_context().new_stage()


# # kernel.submit(todo_sim).result()
# kernel.submit(todo_new_stage).result()
# kernel.submit(todo_sim).result()

# # TODO NOTE valid stage is REQUIRED!!!
# s.attach_stage(kernel.omni.usd.get_context().get_stage_id())
# s.get_attached_stage()
# s.simulate(0, 1)
# # s.fetch_results()
# s.detach_stage()
# kernel.omni.usd.get_context().new_stage()
# s.get_attached_stage()
# dir(kernel._omni.physx.get_physx_simulation_interface())

def todo():
    context = kernel.omni.usd.create_context("sfsa")
    context.can_open_stage()
    context.get_stage()
    context.new_stage()
    return context.get_stage_id()

kernel.submit(todo).result()
kernel.omni.usd.get_context().get_stage()
todo()
kernel.omni.usd.get_context("fafs")
kernel.omni.usd.get_context().can_open_stage()
kernel.omni.usd.get_context_from_stage_id(0)

dir(kernel.omni.usd.get_context())

stage = (kernel.omni.usd.get_context().get_stage())
# dir(stage)

kernel.omni.usd.get_context().get_stage_id()
dir(kernel.pxr.Usd)
# kernel.pxr.UsdUtils.StageCache.Get().Find(kernel.pxr.Usd.Id(0))
dir(kernel.pxr.UsdUtils.StageCache.Get())
kernel.pxr.UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
def todo():
    return kernel.omni.usd.get_context().new_stage()

kernel.submit(todo).result()
dir(kernel._omni.physx)
dir(kernel._omni.physx.get_physx_interface())



# def todo():
#     physx = kernel.omni.physx.get_physx_interface()
#     physx.update_simulation(0, 0)
#     return

# kernel.submit(todo).result()
physx = kernel.omni.physx.acquire_physx_interface()
physx
physx.is_running()
# physx.start_simulation()
# physx.is_running()
# physx.update_simulation(0, 0)

s = kernel.omni.physx.get_physx_simulation_interface()
# s.get_full_contact_report()
# s.simulate(0, 0)
s.simulate(1/60, 1/60)
s.fetch_results()
dir(kernel.omni.physx)
kernel.omni.physx.get_physx_simulation_interface?
s = kernel.omni.physx.get_physx_simulation_interface()
s.get_attached_stage()

list(kernel.omni.usd.get_context().get_stage().Traverse())
```

.conda/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.base.kit
.conda/lib/python3.11/site-packages/isaacsim/exts/isaacsim.core.cloner/isaacsim/core/cloner/impl/cloner.py




```
kernel.omni.physx.get_physx_simulation_interface?

stage = kernel.omni.usd.get_context().get_stage()
for p in (stage.Traverse()):
    p

dir(kernel.pxr.Usd.SchemaRegistry)
p.GetTypeName()
p.GetAppliedSchemas()
p.GetProperties()
import omni.usd

ctx = omni.usd.get_context()
dir(ctx)

kernel.omni.physics.tensors.create_simulation_view("warp", stage_id=-1)
```


```
articulation = Articulation("/Franka", scene=scene)

articulation.dof_types
# articulation._physics_tensor_get_articulation_view().get_dof_drive_model_properties()
articulation.driver.dof_drive_types
articulation.driver.dof_target_positions = articulation.dof_positions * 2
pos_error = articulation.dof_positions - articulation.driver.dof_target_positions
pos_error

articulation.driver.dof_target_velocities
articulation.dof_positions

# %timeit -n 10 articulation._physics_tensor_get_articulation_view().check()
# %timeit -n 100 articulation._physics_tensor_get_articulation_view().get_dof_velocity_targets()
articulation.root_path
articulation.link_paths
```


`Pose` api


FIXME clone tensors before returning!!!!


```
# event = AsyncEvent()
# event.stream()
# await event.future()
```

```
from robotodo.engines.isaac import engine

engine

scene = engine.get_default_scene()
scene

```

```python

# robot.controller.drive
# robot.controller.reach
# robot.controller.grasp


class BaseDriveController:
    def drive(self):
        ...


class BaseMotionController:

    def reach(self, local_pose):
        ...

    def grasp(self, local_poses):
        ...


```


```python
import trimesh
# TODO
from isaacsim.replicator.grasping.sampler_utils import sample_antipodal


sample_antipodal?
```


```
articulation.driver.enable()
articulation.driver.disable()
```


.conda/lib/python3.11/site-packages/omni/kernel/py/omni/ext/_impl/fast_importer.py

import importlib
importlib.invalidate_caches()


```
from isaacsim.replicator.grasping.ui import grasping_ui_utils

# TODO
grasping_ui_utils.clear_debug_draw()
grasping_ui_utils.draw_grasp_samples_as_lines?

```


```
import sapien

pose_a = sapien.Pose(p=[-2, 3, 4], q=[.231, .321, .231, .23])
pose_b = sapien.Pose(p=[1, 2, 3], q=[.23, .21, .12, .32])

pose_a.inv().to_transformation_matrix()
pose_a = Pose(p=[-2, 3, 4], q=[.23, .231, .321, .231])
pose_b = Pose(p=[1, 2, 3], q=[.32, .23, .21, .12])

pose_a.inv().to_matrix()
pose_a.inv() * pose_a
Pose(p=[-2, 3, 4], q=[.23, .231, .321, .231]).inv() #* Pose(p=[1, 2, 3], q=[.32, .23, .21, .12])
```



```python

class FrankaPanda:
    # TODO
    def compute_finger_aperture(self):
        ...

    def reach(self, pose):
        ...

    # TODO
    def grasp(self, candidate_poses):
        ...

    



```