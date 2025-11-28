- ux: camera:
    - near, far for clipping range: https://openusd.org/dev/api/class_usd_geom_camera.html#acef48b476b648999849b0e5efe47fb44

- ux: 
    - `defer_changes`
- doc: usd cam convention:
    - https://github.com/PixarAnimationStudios/OpenUSD/blob/9f6115dd3fa516a252ab6d91056a083c7d4cc38a/pxr/base/gf/matrix4d.cpp#L807
    - https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/reference_conventions.html#usd-axes
    - https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/viewport/camera.html#look-at-a-prim

- ux: curobo:
    https://github.com/NVlabs/curobo/blob/ebb71702f3f70e767f40fd8e050674af0288abe8/src/curobo/cuda_robot_model/cuda_robot_generator.py#L319
    custom kinematics parser needs to be passed

- bug: physx
    - `scene._isaac_physx.reset_simulation()` required when object changes? (e.g. usdphysics joint type)
    - `scene._isaac_physics_tensor_view.update_articulations_kinematic()` may be required. when?

- ux:
    - `scene.add("/todo", "file:/some/path/to.usd")`
    - `scene.remove("/todo")`
    - `scene.get("/todo")`
    - `scene.rename`
    - `scene.traverse`

    - `Articulation(<Entity>)`
    - `Articulation("/some/path/to/prim/in/scene", scene=scene)`
    - `JointSpec`

    - `RigidBody.apply(<Entity>)`
    - `DeformableBody.apply(<Entity>)`

- ux: sensor
    - `camera.stream.to("/some/place.mp4")`
    - `microphone.stream.to("/some/place.mp3")`

- ux: stage unit consistency:
    ```
    UsdGeom.GetStageMetersPerUnit(stage)
    ```

- bug: deformable usdgeom.tetmesh: 
    - `--/app/useFabricSceneDelegate=false`
    - FIX: https://github.com/isaac-sim/IsaacSim/issues/277#issuecomment-3471138590


- articulation: joints, links api
    - curobo motion planning
- entity: attachment (obj moves along?)
    - attach entity: use parenting

- articulation: urdf importer -> articulation api -> usd
    - detail: handle "package://": parent until package.xml is found

- ux: unified api for modeling
    - `load_*(...)` without `scene`: reads the model only

- perf/ux: batch api:
```
async with ChangeBlock():
    some_entity.pose = ...
    ...
```
- ux: Entity.viewer()
- ux: Camera.viewer()


- bug: omni.physics.tensors xform requires step? or maybe just xforms under articulation??
    https://github.com/isaac-sim/IsaacSim/issues/223#issuecomment-3393260151
- perf: isaac: 
    - doc: profiling: 
    https://docs.omniverse.nvidia.com/kit/docs/kit-manual/108.1.0/guide/profiling.html
    - hide viewport by default?? +20ms latency once enabled
    - todo
    ```
    for loop_name in ["main", "present", "rendering_0"]:
        # TODO important
        kernel.get_settings().set(f"/app/runLoops/{loop_name}/rateLimitEnabled", False)
    kernel.get_settings().get("/app/renderer/skipWhileInvisible")
    kernel.get_settings().get("/app/content/emptyStageOnStart")
    ```
- compat: 
    `/app/vulkan=`? test on non-vulkan platforms (BUG: segfault when vulkan install incomplete)

- bug: omni.physics attach stage messes up timeline

- doc: debug aid: gizmo world-to-local: https://forums.developer.nvidia.com/t/gizmo/185706


- doc: 
    - tut: usd search: https://build.nvidia.com/nvidia/usdsearch https://simready.com


- perf: pose: 
    https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/spatial/transform.py#L424