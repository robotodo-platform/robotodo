- entity: attachment (obj moves along?)
- articulation: joints, links

- perf/ux: batch api:
```
with batch_changes:
    some_entity.pose = ...
    ...
```
- ux: Entity.viewer()
- ux: Camera.viewer()


- perf: isaac
```
for loop_name in ["main", "present", "rendering_0"]:
    # TODO important
    kernel.get_settings().set(f"/app/runLoops/{loop_name}/rateLimitEnabled", False)

kernel.get_settings().get("/app/renderer/skipWhileInvisible")
# await kernel._app_framework.app.next_update_async()
# kernel.step_app_loop()
kernel.get_settings().get("/app/content/emptyStageOnStart")
```

- todo: multi scene setup:
```
# TODO
import omni
# omni.usd.get_context_from_stage(scene._usd_stage)
c = omni.usd.get_context()
await c.attach_stage_async(scene._usd_stage)
```