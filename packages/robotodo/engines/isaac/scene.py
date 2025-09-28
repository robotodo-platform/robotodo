
import functools
import contextlib
import asyncio

from robotodo.engines.core.entity_selector import PathExpression, PathExpressionLike
# TODO
from robotodo.utils.event import BaseAsyncEventStream

# TODO
from ._kernel import Kernel, get_default_kernel


# TODO !!!! per stage???
class _PhysicsUpdateAsyncEventStream(BaseAsyncEventStream[None]):
    def __init__(self, scene: "Scene"):
        self._scene = scene

    @functools.cached_property
    def __isaac_physx_interface(self):
        return self._scene._kernel.omni.physx.get_physx_interface()

    # TODO
    @contextlib.contextmanager
    def subscribe(self, callable):
        # TODO cache?
        def physx_callback(dt: float):
            result = callable(None)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        sub = self.__isaac_physx_interface.subscribe_physics_step_events(physx_callback)
        yield
        sub.unsubscribe()

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



# TODO multiple scenes are not supported!!!
# TODO ? default scene: scene = Scene(); new scene: scene = engine.add(Scene())
class Scene:

    # TODO !!!!!
    def __init__(self, _kernel: Kernel | None = None, _usd_stage: ... = None):
        if _kernel is None:
            raise NotImplementedError("TODO")
        # TODO !!!!!
        self._kernel = _kernel
        self._usd_stage = _usd_stage

    # TODO !!!!!
    @property
    def _usd_current_stage(self):
        if self._usd_stage is not None:
            return self._usd_stage

        # TODO !!!
        stage = self._kernel.omni.usd.get_context().get_stage()
        if stage is None:
            # TODO !!!!!  Stage opening or closing already in progress so async???
            self._kernel.omni.usd.get_context().new_stage()
            stage = self._kernel.omni.usd.get_context().get_stage()
        # TODO check None
        assert stage is not None
        return stage

    def traverse(self, path: PathExpressionLike | None = None):
        # TODO
        if path is not None:
            raise NotImplementedError(f"TODO {path}")

        root_prim = (
            self._usd_current_stage.GetPseudoRoot()
            if path is None else
            # TODO
            self._usd_current_stage.GetPrimAtPath(self.resolve(path))
        )

        return (
            prim.GetPath().pathString
            for prim in self._kernel.pxr.Usd.PrimRange(root_prim)
        )

    def resolve(self, path: PathExpressionLike):
        return PathExpression(path).resolve(self.traverse())

    # TODO
    def copy(self, path: PathExpressionLike, target_path: PathExpressionLike):
        path = PathExpression(path)
        target_path = PathExpression(target_path)

        # TODO mv
        import isaacsim.core.cloner

        # TODO check if dir
        cloner = isaacsim.core.cloner.Cloner(stage=self._usd_current_stage)
        source_prim_path = self.resolve(path)
        if len(source_prim_path) != 1:
            raise NotImplementedError("TODO")
        source_prim_path = source_prim_path[0]

        cloner.clone(
            source_prim_path=source_prim_path,
            prim_paths=PathExpression(target_path).expand(),
        )

        # raise NotImplementedError

    def move(self, path: PathExpressionLike, target_path: PathExpressionLike):
        path = PathExpression(path)
        target_path = PathExpression(target_path)

        raise NotImplementedError

    # TODO
    def update(self):
        raise NotImplementedError
    
    # TODO
    @functools.cached_property
    def on_update(self):
        return _PhysicsUpdateAsyncEventStream(scene=self)
    
    # TODO this should be render??
    @property
    def on_render(self):
        raise NotImplementedError

    # TODO prefix with __isaac
    @functools.cached_property
    def __physics_tensor_view_cache(self):
        try:
            # TODO stage_id !!!!!!
            res = self._kernel.omni.physics.tensors.create_simulation_view("torch")
        except Exception as error:
            # TODO
            raise RuntimeError(
                "Failed to create physics view. "
                f"Does a prim of type `PhysicsScene` exist on {self._usd_current_stage}?"
                # TODO rm
                # f"Make sure the physics simulation is running: call {self.timeline_play}"
            ) from error
        if not res.is_valid:
            raise RuntimeError(f"Created physics view is invalid: {res}")
        return res

    def _physics_tensor_get_view(self):
        if not self.__physics_tensor_view_cache.is_valid:
            del self.__physics_tensor_view_cache
        return self.__physics_tensor_view_cache

    # TODO convenience
    @functools.cached_property
    def viewer(self):
        from .viewer import Viewer
        return Viewer(scene=self)
