

from typing import Optional, TypedDict, Unpack

from robotodo.engines.core import PathExpressionLike

from ._kernel import Kernel
from .scene import Scene
from .articulation import Articulation
from .entity import Entity


class USDLoader:
    class Config(TypedDict):
        path: Optional[PathExpressionLike]
        exist_ok: Optional[bool]

    # TODO use scene._usd_current_stage
    async def __call__(
        self,
        resource_or_model: ...,
        scene: Scene,
        config: Config = Config(),
    ):
        
        match resource_or_model:
            case str() as resource:
                pass
            # TODO support for Usd prims directly??
            case _:
                raise NotImplementedError(f"TODO {resource_or_model}")
        
        prim_path = config.get("path", None)
        if prim_path is None:
            prim_path = scene._kernel.omni.usd.get_stage_next_free_path(
                scene._usd_current_stage, 
                path="/",
                prepend_default_prim=False,
            )

        if prim_path is not None:
            if scene._usd_current_stage.GetPrimAtPath(prim_path).IsValid():
                if not config.get("exist_ok", False):
                    raise RuntimeError(f"Path already exists: {prim_path}")
        prim = scene._kernel.isaacsim.core.utils.stage \
            .add_reference_to_stage(
                usd_path=resource,
                prim_path=prim_path,
            )
        
        # TODO FIXME upstream entity: no path roundtrip; ref underlying prim directly
        return Entity(
            path=prim.GetPath().pathString,
            scene=scene,
        )


# TODO usd has two modes: reference and sublayer; 
# this func should handle both; maybe infer from paths? 
async def load_usd(
    resource_or_model: ...,
    scene: Scene,
    config: USDLoader.Config = USDLoader.Config(),
    **config_kwds: Unpack[USDLoader.Config],
):
    return await USDLoader()(
        resource_or_model=resource_or_model, 
        scene=scene, 
        config=USDLoader.Config(config, **config_kwds),
    )



class URDFLoader:
    class Config(TypedDict):
        path: Optional[PathExpressionLike]
        fix_root_link: Optional[bool]
        num_copies: Optional[int]
        exist_ok: Optional[bool]

    # TODO use scene._usd_current_stage
    async def __call__(
        self, 
        resource_or_model: ...,
        scene: Scene,
        config: Config = Config(),
    ) -> Articulation:
        """
        Load a URDF model into the scene and return its articulation view.

        :param resource_or_model: The URDF resource (file path or model object).
        :param scene: The scene to load the URDF model into.
        :param config: Configuration options for loading the URDF model.
        :return: An :class:`Articulation` representing the loaded URDF model.
        """
        
        omni = scene._kernel.omni
        isaacsim = scene._kernel.isaacsim

        scene._kernel.enable_extension("isaacsim.asset.importer.urdf")

        # TODO
        scene._kernel.app.get_extension_manager() \
            .set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
        is_success, import_config = omni.kit.commands.execute(
            "URDFCreateImportConfig",
        )
        assert is_success

        import_config.make_default_prim = False  # Make the robot the default prim in the scene
        import_config.fix_base = config.get("fix_root_link", False) # Fix the base of the robot to the ground
        import_config.merge_fixed_joints = False
        # import_config.convex_decomp = False  # Disable convex decomposition for simplicity
        # import_config.self_collision = False  # Disable self-collision for performance

        if config.get("num_copies", None) is not None:
            raise NotImplementedError("TODO")

        # TODO
        match resource_or_model:
            case str() as resource:
                pass
            case _:
                raise NotImplementedError(f"TODO {resource_or_model}")

        is_success, robot_model = omni.kit.commands.execute(
            "URDFParseFile",
            # TODO
            urdf_path=resource,
            import_config=import_config,
        )
        assert is_success

        prim_path = config.get("path", None)
        if prim_path is None:
            prim_path = omni.usd.get_stage_next_free_path(
                scene._usd_current_stage, 
                path="/",
                prepend_default_prim=False,
            )

        if prim_path is not None:
            if scene._usd_current_stage.GetPrimAtPath(prim_path).IsValid():
                if not config.get("exist_ok", False):
                    raise RuntimeError(f"Path already exists: {prim_path}")
            
        stage_context = omni.usd.get_context_from_stage(scene._usd_current_stage)
        if stage_context is None:
            raise RuntimeError(
                f"The USD stage is invalid. "
                f"Stage: {scene._usd_current_stage}"
            )
        if not stage_context.is_writable():
            raise RuntimeError(
                f"The USD stage does not appear to be writable. Crash may result! "
                f"Stage: {scene._usd_current_stage}"
            )
        
        # TODO
        urdf_interface = isaacsim.asset.importer.urdf._urdf.acquire_urdf_interface()
        prim_path_temp = urdf_interface.import_robot(
            assetRoot="",
            assetName="",
            robot=robot_model,
            importConfig=import_config,
            stage=stage_context.get_stage().GetEditTarget().GetLayer().identifier,
        )

        # TODO rm?
        # is_success, prim_path_temp = omni.kit.commands.execute(
        #     "URDFImportRobot",
        #     urdf_robot=robot_model,
        #     import_config=import_config,
        #     # get_articulation_root=True,
        #     # TODO
        #     dest_path=scene._usd_current_stage.GetEditTarget().GetLayer().identifier,
        # )
        # assert is_success


        if prim_path is None:
            prim_path = prim_path_temp
        else:
            is_success, _ = omni.kit.commands.execute(
                "MovePrim",
                path_from=prim_path_temp,
                path_to=prim_path,
            )
            assert is_success        

        # TODO
        # TODO use this instead of get_articulation_root??
        """
        from isaacsim.core.utils.prims import (
            get_articulation_root_api_prim_path,
            get_prim_at_path,
            get_prim_parent,
            get_prim_property,
            set_prim_property,
        )
        """
        return Articulation(
            scene._kernel.isaacsim.core.utils.prims
                .get_articulation_root_api_prim_path(prim_path), 
            scene=scene,
        )


async def load_urdf(
    resource_or_model: ...,
    scene: Scene,
    config: URDFLoader.Config = URDFLoader.Config(),
    **config_kwds: Unpack[URDFLoader.Config]
):
    return await URDFLoader()(
        scene=scene, 
        resource_or_model=resource_or_model, 
        config=URDFLoader.Config(config, **config_kwds),
    )


class USDSceneLoader:
    async def __call__(
        self,
        resource_or_model: ... = None,
        # TODO mv engine??
        _kernel: Kernel = None,
    ) -> Scene:
        if _kernel is None:
            raise NotImplementedError("TODO")

        # TODO
        omni = _kernel.omni

        ctx = omni.usd.get_context()

        # TODO NOTE current impl opens model as sublayer: prob safest??
        is_success, message = await ctx.new_stage_async()
        if not is_success:
            raise RuntimeError(f"Failed to create empty USD scene: {message}")
        stage = ctx.get_stage()
        if stage is None:
            # TODO
            raise RuntimeError("TODO")
        stage.GetRootLayer().subLayerPaths.append(resource_or_model)

        return Scene(_kernel=_kernel, _usd_stage=stage)


        # TODO
        # is_success, message = await ctx.open_stage_async(resource_or_model)
        # if not is_success:
        #     raise RuntimeError(f"Failed to load USD scene {resource_or_model}: {message}")

        # stage = ctx.get_stage()
        # # TODO check None
        # # TODO
        # return Scene(_kernel=_kernel, _usd_stage=stage)
    

async def load_usd_scene(
    resource_or_model: ...,
    _kernel: Kernel,
):
    return await USDSceneLoader()(
        resource_or_model=resource_or_model,
        _kernel=_kernel,
    )