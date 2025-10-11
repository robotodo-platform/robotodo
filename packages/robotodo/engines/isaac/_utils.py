import asyncio
import contextlib
import dataclasses
import functools

import numpy
import torch

from robotodo.utils.pose import Pose
from robotodo.utils.geometry import PolygonMesh
from robotodo.utils.event import BaseSubscriptionPartialAsyncEventStream
from robotodo.engines.core.entity_selector import PathExpression, PathExpressionLike

from .scene import Scene


class USDPrimHelper:
    # TODO drop dep on Scene; pass usd stage directly
    # TODO support usd prims directly??
    def __init__(self, path: PathExpressionLike, scene: Scene, _usd_prims_ref: ... = None):
        if _usd_prims_ref is not None:
            # TODO
            raise NotImplementedError

        self._scene = scene
        self._path = PathExpression(path)

        self._usd_prims_ref = _usd_prims_ref

    # TODO FIXME performance thru prim obj caching
    @property
    # TODO invalidate !!!!!
    # @functools.cached_property
    def _usd_prims(self):
        # TODO
        if self._usd_prims_ref is not None:
            return self._usd_prims_ref

        return [
            self._scene._usd_stage.GetPrimAtPath(p)
            for p in self._scene.resolve(self._path)
        ]
        
    # TODO invalidate!!!!
    @functools.cached_property
    def _usd_xform_cache(self):
        # TODO Usd.TimeCode.Default()
        cache = self._scene._kernel.pxr.UsdGeom.XformCache()

        def _on_changed(notice, sender):
            # TODO
            cache.Clear()

        # TODO NOTE life cycle
        cache._notice_handler = _on_changed
        # TODO
        cache._notice_token = self._scene._kernel.pxr.Tf.Notice.Register(
            self._scene._kernel.pxr.Usd.Notice.ObjectsChanged, 
            cache._notice_handler, 
            self._scene._usd_stage,
        )

        return cache
    
    # TODO
    # TODO https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/usd/transforms/compute-prim-bounding-box.html
    @functools.cached_property
    def _usd_bbox_cache(self):
        raise NotImplementedError
        return self._scene._kernel.pxr.UsdGeom.BBoxCache(
            self._scene._kernel.pxr.Usd.TimeCode.Default(),
            [self._scene._kernel.pxr.UsdGeom.Tokens.default_,],
        )

    @property
    def pose(self):
        # TODO 
        return Pose.from_matrix(
            numpy.stack([
                numpy.asarray(
                    self._usd_xform_cache
                    .GetLocalToWorldTransform(prim)
                    .RemoveScaleShear()
                ).T # TODO NOTE col-major
                for prim in self._usd_prims
            ])
        )
    
    @pose.setter
    def pose(self, value: Pose):
        pxr = self._scene._kernel.pxr
        omni = self._scene._kernel.omni
        # TODO
        self._scene._kernel.enable_extension("omni.physx")
        self._scene._kernel.import_module("omni.physx.scripts.physicsUtils")

        p = numpy.broadcast_to(value.p, (len(self._usd_prims), 3))
        q = numpy.broadcast_to(value.q, (len(self._usd_prims), 4))

        p_vec3fs = pxr.Vt.Vec3fArrayFromBuffer(p)
        # NOTE this auto-converts from xyzw to wxyz
        q_quatfs = pxr.Vt.QuatfArrayFromBuffer(q)

        with pxr.Sdf.ChangeBlock():
            for prim, p_vec3f, q_quatf in zip(self._usd_prims, p_vec3fs, q_quatfs):
                xformable = pxr.UsdGeom.Xformable(prim)
                omni.physx.scripts.physicsUtils \
                    .set_or_add_translate_op(xformable, p_vec3f)
                omni.physx.scripts.physicsUtils \
                    .set_or_add_orient_op(xformable, q_quatf)