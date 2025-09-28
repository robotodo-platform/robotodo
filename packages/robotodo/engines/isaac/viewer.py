
from typing import TypedDict

import numpy
import einops
from robotodo.utils.pose import Pose

from .scene import Scene
from .entity import Entity


class Viewer:
    def __init__(self, scene: Scene):
        self._scene = scene

    # TODO
    def open(self):
        raise NotImplementedError

    # TODO
    def close(self):
        raise NotImplementedError

    # TODO
    def play(self):
        raise NotImplementedError

    # TODO
    def pause(self):
        raise NotImplementedError

    # TODO
    @property
    def selected_entity(self):
        omni = self._scene._kernel.omni
        # TODO
        selection = omni.usd.get_context().get_selection()

        # TODO entity: support list
        # Entity(..., scene=self._scene)

        selection.get_selected_prim_paths()

        raise NotImplementedError

    # TODO
    @property
    def _isaac_debug_draw_interface(self):
        # TODO
        self._scene._kernel.enable_extension("isaacsim.util.debug_draw")
        return (
            self._scene._kernel.isaacsim.util.debug_draw._debug_draw
            .acquire_debug_draw_interface()
        )
    
        # TODO lifecycle
        # isaacsim = self._scene._kernel.isaacsim
        # isaacsim.util.debug_draw._debug_draw.release_debug_draw_interface

    def clear_drawings(self):
        iface = self._isaac_debug_draw_interface
        iface.clear_points()
        iface.clear_lines()

    class DrawPoseOptions(TypedDict):
        axis_length: float | None
        line_thickness: float | None
        line_opacity: float | None

    def draw_pose(self, pose: Pose, options: DrawPoseOptions = DrawPoseOptions()):
        """
        TODO doc


        """

        axis_length: float = options.get("axis_length", .02)
        line_thickness: float = options.get("line_thickness", 2)
        line_opacity: float = options.get("line_opacity", .5)

        # TODO x y z
        for mask in (
            numpy.asarray([1., 0., 0.]),
            numpy.asarray([0., 1., 0.]),
            numpy.asarray([0., 0., 1.]),
        ):
            start_points = pose.p
            # TODO
            end_points = (pose * Pose(p=mask * [axis_length, axis_length, axis_length])).p

            start_points, _ = einops.pack([start_points], "* xyz")
            end_points, _ = einops.pack([end_points], "* xyz")

            colors = einops.repeat(
                numpy.asarray([*mask, line_opacity]),
                "rgba -> b rgba",
                **einops.parse_shape(start_points, "b _"),
            )
            thicknesses = einops.repeat(
                numpy.asarray(line_thickness),
                "-> b",
                **einops.parse_shape(start_points, "b _"),
            )

            self._isaac_debug_draw_interface.draw_lines(
                numpy.asarray(start_points).tolist(), 
                numpy.asarray(end_points).tolist(), 
                numpy.asarray(colors).tolist(), 
                numpy.asarray(thicknesses).tolist(),
            )

        # TODO
        return



        # Axis colors: X=Red, Y=Green, Z=Blue
        x_color = [1.0, 0.0, 0.0, line_opacity]
        y_color = [0.0, 1.0, 0.0, line_opacity]
        z_color = [0.0, 0.0, 1.0, line_opacity]

        start_points = []
        end_points = []
        colors = []
        thicknesses = []


        for location, quat in grasp_poses:
            origin = [location[0], location[1], location[2]]
            # X axis
            x_axis = quat.Transform(Gf.Vec3d(1, 0, 0)) * axis_length
            x_end = [origin[0] + x_axis[0], origin[1] + x_axis[1], origin[2] + x_axis[2]]
            start_points.append(origin)
            end_points.append(x_end)
            colors.append(x_color)
            thicknesses.append(line_thickness)
            # Y axis
            y_axis = quat.Transform(Gf.Vec3d(0, 1, 0)) * axis_length
            y_end = [origin[0] + y_axis[0], origin[1] + y_axis[1], origin[2] + y_axis[2]]
            start_points.append(origin)
            end_points.append(y_end)
            colors.append(y_color)
            thicknesses.append(line_thickness)
            # Z axis
            z_axis = quat.Transform(Gf.Vec3d(0, 0, 1)) * axis_length
            z_end = [origin[0] + z_axis[0], origin[1] + z_axis[1], origin[2] + z_axis[2]]
            start_points.append(origin)
            end_points.append(z_end)
            colors.append(z_color)
            thicknesses.append(line_thickness)
        self._isaac_debug_draw_interface.draw_lines(start_points, end_points, colors, thicknesses)

