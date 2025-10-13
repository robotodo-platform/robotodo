"""
TODO

"""


import numpy
import scipy.spatial.transform
import einops


# TODO typing
# TODO ref https://graphicscompendium.com/opengl/18-lookat-matrix
def lookat(eye: ..., center: ..., up: ...):
    """
    TODO doc

    # >>> lookat([1, 1, 1], [0, 0, 0], up=[0, 0, 1])
    # >>> lookat([0, 1, 0], [0, 0, 0], up=[0, 0, 1])

    """

    # TODO float cast necesito?
    eye = numpy.asarray(eye, dtype="float")
    center = numpy.asarray(center, dtype="float")
    up = numpy.asarray(up, dtype="float")

    forward = center - eye
    forward /= numpy.linalg.norm(forward)

    u = numpy.cross(forward, up)
    v = numpy.cross(u, forward)
    w = -forward

    return numpy.stack((u, v, w), axis=-1)


# TODO
class ProtoPose:
    p: ...  # TODO xyz
    q: ...  # TODO xyzw
    # radians: ... # TODO radian xyz


# TODO FIXME perf
# TODO support (*batch_dims, data)
# TODO dataclass???
# TODO testing
class Pose:
    p: ...  # TODO xyz
    q: ...  # TODO xyzw
    # randians: ... # TODO radian xyz
    # degrees: ... # TODO radian xyz

    @classmethod
    def from_lookat(cls, p: ..., p_target: ..., up_axis: ... = [0, 0, 1]):
        """
        TODO doc
        """

        return cls(
            p=p,
            q=scipy.spatial.transform.Rotation.from_matrix(
                lookat(p, p_target, up=up_axis)
            ).as_quat(),
        )

    # TODO support for rotation matrix as well?
    @classmethod
    def from_matrix(cls, matrix):
        # TODO !!!!
        matrix = numpy.asarray(matrix)
        return cls(
            p=matrix[..., :3, 3],
            q=scipy.spatial.transform.Rotation.from_matrix(matrix[..., :3, :3]).as_quat(),
        )

    def __init__(
        self,
        # TODO
        # other: ProtoPose,
        *,
        p: ... = [0, 0, 0],
        q: ... = [0, 0, 0, 1],
        # radians: ... = [0, 0, 0],
        # degrees: ... = [0, 0, 0],
    ):
        # TODO broadcast
        self.p = numpy.asarray(p, dtype=numpy.float_)
        self.q = numpy.asarray(q, dtype=numpy.float_)

    def __repr__(self):
        return f"{Pose.__qualname__}(p={self.p!r}, q={self.q!r})"

    # TODO better ux!!!
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = tuple((index, ))
        return Pose(
            p=self.p[*index, :],
            q=self.q[*index, :],
        )

    # TODO
    def facing(self, p):
        # TODO
        raise NotImplementedError

    def rotated(self, angles):
        # raise NotImplementedError
        # TODO
        return Pose(
            p=self.p,
            q=(
                scipy.spatial.transform.Rotation.from_quat(self.q)
                * scipy.spatial.transform.Rotation.from_euler("XYZ", angles)
            ).as_quat(),
        )

    def translated(self, positions):
        raise NotImplementedError

    # TODO FIXME performance
    def inv(self):
        rot_inv = scipy.spatial.transform.Rotation.from_quat(self.q).inv()
        return Pose(
            # TODO
            p=-rot_inv.apply(self.p),
            q=rot_inv.as_quat(),
        )

    # TODO FIXME performance
    def __mul__(self, other: "Pose"):
        rot = scipy.spatial.transform.Rotation.from_quat(self.q)
        rot_other = scipy.spatial.transform.Rotation.from_quat(other.q)

        return Pose(
            p=self.p + rot.apply(other.p),
            # TODO this is wrong???
            q=(rot * rot_other).as_quat(),
            # q=other.q,
        )
    
    # TODO pack/unpack for all other methods!!!!!
    @property
    def radians(self):
        q_packed, ps = einops.pack([self.q], "* quat")
        [eulers] = einops.unpack(
            # TODO
            scipy.spatial.transform.Rotation.from_quat(q_packed).as_euler("XYZ"), 
            ps, 
            "* eulers",
        )
        return eulers
    
    @property
    def degrees(self):
        return numpy.rad2deg(self.radians)
    
    # TODO !!!
    def to_matrix(self):
        p = numpy.asarray(self.p)
        q = numpy.asarray(self.q)

        # TODO
        q_packed, ps = einops.pack([q], "* quat")
        R = scipy.spatial.transform.Rotation.from_quat(q_packed).as_matrix()
        [R] = einops.unpack(R, ps, "* out_base in_base",)

        eye = numpy.eye(4)
        shape = p.shape[:-1] + (4, 4)
        T = numpy.broadcast_to(eye, shape).copy()
        T[..., :3, :3] = R
        T[..., :3, 3] = p
        return T
    

    def __eq__(self, other: "Pose"):
        return (
            numpy.array_equiv(self.p, other.p) 
            and numpy.array_equiv(self.q, other.q)
        )
    
    def isclose(self, other: "Pose", rtol: float = 1.e-5, atol: float = 1.e-8):
        return numpy.allclose(
            (self.inv() * other).to_matrix(),
            numpy.eye(4),
            rtol=rtol, atol=atol,
        )
        

# TODO
def _todo_test_pose():
    a = Pose.from_matrix(
        numpy.array([
            [[0.0, -1.0, 0.0, 1.0],
            [1.0,  0.0, 0.0, 2.0],
            [0.0,  0.0, 1.0, 3.0],
            [0.0,  0.0, 0.0, 1.0]],
            
            [[1.0, 0.0, 0.0, 4.0],
            [0.0, 1.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 6.0],
            [0.0, 0.0, 0.0, 1.0]]
        ])    
    )

    a.to_matrix()

    Pose.from_matrix(
        numpy.array([
            [0.0, -1.0, 0.0, 1.0],
            [1.0,  0.0, 0.0, 2.0],
            [0.0,  0.0, 1.0, 3.0],
            [0.0,  0.0, 0.0, 1.0],
        ])
    ).to_matrix()
    # TODO
    # Pose().inv()
    # Pose(p=[1, 0, 0], q=[0, 0, 0, 1]) * Pose(p=[0, 0, 0], q=[0, 0, 0, 1])
    # Pose(p=[0, 0, 0], q=[0, 0, 0, 1]).inv()