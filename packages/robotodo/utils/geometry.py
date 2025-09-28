

import dataclasses



@dataclasses.dataclass(slots=True)
class PolygonMesh:
    vertices: ...
    face_vertex_counts: ...
    face_vertex_indices: ...

    # TODO
    # TODO FIXME perf
    # TODO ref: from isaacsim.replicator.grasping.sampler_utils import usd_mesh_to_trimesh
    def to_triangular(self, copy: bool = False):
        # TODO !!!
        if copy: raise NotImplementedError("TODO")

        vertices = self.vertices
        tri_face_vertex_indices = []

        offset = 0
        # Iterate over the face vertex counts where 'count' is the number of vertices in each face
        for count in self.face_vertex_counts:
            # Current face indices using the offset and count
            indices = self.face_vertex_indices[offset : offset + count]
            if count == 3:
                # If the face is a triangle, add it directly to the faces list
                tri_face_vertex_indices.append(indices)
            elif count == 4:
                # If the face is a quad, split it into two triangles
                tri_face_vertex_indices.extend([[indices[0], indices[1], indices[2]], [indices[0], indices[2], indices[3]]])
            else:
                # Fan triangulation for polygons with more than 4 vertices
                # NOTE: This approach works for convex polygons but may not be optimal for concave ones
                for i in range(1, count - 1):
                    tri_face_vertex_indices.append([indices[0], indices[i], indices[i + 1]])
            offset += count

        return TriangularMesh(
            vertices=vertices,
            # TODO asarray
            face_vertex_indices=tri_face_vertex_indices,
        )


@dataclasses.dataclass(slots=True)
class TriangularMesh:
    vertices: ...
    face_vertex_indices: ...


# TODO
@dataclasses.dataclass(slots=True)
class Cube:
    ...


@dataclasses.dataclass(slots=True)
class Sphere:
    radius: ...
