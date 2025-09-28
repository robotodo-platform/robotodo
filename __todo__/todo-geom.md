- rm dataclass dep??
```python
# class Mesh:
#     __slots__ = ("points", "face_vertex_counts", "face_vertex_indices")

#     points: ...
#     face_vertex_counts: ...
#     face_vertex_indices: ...

#     def __init__(
#         self,
#         points: ...,
#         face_vertex_counts: ...,
#         face_vertex_indices: ...,
#     ):
#         self.points = points
#         self.face_vertex_counts = face_vertex_counts
#         self.face_vertex_indices = face_vertex_indices

# %timeit -n 1000 Mesh(1, 2, 3)
```
