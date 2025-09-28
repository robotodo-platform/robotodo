import numpy as np
import scipy.stats as stats
import trimesh
import trimesh.transformations as tra

from robotodo.utils.pose import Pose
from robotodo.utils.geometry import TriangularMesh, PolygonMesh


# TODO perf
# TODO ref: from isaacsim.replicator.grasping.sampler_utils import sample_antipodal
def sample_antipodal(object_mesh: trimesh.Trimesh, **kwargs) -> list[np.ndarray]:
    """Sample antipodal grasp poses for a given mesh.

    Args:
        object_mesh: A trimesh.Trimesh object to sample grasp poses from.
        **kwargs: Dictionary of parameters:
            num_candidates: Target number of grasp candidates to attempt to sample.
            num_orientations: Number of different orientations to sample per valid grasp axis.
            gripper_maximum_aperture: Maximum width between gripper fingers in meters.
            gripper_standoff_fingertips: Distance from fingertip contact points to the gripper's origin along the negative approach direction.
            gripper_approach_direction: Unit vector [x, y, z] indicating the approach direction in the gripper's local frame.
            grasp_align_axis: Unit vector [x, y, z] indicating the gripper's local axis to align with the physical grasp line.
            orientation_sample_axis: Unit vector [x, y, z] indicating the gripper's local axis around which to sample orientations.
            lateral_sigma: Standard deviation for random perturbation of grasp center point along grasp axis.
            random_seed: Seed for random number generation for reproducibility.
            verbose: If True, print detailed messages during processing.

    Returns:
        list: List of 4x4 homogeneous transformation matrices representing valid grasp poses
    """
    # Extract parameters with defaults
    num_candidates = kwargs.get("num_candidates", 100)
    num_orientations = kwargs.get("num_orientations", 1)
    gripper_maximum_aperture = kwargs.get("gripper_maximum_aperture", 0.08)
    gripper_standoff_fingertips = kwargs.get("gripper_standoff_fingertips", 0.1)
    gripper_approach_direction = kwargs.get("gripper_approach_direction", np.array([0, 0, 1]))
    grasp_align_axis = kwargs.get("grasp_align_axis", np.array([0, 1, 0]))
    orientation_sample_axis = kwargs.get("orientation_sample_axis", np.array([0, 1, 0]))
    lateral_sigma = kwargs.get("lateral_sigma", 0.0)
    random_seed = kwargs.get("random_seed")
    verbose = kwargs.get("verbose", False)

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Validate input parameters
    if num_candidates < 1:
        raise ValueError("num_candidates must be positive")
    if num_orientations < 1:
        raise ValueError("num_orientations must be positive")
    if gripper_maximum_aperture <= 0:
        raise ValueError("gripper_maximum_aperture must be positive")

    # Normalize input vectors and ensure they are numpy arrays
    gripper_approach_direction = np.array(gripper_approach_direction, dtype=float)
    if not np.isclose(np.linalg.norm(gripper_approach_direction), 1.0):
        raise ValueError("gripper_approach_direction must be a unit vector.")

    grasp_align_axis = np.array(grasp_align_axis, dtype=float)
    if not np.isclose(np.linalg.norm(grasp_align_axis), 1.0):
        raise ValueError("grasp_align_axis must be a unit vector.")

    orientation_sample_axis = np.array(orientation_sample_axis, dtype=float)
    if not np.isclose(np.linalg.norm(orientation_sample_axis), 1.0):
        raise ValueError("orientation_sample_axis must be a unit vector.")

    # Algorithm: Sample antipodal pairs using rejection sampling.
    # 1. Choose random points on the object surface
    # 2. Cast rays in opposite direction of the surface normal
    # 3. Find intersections with the other side of the object
    # 4. Create grasp axis between these antipodal points
    # 5. Only keep points with distance <= gripper aperture

    max_gripper_width = gripper_maximum_aperture

    # Calculate number of surface samples needed based on candidates and orientations
    num_surface_samples = max(1, int(num_candidates // num_orientations))
    if verbose:
        print(
            f"Attempting to generate {num_candidates} candidates from {num_surface_samples} surface samples and {num_orientations} orientations."
        )

    # Sample points from the mesh surface
    surface_points, face_indices = object_mesh.sample(num_surface_samples, return_index=True)
    surface_normals = object_mesh.face_normals[face_indices]

    # Cast rays in opposite direction of the surface normal
    ray_directions = -surface_normals
    ray_intersections, ray_indices, _ = object_mesh.ray.intersects_location(
        surface_points, ray_directions, multiple_hits=True
    )

    failed_distance_checks = 0
    grasp_centers = []
    grasp_axes = []

    # Process each sampled point to find valid grasp candidates
    for point_idx in range(num_surface_samples):
        try:
            # Find the intersection points for this ray
            ray_hits = ray_intersections[np.where(ray_indices == point_idx)]

            # Skip if no intersections found
            if len(ray_hits) == 0:
                continue

            # Find the furthest intersection point for more stable grasps
            if len(ray_hits) > 1:
                # Calculate distances from original point
                distances = np.linalg.norm(ray_hits - surface_points[point_idx], axis=1)
                # Get the furthest valid point (within gripper width constraint)
                valid_indices = np.where(distances <= max_gripper_width)[0]
                if len(valid_indices) > 0:
                    furthest_idx = valid_indices[np.argmax(distances[valid_indices])]
                    opposing_point = ray_hits[furthest_idx]
                else:
                    # All hits were further than max aperture
                    failed_distance_checks += 1
                    continue
            else:
                # Only one hit, check its distance
                opposing_point = ray_hits[0]  # Use index 0 as it's the only hit
                distance = np.linalg.norm(opposing_point - surface_points[point_idx])
                if distance > max_gripper_width:
                    failed_distance_checks += 1
                    continue

            # Calculate grasp axis and distance
            grasp_axis = opposing_point - surface_points[point_idx]
            axis_length = np.linalg.norm(grasp_axis)

            # Only accept points with valid distances
            # Check axis_length > trimesh.tol.zero in case start and end points are coincident
            if axis_length > trimesh.tol.zero and axis_length <= max_gripper_width:
                # Normalize grasp axis
                grasp_axis /= axis_length

                # Apply lateral perturbation if requested
                if lateral_sigma > 0:
                    # Center is perturbed along the grasp axis using a truncated normal distribution
                    # Boundaries ensure the perturbed center stays between the two contact points
                    center_ratio_lower = 0.0  # Ratio along axis for surface_points[point_idx]
                    center_ratio_upper = 1.0  # Ratio along axis for opposing_point
                    midpoint_ratio = 0.5
                    sigma_ratio = lateral_sigma / axis_length  # Scale sigma relative to axis length

                    # Define bounds for the truncated normal distribution in terms of standard deviations
                    a = (center_ratio_lower - midpoint_ratio) / sigma_ratio
                    b = (center_ratio_upper - midpoint_ratio) / sigma_ratio

                    truncated_dist = stats.truncnorm(a, b, loc=midpoint_ratio, scale=sigma_ratio)
                    center_offset_ratio = truncated_dist.rvs()
                    grasp_center = surface_points[point_idx] + grasp_axis * axis_length * center_offset_ratio
                else:
                    # Place grasp center exactly at midpoint between contacts
                    grasp_center = surface_points[point_idx] + grasp_axis * axis_length * 0.5

                grasp_centers.append(grasp_center)
                grasp_axes.append(grasp_axis)  # Store normalized axis
            else:
                failed_distance_checks += 1

        except IndexError as e:
            # Log the error with specific index and continue processing other points
            print(f"Error processing surface sample point {point_idx}. IndexError: {e}")
        except Exception as e:
            # Catch other potential errors during processing
            print(f"Unexpected error processing surface sample point {point_idx}: {e}")

    # Generate different orientations around each grasp axis
    rotation_angles = np.linspace(-np.pi, np.pi, num_orientations, endpoint=False)

    # Calculate standoff translation vector (along negative approach direction)
    # Ensure approach direction is normalized
    standoff_translation = gripper_approach_direction * -gripper_standoff_fingertips

    # Create final grasp transformations in world frame
    grasp_transforms = []
    for center, axis in zip(grasp_centers, grasp_axes):
        # Align the specified gripper axis (grasp_align_axis) with the calculated grasp axis ('axis').
        # The third axis orientation is determined implicitly by trimesh.
        try:
            align_matrix = trimesh.geometry.align_vectors(grasp_align_axis, axis)
        except ValueError as e:
            # Handle cases where vectors are collinear, etc.
            if verbose:
                print(f"Skipping grasp due to alignment error for axis {axis}: {e}")
            continue

        # Combine transformations: translate to center, align axes, apply orientation rotation, apply standoff
        center_transform = tra.translation_matrix(center)

        # Create transformation matrices for each orientation around the orientation_sample_axis
        orientation_transforms = [
            tra.rotation_matrix(angle=angle, direction=orientation_sample_axis) for angle in rotation_angles
        ]

        standoff_transform = tra.translation_matrix(standoff_translation)  # Translation along Z (approach)

        for orient_tf_rot in orientation_transforms:
            # Full transform: T_center * R_align * R_orient * T_standoff
            # R_orient rotates around the specified orientation_sample_axis in the aligned frame
            # T_standoff translates along the approach_direction in the aligned frame
            full_orientation_tf = align_matrix.dot(orient_tf_rot).dot(standoff_transform)
            grasp_world_tf = center_transform.dot(full_orientation_tf)
            grasp_transforms.append(grasp_world_tf)

    if verbose:
        print(f"Generated {len(grasp_transforms)} grasp transforms from {num_surface_samples} surface samples.")
        print(f"Initial candidates before filtering: {num_surface_samples * num_orientations}")  # Adjusted calculation
        print(f"Rejected {failed_distance_checks} candidates due to distance constraints during antipodal search.")

    return grasp_transforms


from typing import Optional, TypedDict, Unpack

class AntipodalPoseSampler:
    """
    TODO doc

    Example:
    ...
    
    """

    # TODO
    class Config(TypedDict):
        gripper_maximum_aperture: Optional[float]
        """Maximum width between gripper fingers."""
        gripper_standoff_fingertips: Optional[float]
        """Distance from fingertip contact points to the gripper's origin along the negative approach direction."""
        gripper_approach_direction: ...
        """Unit vector [x, y, z] indicating the approach direction in the gripper's local frame."""
        # TODO mv gripper_align_axis: ...
        grasp_align_axis: ...
        """Unit vector [x, y, z] indicating the gripper's local axis to align with the physical grasp line."""

    def __init__(self, config: Config = Config(), **config_kwds: Unpack[Config]):
        # TODO
        ...
        self._config = self.Config(config, **config_kwds)

    # TODO !!!!!
    def sample(self, mesh: PolygonMesh | TriangularMesh) -> Pose:
        match mesh:
            case PolygonMesh():
                todo = mesh.to_triangular()
                return Pose.from_matrix(
                    sample_antipodal(
                        trimesh.Trimesh(
                            vertices=todo.vertices,
                            faces=todo.face_vertex_indices,
                        ),
                        **self._config,
                    )
                )
            case TriangularMesh():
                todo = mesh
                return Pose.from_matrix(
                    sample_antipodal(
                        trimesh.Trimesh(
                            vertices=todo.vertices,
                            faces=todo.face_vertex_indices,
                        ),
                        **self._config,
                    )
                )
            case _:
                raise NotImplementedError("TODO")
