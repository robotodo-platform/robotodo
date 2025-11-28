


from tensorspecs import TensorSpec, TensorTableSpec, TensorTableLike


# TODO
class GraspPlanner:

    observation_spec = TensorTableSpec({
        "dof_positions": TensorSpec("n? dof"),
        # "candidate_poses": PoseSpec,
        # "obstacles": ...,
    })

    action_spec = TensorTableSpec({
        "dof_positions": TensorSpec("time n? dof"),
        "dof_velocities": TensorSpec("time n? dof"),
    })

    # TODO
    def compute_action(
        self, 
        observation: TensorTableLike[observation_spec],
        *,
        include_raw_result: bool = False,
    ) -> TensorTableLike[action_spec]:
        raise NotImplementedError
