
class TestCuroboKinematicsParser:
    def test_todo(self):
        _ = """
        from robotodo.engines.isaac.articulation import Articulation

        panda = Articulation("/_01", scene=scene)
        from curobo.cuda_robot_model.urdf_kinematics_parser import UrdfKinematicsParser as CuroboUrdfKinematicsParser

        # TODO
        urdf_kinematics_parser = CuroboUrdfKinematicsParser(urdf_path="./todo-curobo-sample-configs/franka/franka_description/franka_panda.urdf")
        kinematics_parser = _CuroboKinematicsParser(
            list(panda.joints.values())
        )
        import numpy


        # TODO test
        parent_map_desired = urdf_kinematics_parser._parent_map
        parent_map_actual = kinematics_parser._parent_map

        assert {
            (key, value["parent"])
            for key, value in parent_map_desired.items()
        } == {
            (key, value["parent"])
            for key, value in parent_map_actual.items()
        }


        for link_name, base in (
            ("base_link", True),
            ("panda_link0", False),
            ("panda_link1", False),
            ("panda_link2", False),
            ("panda_link3", False),
            ("panda_link4", False),
            ("panda_link5", False),
            ("panda_link6", False),
            ("panda_link7", False),
            ("panda_link8", False),
            ("panda_hand", False),
            ("ee_link", False),
            ("panda_leftfinger", False),
            ("panda_rightfinger", False),
            ("right_gripper", False),
        ):
            # TODO
            urdf_link_params = urdf_kinematics_parser.get_link_parameters(link_name=link_name, base=base)
            link_params = kinematics_parser.get_link_parameters(link_name=f"{link_name}", base=base)

            assert urdf_link_params.joint_type == link_params.joint_type
            numpy.testing.assert_almost_equal(
                desired=urdf_link_params.fixed_transform, 
                actual=link_params.fixed_transform,
                decimal=3,
                err_msg=f"TODO {link_name} {base}",
            )

            joint_limits_desired = urdf_link_params.joint_limits
            joint_limits_actual = link_params.joint_limits
            if any(x is None for x in (joint_limits_desired, joint_limits_actual)):
                numpy.testing.assert_equal(
                    desired=joint_limits_desired,
                    actual=joint_limits_actual,
                )
            else:
                numpy.testing.assert_almost_equal(
                    desired=joint_limits_desired,
                    actual=joint_limits_actual,
                    decimal=3,
                )
        """
        