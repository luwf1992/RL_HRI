import pybullet as p
import numpy as np


class Kinematics:

    def solve_vIK(self, robot, end_effector_index, linear_v, angular_v):
        """Caculate the joint velocity of a Cartesian velocity

        :param robot: robot ID from loadURDF
        :param end_effector_index: int, end-effector link index
        :param linear_v: list of float, desired end-effector linear Cartesian velocity
        :param angular_v: list of float, desired end-effector angular Cartesian velocity
        :return: numpy array [num, 1], desired joint velocity
        """

        linear_v_array = np.array(linear_v).reshape(3, 1)
        angular_v_array = np.array(angular_v).reshape(3, 1)
        v_array = np.concatenate((linear_v_array, angular_v_array), axis=0)
        _, _, local_position, _, _, _, _, _ = p.getLinkState(robot, end_effector_index,
                                            computeLinkVelocity=1, computeForwardKinematics=1)
        joint_positions, joint_velocities, _ = self.getMotorJointStates(robot)
        zero_vec = [0.0] * len(joint_positions)
        jac_t, jac_r = p.calculateJacobian(robot, end_effector_index, local_position,
                                           joint_positions, zero_vec, zero_vec)
        jac = np.concatenate((jac_t, jac_r), axis=0)
        jac_inv = np.linalg.pinv(jac)
        joint_vel = np.dot(jac_inv, v_array)
        return joint_vel

    def getMotorJointStates(self, robot):
        joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
        joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    # def solve_IK(self, pose_p):
    #     """Caculate the joint configuration of a cartesian position
    #     pose_p = position[3] + orn[3]
    #     """
    #
    #     homej_list = np.array(self.homej).tolist()
    #
    #     pos = pose_p[:3]
    #     # Quaternion[4]
    #     orn = p.getQuaternionFromEuler(pose_p[3:])
    #
    #     joints = p.calculateInverseKinematics(
    #         bodyUniqueId=self.UR5_Uid,
    #         endEffectorLinkIndex=self.ee_tip_link,
    #         targetPosition=pos,
    #         targetOrientation=orn,
    #         lowerLimits=[-17, -2.3562, -17, -17, -17, -17],
    #         upperLimits=[17, 0, 17, 17, 17, 17],
    #         jointRanges=[17] * 6,
    #         restPoses=homej_list,
    #         maxNumIterations=100,
    #         residualThreshold=1e-5)
    #     joints = np.array(joints)
    #     joints[joints > 2 * np.pi] = joints[joints > 2 * np.pi] - 2 * np.pi
    #     joints[joints < -2 * np.pi] = joints[joints < -2 * np.pi] + 2 * np.pi
    #
    #     return joints
    #
    # def solve_vIK(self, pose_p, pose_v):
    #     """Caculate the joint velocity of a cartesian position, velocity
    #     q_dot = J_inv(q)*x_dot
    #     pose_p = position[3] + orn[3]
    #     pose_v = x_dot[3] + w[3]
    #     """
    #
    #     targetj_p = self.solve_IK(pose_p).tolist()
    #
    #     _, _, localPosition, _, _, _, _, _ = p.getLinkState(self.UR5_Uid,
    #                                                         self.ee_tip_link,
    #                                                         computeLinkVelocity=1,
    #                                                         computeForwardKinematics=1)
    #
    #     zero_vec = [0.0] * len(targetj_p)
    #     J_t, J_r = p.calculateJacobian(self.UR5_Uid,
    #                                    self.ee_tip_link,
    #                                    localPosition,
    #                                    targetj_p,
    #                                    zero_vec,
    #                                    zero_vec)
    #
    #     J_q = J_t + J_r  # jacobian at the configuration targetj_p
    #     J_inv = np.linalg.pinv(J_q)
    #     targetj_v = np.dot(J_inv, pose_v)
    #
    #     return targetj_p, targetj_v


