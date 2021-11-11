import pybullet as p
import pybullet_data as pd
import time
import numpy as np
from kinematics import Kinematics
import pickle
import transformations as tf

pandaNumDofs = 7
ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
angle2rad = np.pi / 180


class PandaEnv():

    def __init__(self):
        self.dt = 1./240.
        self.g = 0. #[-9.81, 0.]
        self.cup_position = [0.5, 0.4, 0.07]
        self.cup_orientation = p.getQuaternionFromEuler([0.*angle2rad, 0.*angle2rad, 0.*angle2rad])
        self.jointPositions = [-0.8120802918224433, 1.7117461835241918, 1.3490842089772912, -2.1637789872092332, 0.24385141998571924, 2.9252367233956718, 1.8607798722822901, 0.03, 0.03]
        self.alpha = 0.99
        self.finger_target = 0.014
        self.finger_force = 140.
        # 0: regulate angle, 1: lift, 2: drop
        self.state = 2
        self.lift_position = 0.2
        self.put_position = 0.
        self.target_ee_orn = [-0.47772375,  0.52131663,  0.52135299,  0.47770289]
        self.m = 10.
        self.b = 300.
        self.m_r = 0.05
        self.b_r = 10.
        self.k_p = 1.
        self.k_i = 0.5
        self.f_d = -10.
        self.f_error_integral = 0.
        self.pandaEndEffectorIndex = 11
        self.pandaFtIndex = 7
        self.panda_joint_num = 9
        self.joint_list = range(self.panda_joint_num)
        self.force_list = [87., 87., 87., 87., 12., 12., 12., self.finger_force, self.finger_force]
        self.max_vel = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.05, 0.05]).reshape(9, 1)
        self.plane_stiffness = 10000.
        self.max_stiffness_noise = 0.
        self.gripper_mass = 1.01
        self.kinematics = Kinematics()
        self.flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        p.connect(p.GUI)

    def reset(self):
        # pybullet
        p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, self.g)
        self.t = 0.
        self.contact_flag = False

        self.planeId = p.loadURDF("plane.urdf")
        stiffness_noise = np.random.normal(0., self.max_stiffness_noise)
        p.changeDynamics(self.planeId, -1, contactStiffness=self.plane_stiffness+stiffness_noise, contactDamping=0.)
        self.cup = p.loadURDF('cup.urdf', self.cup_position, self.cup_orientation)
        self.panda = p.loadURDF("panda/panda.urdf", [0, 0, 0], [0, 0, 0, 1],
                                useFixedBase=True, flags=self.flags)

        p.addUserDebugLine([0, 0, 0], [0.2, 0, 0], [1, 0, 0], lineWidth=5,
                           parentObjectUniqueId=self.panda, parentLinkIndex=self.pandaEndEffectorIndex)
        p.addUserDebugLine([0, 0, 0], [0, 0.2, 0], [0, 1, 0], lineWidth=5,
                           parentObjectUniqueId=self.panda, parentLinkIndex=self.pandaEndEffectorIndex)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.2], [0, 0, 1], lineWidth=5,
                           parentObjectUniqueId=self.panda, parentLinkIndex=self.pandaEndEffectorIndex)

        p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], lineWidth=5,
                           parentObjectUniqueId=self.cup)
        p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], lineWidth=5,
                           parentObjectUniqueId=self.cup)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], lineWidth=5,
                           parentObjectUniqueId=self.cup)

        index = 0
        for j in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.panda, j)
            jointName = info[1]
            jointType = info[2]
            # print(info)
            if (jointType == p.JOINT_PRISMATIC):
                p.resetJointState(self.panda, j, self.jointPositions[index])
                index = index + 1
            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.panda, j, self.jointPositions[index])
                index = index + 1

        c = p.createConstraint(self.panda, self.pandaEndEffectorIndex,
                               self.cup, -1,
                               jointType=p.JOINT_FIXED,
                               jointAxis=[0, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0.])

        p.enableJointForceTorqueSensor(self.panda, self.pandaEndEffectorIndex)
        _, _, self.ee_ft, _ = p.getJointState(self.panda, self.pandaEndEffectorIndex)
        self.ee_pos, self.ee_orn, _, _, _, _, self.ee_vel, _ = p.getLinkState(
            self.panda, self.pandaEndEffectorIndex, computeLinkVelocity=1, computeForwardKinematics=1)

        ft_pos, ft_orn, _, _, _, _, _, _ = p.getLinkState(
            self.panda, self.pandaFtIndex, computeLinkVelocity=1, computeForwardKinematics=1)
        cup_pos, cup_orn = p.getBasePositionAndOrientation(self.cup)
        print('ee_pos')
        print(self.ee_pos)
        print('cup_pos')
        print(cup_pos)

        # self.ee_pos = list(self.ee_pos)
        # self.ee_pos[0] = 0.5
        # self.ee_pos[1] = 0.4
        # self.ee_pos[2] = 0.17
        # self.ee_orn = [0., 0., 0., 1.]
        # jointPoses = p.calculateInverseKinematics(self.panda,
        #                                           self.pandaEndEffectorIndex,
        #                                           self.ee_pos,
        #                                           self.ee_orn,
        #                                           ll, ul, jr,
        #                                           self.jointPositions,
        #                                           maxNumIterations=200)
        # print('jointPoses')
        # print(jointPoses)

        ft_R = np.array(p.getMatrixFromQuaternion(ft_orn)).reshape(3, 3)
        cup_R = np.array(p.getMatrixFromQuaternion(cup_orn)).reshape(3, 3)
        self.ft2cup = np.matmul(np.linalg.inv(cup_R), ft_R)

        distance = (np.array(cup_pos) - np.array(ft_pos)).reshape(3, 1)
        self.cup_com2sensor = np.matmul(np.linalg.inv(cup_R), distance).reshape(-1)
        print('cup_com2sensor')
        print(self.cup_com2sensor)

        self.ee_position = list(self.ee_pos)
        self.ee_orientation = list(self.ee_orn)

        q_offset = p.getQuaternionFromEuler([0., np.pi/36., 0.])
        R_offset = np.array(p.getMatrixFromQuaternion(q_offset)).reshape(3, 3)
        ee_R = np.array(p.getMatrixFromQuaternion(self.ee_orn)).reshape(3, 3)
        ee_R_ = np.matmul(R_offset, ee_R)
        ee_T_ = np.eye(4, 4)
        ee_T_[:3, :3] = ee_R_
        ee_q_ = tf.quaternion_from_matrix(ee_T_)
        print('ee_q_')
        print(ee_q_)

        self.ee_y_position = self.ee_pos[1]

    def step(self):
        p.stepSimulation()
        time.sleep(self.dt)
        self.t += self.dt

        stiffness_noise = np.random.normal(0., self.max_stiffness_noise)
        p.changeDynamics(self.planeId, -1, contactStiffness=self.plane_stiffness+stiffness_noise,
                         contactDamping=0.)

        _, _, self.ee_ft, _ = p.getJointState(self.panda, self.pandaEndEffectorIndex)
        # print('ee_ft', type(self.ee_ft))
        # print(self.ee_ft)
        ee_f = np.array(self.ee_ft)[:3].reshape(3, 1)
        ee_t = np.array(self.ee_ft)[3:].reshape(3, 1)
        cup_f = np.matmul(self.ft2cup, np.array(ee_f))
        ee_t2cup = np.matmul(self.ft2cup, np.array(ee_t))
        ee_tBycupF = np.cross(self.cup_com2sensor, cup_f.reshape(-1))
        cup_t = (ee_t2cup.reshape(-1) - ee_tBycupF).reshape(3, 1)
        self.cup_ft = np.concatenate((cup_f, cup_t), axis=0).reshape(-1)

        print('ee_ft')
        print(self.ee_ft)
        print('cup_f')
        print(cup_f)
        print('ee_t2cup')
        print(ee_t2cup)
        print('ee_tBycupF')
        print(ee_tBycupF)
        print('cup_ft')
        print(self.cup_ft)

        self.ee_pos, self.ee_orn, _, _, _, _, self.ee_vel, self.ee_angular_vel = p.getLinkState(
            self.panda, self.pandaEndEffectorIndex, computeLinkVelocity=1, computeForwardKinematics=1)

        if self.state == 0:
            self.ee_orientation[0] = self.alpha * self.ee_orientation[0] \
                                     + (1. - self.alpha) * self.target_ee_orn[0]
            self.ee_orientation[1] = self.alpha * self.ee_orientation[1] \
                                     + (1. - self.alpha) * self.target_ee_orn[1]
            self.ee_orientation[2] = self.alpha * self.ee_orientation[2] \
                                     + (1. - self.alpha) * self.target_ee_orn[2]
            self.ee_orientation[3] = self.alpha * self.ee_orientation[3] \
                                     + (1. - self.alpha) * self.target_ee_orn[3]
            jointPoses = p.calculateInverseKinematics(self.panda,
                                                      self.pandaEndEffectorIndex,
                                                      self.ee_position,
                                                      self.ee_orientation,
                                                      ll, ul, jr,
                                                      self.jointPositions,
                                                      maxNumIterations=200)
            jointPoses = np.array(jointPoses)
            jointPoses[7] = 0.03
            jointPoses[8] = 0.03
            p.setJointMotorControlArray(self.panda, self.joint_list, p.POSITION_CONTROL, jointPoses, forces=self.force_list)

            if np.linalg.norm(np.array(self.ee_orn)
                              - np.array(self.target_ee_orn)) < 1e-3:
                self.state = 1
                self.ee_position = list(self.ee_pos)

        if self.state == 1:
            self.ee_position[2] = self.alpha * self.ee_position[2] \
                                  + (1. - self.alpha) * self.lift_position
            jointPoses = p.calculateInverseKinematics(self.panda,
                                                      self.pandaEndEffectorIndex,
                                                      self.ee_position,
                                                      self.ee_orientation,
                                                      ll, ul, jr,
                                                      self.jointPositions,
                                                      maxNumIterations=200)
            jointPoses = np.array(jointPoses)
            jointPoses[7] = 0.03
            jointPoses[8] = 0.03
            p.setJointMotorControlArray(self.panda, self.joint_list,
                                        p.POSITION_CONTROL, jointPoses,
                                        forces=self.force_list)
            if np.linalg.norm(self.ee_pos[2] - self.lift_position) < 1e-3:
                print('Robot has approached the desired lift position')
                self.state = 2

        if self.state == 2:
            p.applyExternalTorque(self.cup, -1, [0.1, 0.3, 0.5], flags=p.WORLD_FRAME)
            p.applyExternalForce(self.cup, -1, [10., 20., 30.],
                                 [0., 0., 0.], flags=p.LINK_FRAME)

            f_z = -self.cup_ft[0]
            vel_z = self.ee_vel[2]
            pos_z = self.ee_pos[2]
            acc = self.admittance_control_translation(f_z, vel_z)
            vel_z_ = vel_z + acc * self.dt
            pos_vel = [0., 0., 0.]

            t_z_cup = -self.cup_ft[2]
            angular_vel_y = self.ee_angular_vel[1]
            angular_acc_y = self.admittance_control_rotation(t_z_cup, angular_vel_y)
            # print('angular_acc_y')
            # print(angular_acc_y)
            angular_vel_y_ = angular_vel_y + angular_acc_y * self.dt
            # print('angular_vel_y_')
            # print(angular_vel_y_)
            orn_vel = [0., 0., 0.]

            joint_vel = self.kinematics.solve_vIK(self.panda, self.pandaEndEffectorIndex, pos_vel, orn_vel)
            joint_vel = np.clip(joint_vel, -self.max_vel, self.max_vel)
            self.target_vel = joint_vel[:, 0]
            self.target_vel[7] = 0.
            self.target_vel[8] = 0.
            p.setJointMotorControlArray(self.panda, self.joint_list,
                                        p.VELOCITY_CONTROL,
                                        targetVelocities=self.target_vel,
                                        forces=self.force_list)
            contacts = p.getContactPoints(bodyA=self.cup, bodyB=self.planeId)
            if len(contacts):
                contact_pos = contacts[0][5]
                cup_pos, _ = p.getBasePositionAndOrientation(self.cup)
                # print('contact_point')
                # print(contact_pos)
                # print('cup_pos')
                # print(cup_pos)

        # ft_noise = np.random.normal(0., 0.1)

    def admittance_control(self, f_r, dx):
        ddx = (-self.b_r * dx - (1. + self.alpha) * f_r) / (self.m_r - self.alpha * self.cup_mass)


    def admittance_control_translation(self, f_e, dx):
        ddx = (self.f_d - f_e - self.b * dx) / self.m
        return ddx

    def admittance_control_rotation(self, t_e, dw):
        ddw = (-t_e - self.b_r * dw) / self.m_r
        return ddw

    def before_contact(self, dx, x):
        m = 1.
        b = 10.
        k = 1.
        x_r = -0.1
        ddx = (k * (x_r - x) - b * dx) / m
        return ddx


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    env = PandaEnv()
    env.reset()
    pos_x = []
    pos_y = []
    pos_z = []
    vel_x = []
    vel_y = []
    vel_z = []
    angular_vel_y = []
    f_x = []
    f_y = []
    f_z = []
    t_x = []
    t_y = []
    t_z = []
    step_num = 3000
    dt = 1./240.
    for i in range(1000):
        env.step()
        if env.state == 2:
            pos_x.append(env.ee_pos[0])
            pos_y.append(env.ee_pos[1])
            pos_z.append(env.ee_pos[2])
            vel_x.append(env.ee_vel[0])
            vel_y.append(env.ee_vel[1])
            vel_z.append(env.ee_vel[2])
            f_x.append(env.ee_ft[0])
            f_y.append(env.ee_ft[1])
            f_z.append(env.ee_ft[2])
            t_x.append(env.ee_ft[3])
            t_y.append(env.ee_ft[4])
            t_z.append(env.ee_ft[5])
            angular_vel_y.append(env.ee_angular_vel[1])

    # with open('./data/pos_z.pickle', 'wb') as f:
    #     pickle.dump(pos_z, f)
    # with open('./data/vel_z.pickle', 'wb') as f:
    #     pickle.dump(vel_z, f)
    # with open('./data/f_z.pickle', 'wb') as f:
    #     pickle.dump(f_z, f)

    t = np.linspace(0, len(t_z)*dt, len(t_z))

    plt.figure(1)
    plt.plot(t, f_x, label='f_x')
    plt.plot(t, f_y, label='f_y')
    plt.plot(t, f_z, label='f_z')
    plt.xlabel('Time')
    plt.ylabel('cup force')
    plt.legend()

    plt.figure(2)
    plt.plot(t, t_x, label='t_x')
    plt.plot(t, t_y, label='t_y')
    plt.plot(t, t_z, label='t_z')
    plt.xlabel('Time')
    plt.ylabel('cup torque')
    plt.legend()

    plt.figure(3)
    plt.plot(t, angular_vel_y, label='angular_vel_y')
    plt.xlabel('Time')
    plt.ylabel('Robot ee-y angular velocity')
    plt.legend()

    plt.figure(4)
    plt.plot(t, vel_z, label='linear_vel_z')
    plt.xlabel('Time')
    plt.ylabel('Robot ee-z linear velocity')
    plt.legend()

    # plt.figure(3)
    # plt.plot(t, f_z, label='f_z')
    # plt.xlabel('Time')
    # plt.ylabel('Force')
    # plt.title('m='+str(env.m)+', b='+str(env.b))
    # plt.legend()
    #
    # plt.figure(4)
    # plt.plot(pos_x, pos_y)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()

    plt.show()












