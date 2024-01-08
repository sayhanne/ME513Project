import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import raisimpy as raisim
import time
from environment.Franka_Jac_Calculation import Franka_Jacobian
from FuncPoly5th import FuncPoly5th


class Environment:
    def __init__(self, robot_path='/raisimLib/rsc/Panda/panda.urdf',
                 raisim_act_path="/../raisimLib/rsc/activation.raisim",
                 timestep=0.001):
        raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + raisim_act_path)
        self.world = raisim.World()
        self.timestep = timestep
        self.world.setTimeStep(self.timestep)
        self.world.addGround()
        robot_urdf_file = os.environ['WORKSPACE'] + robot_path
        self.robot: raisim.ArticulatedSystem = self.world.addArticulatedSystem(robot_urdf_file)
        self.cube = None
        self.table = None

        # States
        self.reached = False
        self.picked = False
        self.lifted = False
        self.carried = False
        self.lowered = False
        self.placed = False
        self.reset_done = True
        self.set_targets = True

        self.home_pos = np.zeros(3)
        self.pick_pos = np.zeros(3)
        self.place_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.start_pos = np.zeros(3)

        self.iterations = 0
        self.t_start = 0.1
        self.t_end = 0

        self.realtime = 0
        self.trajectory_data = {'pos': [], 'joint_angle': [], 'joint_velocity': [], 'torque': []}

        self.p_gain_ref = np.array([1000, 2500, 1000, 2500, 500, 50, 50, 50, 50])
        self.i_gain_ref = np.array([200, 500, 200, 500, 100, 10, 10, 10, 10])
        # self.p_gain = np.array([2000, 4000, 2000, 4000, 1000, 100, 100, 100, 100])
        # self.i_gain = np.array([400, 800, 400, 800, 200, 20, 20, 20, 20])

        # q_ref, prev_q_ref, dq_ref, prev_dq_ref
        self.prev_dq_ref = np.zeros((7, 1))
        self.dq_ref = np.zeros((7, 1))
        self.prev_q_ref = np.zeros((7, 1))
        self.q_ref = np.expand_dims(np.array([0, -0.785, 0, -2.356, 0, 1.65806, 0.7853]), axis=1)

        self.gripper_angles = [0., 0.]
        self.euler = None

        # Grid settings
        self.pick_points = None
        self.place_points = None
        self.grid_1_i = 0
        self.grid_2_i = 0
        self.curr_joint_angle = []
        self.curr_joint_vel = []
        self.target_angle = []
        self.target_velocity = []

    def create_grid(self):
        pick_x_range = np.linspace(0.4, 0.6, 4)
        pick_y_range = np.linspace(-0.1, -0.2, 4)
        fixed_z = 0.315
        grid_1, grid_2 = np.meshgrid(pick_x_range, pick_y_range)
        self.pick_points = np.array(list(zip(grid_1.flatten(), grid_2.flatten(), [fixed_z] * len(grid_1.flatten()))))
        # pick position
        place_x_range = np.linspace(0.4, 0.6, 4)
        place_y_range = np.linspace(0.1, 0.2, 4)
        fixed_z = 0.315
        grid_1, grid_2 = np.meshgrid(place_x_range, place_y_range)
        self.place_points = np.array(list(zip(grid_1.flatten(), grid_2.flatten(), [fixed_z] * len(grid_1.flatten()))))

        for p in range(len(self.pick_points)):
            s: raisim.Visual = server.addVisualSphere(name="grid1_point_{}".format(p), radius=0.005, colorR=0, colorG=1,
                                                      colorB=0)
            pos = self.pick_points[p].copy()
            pos[2] = 0.3 + 0.00006
            s.setPosition(pos)
        for p in range(len(self.place_points)):
            s: raisim.Visual = server.addVisualSphere(name="grid2_point_{}".format(p), radius=0.005, colorR=0, colorG=0,
                                                      colorB=1)
            pos = self.place_points[p].copy()
            pos[2] = 0.3 + 0.00006
            s.setPosition(pos)

    def record_data(self):
        # time = self.world.getSimulationTime()self.
        pos = self.robot.getFramePosition(7)
        angle = self.robot.getGeneralizedCoordinate()
        vel = self.robot.getGeneralizedVelocity()
        torque = self.robot.getGeneralizedForce()

        # self.trajectory_data['time'].append(time)
        self.trajectory_data['pos'].append(pos)
        self.trajectory_data['joint_angle'].append(angle)
        self.trajectory_data['joint_velocity'].append(vel)
        self.trajectory_data['torque'].append(torque)

    def reset_robot(self):
        # initial position
        self.robot.setPdGains(self.p_gain_ref, self.i_gain_ref)
        while self.realtime < 2.7:
            self.realtime = self.world.getWorldTime()
            server.integrateWorldThreadSafe()
            self.robot.setPdTarget([0, -0.785, 0, -2.356, 0, 1.65806, 0.7853, 0.014, 0.014], vel_targets=np.zeros([9]))
            time.sleep(self.world.getTimeStep())

        self.spawn_objects()
        self.create_grid()

    def spawn_objects(self):
        # Objects
        self.table: raisim.Box = self.world.addBox(x=0.8, y=0.8, z=0.3, mass=10, material='default')
        table_pos = [0.5, 0, 0.15]
        self.table.setPosition(table_pos)
        self.table.setName("table")
        self.cube: raisim.Box = self.world.addBox(x=0.04, y=0.04, z=0.03, mass=0.001, material='default')
        self.cube.setName("cube")
        self.cube.setAppearance("red")

    def next_ep(self):
        # TODO : add a condition to select all points and return a boolean value for seeing all points.
        if self.grid_2_i == 15:
            self.grid_1_i += 1
            self.grid_2_i = 0
        self.pick_pos = self.pick_points[self.grid_1_i]
        self.cube.setPosition(self.pick_pos.copy())
        self.place_pos = self.place_points[self.grid_2_i]
        self.grid_2_i += 1

    def get_target_pos(self, cube_pos):
        pos_end = cube_pos
        pos_end[0] += 0.014  # x-axis offset
        pos_end[1] -= 0.0002
        pos_end[2] += 0.16  # z-axis offset
        return pos_end

    def wait(self, target_pos):
        if np.linalg.norm(target_pos - self.robot.getFramePosition(7)) <= 0.001:
            print("Reached target pos")
            return True
        elif self.iterations >= (self.t_end - self.t_start) * 1000:
            return True
        else:
            return False

    def reach(self):
        if self.set_targets:
            self.home_pos = self.robot.getFramePosition(7)
            self.start_pos = self.home_pos.copy()
            self.target_pos = self.get_target_pos(self.cube.getPosition())
            self.target_pos[2] += 0.09
            self.t_start = np.ceil(self.realtime)
            self.t_end = self.t_start + 4
            self.set_targets = False
        if self.wait(target_pos=self.target_pos):
            self.reset_done = False
            self.reached = True
            self.iterations = 0
            self.set_targets = True
        else:
            self.iterations += 1

    def pick(self):
        if self.set_targets:
            self.open_gripper()
            self.t_start = np.ceil(self.realtime)
            self.t_end = self.t_start + 3
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.start_pos.copy()
            self.target_pos[2] -= 0.043
            self.set_targets = False
        if self.wait(target_pos=self.target_pos):
            self.picked = True
            self.reached = False
            self.iterations = 0
            self.set_targets = True
        else:
            self.iterations += 1

    def lift(self):
        if self.set_targets:
            self.close_gripper()
            self.t_start = np.ceil(self.realtime)
            self.t_end = self.t_start + 2
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.start_pos.copy()
            self.target_pos[2] += 0.05
            self.set_targets = False
        if self.wait(target_pos=self.target_pos):
            self.lifted = True
            self.picked = False
            self.set_targets = True
            self.iterations = 0
        else:
            self.iterations += 1

    def carry(self):
        if self.set_targets:
            self.t_start = np.ceil(self.realtime)
            self.t_end = self.t_start + 5
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.get_target_pos(self.place_pos)
            self.target_pos[2] = self.start_pos[2]
            self.set_targets = False
        if self.wait(target_pos=self.target_pos):
            self.carried = True
            self.lifted = False
            self.set_targets = True
            self.iterations = 0
        else:
            self.iterations += 1

    def lower(self):
        if self.set_targets:
            self.t_start = np.ceil(self.realtime)
            self.t_end = self.t_start + 3
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.start_pos.copy()
            self.target_pos[2] -= 0.04
            self.set_targets = False
        if self.wait(target_pos=self.target_pos):
            self.lowered = True
            self.carried = False
            self.set_targets = True
            self.iterations = 0
        else:
            self.iterations += 1

    def place(self):
        if self.set_targets:
            self.open_gripper()
            self.t_start = np.ceil(self.realtime)
            self.t_end = self.t_start + 2
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.start_pos.copy()
            self.target_pos[2] += 0.1
            self.set_targets = False
        if self.wait(target_pos=self.target_pos):
            self.placed = True
            self.lowered = False
            self.set_targets = True
            self.iterations = 0
        else:
            self.iterations += 1

    def reset(self):
        if self.set_targets:
            self.close_gripper()
            self.t_start = np.ceil(self.realtime)
            self.t_end = self.t_start + 10
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.home_pos.copy()
            self.set_targets = False
        if self.wait(target_pos=self.target_pos):
            self.reset_done = True
            self.placed = False
            self.set_targets = True
            self.iterations = 0
            return True

        else:
            self.iterations += 1
            return False

    def step(self):
        done = False

        if self.reset_done:
            self.reach()

        elif self.reached:
            self.pick()

        elif self.picked:
            self.lift()

        elif self.lifted:
            self.carry()

        elif self.carried:
            self.lower()

        elif self.lowered:
            self.place()

        elif self.placed:
            done = self.reset()

        jac_res, twist_res = self.trajectory_planning(real_time=self.realtime, t_start=self.t_start, t_end=self.t_end,
                                                      timestep=self.timestep,
                                                      pos_start=self.start_pos, pos_end=self.target_pos,
                                                      euler_start=self.euler, euler_end=self.euler)

        self.control(jac_res, twist_res)
        # self.record_data()

        return done

    def control(self, jac, twist):
        self.prev_dq_ref = self.dq_ref.copy()
        self.dq_ref = np.linalg.lstsq(jac, twist, rcond=None)[0]
        self.prev_q_ref = self.q_ref.copy()
        self.q_ref = np.add(np.add(self.prev_dq_ref.copy(), self.dq_ref.copy()) * self.world.getTimeStep() * 0.5,
                            self.prev_q_ref.copy())  # Integral
        self.robot.setPdTarget(pos_targets=np.hstack((self.q_ref.squeeze(), self.gripper_angles)), vel_targets=np.hstack((self.dq_ref.squeeze(), [0., 0.])))
        print(self.robot.getGeneralizedForce())
        # self.set_force(target_joint_angle=self.q_ref, target_joint_vel=self.dq_ref,
        #                p_gain=self.p_gain, i_gain=self.i_gain)

    def trajectory_planning(self, real_time, t_start, t_end, timestep, pos_start, pos_end,
                            euler_start=None, euler_end=None):
        pos_ref = np.zeros((3, 1))
        vel_ref = np.zeros((3, 1))
        acc_ref = np.zeros((3, 1))
        eul_ref = np.zeros((3, 1))
        deul_ref = np.zeros((3, 1))
        ddeul_ref = np.zeros((3, 1))
        for dim in range(3):
            pos_ref[dim], vel_ref[dim], acc_ref[dim] = FuncPoly5th(RealTime=real_time, tstart=t_start, te=t_end,
                                                                   z01=pos_start[dim], v01=0, a01=0,
                                                                   z02=pos_end[dim], v02=0, a02=0, dt=timestep)
            eul_ref[dim], deul_ref[dim], ddeul_ref[dim] = FuncPoly5th(RealTime=real_time, tstart=t_start, te=t_end,
                                                                      z01=euler_start[dim], v01=0, a01=0,
                                                                      z02=euler_end[dim], v02=0, a02=0, dt=timestep)

        twist = np.zeros((6, 1))
        twist[0] = vel_ref[0]
        twist[1] = vel_ref[1]
        twist[2] = vel_ref[2]
        twist[3] = deul_ref[0] + deul_ref[2] * math.sin(eul_ref[1])
        twist[4] = deul_ref[1] * math.cos(eul_ref[0]) - deul_ref[2] * math.cos(eul_ref[1]) * math.sin(eul_ref[0])
        twist[5] = deul_ref[1] * math.sin(eul_ref[0]) + deul_ref[2] * math.cos(eul_ref[0]) * math.sin(eul_ref[1])
        q = self.robot.getGeneralizedCoordinate()
        jac = Franka_Jacobian(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        return jac, twist

    # def set_force(self, target_joint_angle, target_joint_vel, p_gain, i_gain):
    #     joint_angle_cur = self.robot.getGeneralizedCoordinate()
    #     joint_vel_cur = self.robot.getGeneralizedVelocity()
    #     joint_angle_err = np.hstack((target_joint_angle.squeeze(), self.gripper_angles)) - joint_angle_cur
    #     joint_vel_err = np.hstack((target_joint_vel.squeeze(), [0., 0.])) - joint_vel_cur
    #     self.curr_joint_angle.append(joint_angle_cur[:-2])
    #     self.curr_joint_vel.append(joint_vel_cur[:-2])
    #     self.target_angle.append(target_joint_angle.squeeze())
    #     self.target_velocity.append(target_joint_vel.squeeze())
    #     # ddqref = (self.dq_ref - self.prev_dq_ref) / self.timestep
    #     # acc_ref = np.concatenate((ddqref.squeeze(), [0., 0.]))
    #     tau = p_gain * joint_angle_err + i_gain * joint_vel_err + self.robot.getNonlinearities(self.world.getGravity())
    #     self.robot.setGeneralizedForce(tau)
    # self.record_data()

    def open_gripper(self):
        self.gripper_angles = [0.04, 0.04]

    def close_gripper(self):
        self.gripper_angles = [0.014, 0.014]

    def rot2eul(self, rot):
        beta = -np.arcsin(rot[2, 0])
        alpha = np.arctan2(rot[2, 1] / np.cos(beta), rot[2, 2] / np.cos(beta))
        gamma = np.arctan2(rot[1, 0] / np.cos(beta), rot[0, 0] / np.cos(beta))
        return np.array((alpha, beta, gamma))


if __name__ == '__main__':
    dt = 0.001
    episode_no = 0
    env = Environment(timestep=dt)
    server = raisim.RaisimServer(env.world)
    server.launchServer(8080)
    env.reset_robot()

    rot_mat = env.robot.getFrameOrientation(7)
    env.euler = env.rot2eul(rot_mat)

    env.next_ep()
    #
    while True:
        env.realtime = env.world.getWorldTime()
        server.integrateWorldThreadSafe()
        # env.grid_points()
        ep_done = env.step()
        if ep_done:
            print("Episode ", episode_no, "Finished.")
            # Trajectories.append(env.trajectory_data)
            env.next_ep()
        time.sleep(dt)
        # np.save('traj.npy', Trajectories)

    # for i in range(7):
    #     plt.figure()
    #     plt.plot(range(len(env.curr_joint_angle[3000:])), np.asarray(env.curr_joint_angle)[3000:, i],
    #              label="Curr angle {}".format(i))
    #     plt.plot(range(len(env.target_angle[3000:])), np.asarray(env.target_angle)[3000:, i],
    #              label="target angle{}".format(i))
    #     plt.legend()
    #     plt.figure()
    #     plt.plot(range(len(env.curr_joint_vel[3000:])), np.asarray(env.curr_joint_vel)[3000:, i],
    #              label="Curr vel {}".format(i))
    #     plt.plot(range(len(env.target_velocity[3000:])), np.asarray(env.target_velocity)[3000:, i],
    #              label="target vel {}".format(i))
    #     plt.legend()
    # plt.show()
    server.killServer()
