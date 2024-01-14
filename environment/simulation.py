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
        self.carried = False
        self.placed = False
        self.released = False
        self.reset_done = True
        self.set_targets = True

        self.pick_pos = np.zeros(3)
        self.place_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.start_pos = np.zeros(3)

        self.iterations = 0
        self.t_start = 0
        self.t_end = 0

        self.realtime = 0
        self.isRealTime = False
        # 4 different tasks
        self.trajectory_data = {'reach': {'pos': [], 'joint_angle': [], 'joint_velocity': [], 'torque': []},
                                'pick': {'pos': [], 'joint_angle': [], 'joint_velocity': [], 'torque': []},
                                'carry': {'pos': [], 'joint_angle': [], 'joint_velocity': [], 'torque': []},
                                'place': {'pos': [], 'joint_angle': [], 'joint_velocity': [], 'torque': []}}
        self.context_data = {'reach': {'start_pos': [], 'target_pos': []},
                             'pick': {'start_pos': [], 'target_pos': []},
                             'carry': {'start_pos': [], 'target_pos': []},
                             'place': {'start_pos': [], 'target_pos': []}}

        self.p_gain_ref = np.array([2500, 4000, 1000, 4000, 500, 500, 100, 20, 20])
        self.i_gain_ref = np.array([500, 800, 200, 800, 100, 100, 20, 4, 4])
        self.robot.setPdGains(self.p_gain_ref, self.i_gain_ref)
        # self.p_gain = np.array([2000, 4000, 2000, 4000, 1000, 100, 100, 100, 100])
        # self.i_gain = np.array([400, 800, 400, 800, 200, 20, 20, 20, 20])

        # q_ref, prev_q_ref, dq_ref, prev_dq_ref
        self.prev_dq_ref = None
        self.dq_ref = None
        self.prev_q_ref = None
        self.q_ref = None

        self.gripper_angles = [0., 0.]
        self.euler = None
        self.step_counter = 0

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
        fixed_z = 0.215
        grid_1, grid_2 = np.meshgrid(pick_x_range, pick_y_range)
        self.pick_points = np.array(list(zip(grid_1.flatten(), grid_2.flatten(), [fixed_z] * len(grid_1.flatten()))))
        # pick position
        place_x_range = np.linspace(0.4, 0.6, 4)
        place_y_range = np.linspace(0.1, 0.2, 4)
        fixed_z = 0.215
        grid_1, grid_2 = np.meshgrid(place_x_range, place_y_range)
        self.place_points = np.array(list(zip(grid_1.flatten(), grid_2.flatten(), [fixed_z] * len(grid_1.flatten()))))

        for p in range(len(self.pick_points)):
            s: raisim.Visual = server.addVisualSphere(name="grid1_point_{}".format(p), radius=0.005, colorR=0, colorG=1,
                                                      colorB=0)
            pos = self.pick_points[p].copy()
            pos[2] = 0.2 + 0.00006
            s.setPosition(pos)
        for p in range(len(self.place_points)):
            s: raisim.Visual = server.addVisualSphere(name="grid2_point_{}".format(p), radius=0.005, colorR=0, colorG=0,
                                                      colorB=1)
            pos = self.place_points[p].copy()
            pos[2] = 0.2 + 0.00006
            s.setPosition(pos)

    def record_data(self, task_name):
        pos = self.robot.getFramePosition(7)
        angle = self.robot.getGeneralizedCoordinate()[:-2]  # Discard finger joints
        vel = self.robot.getGeneralizedVelocity()[:-2]  # Discard finger joints
        torque = self.robot.getGeneralizedForce()[:-2]  # Discard finger joints

        # Trajectory
        self.trajectory_data[task_name]['pos'].append(pos)
        self.trajectory_data[task_name]['joint_angle'].append(angle)
        self.trajectory_data[task_name]['joint_velocity'].append(vel)
        self.trajectory_data[task_name]['torque'].append(torque)

        # Context
        self.context_data[task_name]['start_pos'].append(self.start_pos)
        self.context_data[task_name]['target_pos'].append(self.target_pos)

    def reset_robot(self, tstart=0., t=3):
        # home position
        while self.realtime - tstart < t:
            self.realtime = self.world.getWorldTime()
            server.integrateWorldThreadSafe()
            self.robot.setPdTarget([0, -0.785, 0, -2.356, 0, 1.65806, 0.7853, 0.014, 0.014],
                                   vel_targets=np.zeros([9]))
            if self.isRealTime:
                time.sleep(self.world.getTimeStep())
        self.close_gripper()
        self.prev_dq_ref = np.zeros((7, 1))
        self.dq_ref = np.zeros((7, 1))
        self.prev_q_ref = np.zeros((7, 1))
        self.q_ref = np.expand_dims(self.robot.getGeneralizedCoordinate()[:-2], axis=1)

    def spawn_objects(self):
        # Objects
        self.table: raisim.Box = self.world.addBox(x=0.8, y=0.8, z=0.2, mass=10, material='default')
        table_pos = [0.5, 0, 0.1]
        self.table.setPosition(table_pos)
        self.table.setName("table")
        self.cube: raisim.Box = self.world.addBox(x=0.04, y=0.04, z=0.03, mass=0.001, material='default')
        self.cube.setName("cube")
        self.cube.setAppearance("red")

    def next_ep(self):
        # TODO : add a condition to select all points and return a boolean value for seeing all points.
        self.pick_pos = self.pick_points[self.grid_1_i]
        self.cube.setPosition(self.pick_pos.copy())
        self.place_pos = self.place_points[self.grid_2_i]

        # Update index
        if self.grid_2_i == 15:
            if self.grid_1_i < 15:
                self.grid_1_i += 1
            self.grid_2_i = 0
        else:
            self.grid_2_i += 1

    def get_target_pos(self, cube_pos):
        offset = np.asarray(self.robot.getFramePosition(7)) - np.asarray(self.robot.getFramePosition(8))
        pos_end = cube_pos
        pos_end[0] += offset[0]  # x-axis offset
        # pos_end[1] += offset[1]  # y-axis offset
        pos_end[2] += offset[2]  # z-axis offset
        return pos_end

    def wait(self):
        # if np.linalg.norm(self.target_pos - self.robot.getFramePosition(7)) <= 0.001:
        #     print("Reached target pos")
        #     return True
        if self.iterations >= (self.t_end - self.t_start) * 1000:
            return True
        else:
            return False

    def reach(self):
        if self.set_targets:
            self.open_gripper()
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.get_target_pos(self.cube.getPosition())
            self.target_pos[2] += 0.06
            self.t_start = self.realtime
            self.t_end = self.t_start + 4
            self.set_targets = False
        if self.wait():
            self.reset_done = False
            self.reached = True
            self.iterations = 0
            self.set_targets = True
        else:
            self.iterations += 1

    def pick(self):
        if self.set_targets:
            self.close_gripper()
            self.t_start = self.realtime
            self.t_end = self.t_start + 4
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.start_pos.copy()
            self.target_pos[2] += 0.05
            self.set_targets = False
        if self.wait():
            self.picked = True
            self.reached = False
            self.set_targets = True
            self.iterations = 0
        else:
            self.iterations += 1

    def carry(self):
        if self.set_targets:
            self.t_start = self.realtime
            self.t_end = self.t_start + 4
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.get_target_pos(self.place_pos)
            self.target_pos[2] = self.start_pos[2]
            self.set_targets = False
        if self.wait():
            self.carried = True
            self.picked = False
            self.set_targets = True
            self.iterations = 0
        else:
            self.iterations += 1

    def place(self):
        if self.set_targets:
            self.t_start = self.realtime
            self.t_end = self.t_start + 4
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.start_pos.copy()
            self.target_pos[2] -= 0.05
            self.set_targets = False
        if self.wait():
            self.placed = True
            self.carried = False
            self.set_targets = True
            self.iterations = 0
        else:
            self.iterations += 1

    def release(self):
        if self.set_targets:
            self.open_gripper()
            self.t_start = self.realtime
            self.t_end = self.t_start + 1
            self.start_pos = self.robot.getFramePosition(7)
            self.target_pos = self.start_pos.copy()
            self.target_pos[2] += 0.05
            self.set_targets = False
        if self.wait():
            self.released = True
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
            self.carry()

        elif self.carried:
            self.place()

        elif self.placed:
            done = self.release()

        jac_res, twist_res = self.trajectory_planning(real_time=self.realtime, t_start=self.t_start, t_end=self.t_end,
                                                      timestep=self.timestep,
                                                      pos_start=self.start_pos, pos_end=self.target_pos,
                                                      euler_start=self.euler, euler_end=self.euler)

        self.control(jac_res, twist_res)
        if self.step_counter % 100 == 0:
            # Record reach traj until reached
            if self.reset_done:
                self.record_data(task_name="reach")
            # Record pick traj until picked
            elif self.reached:
                self.record_data(task_name="pick")
            # Record carry traj until carried
            elif self.picked:
                self.record_data(task_name="carry")
            # Record place traj until placed
            elif self.carried:
                self.record_data(task_name="place")

        self.step_counter += 1

        return done

    def control(self, jac, twist):
        self.prev_dq_ref = self.dq_ref.copy()
        self.dq_ref = np.linalg.lstsq(jac, twist, rcond=None)[0]
        self.prev_q_ref = self.q_ref.copy()
        self.q_ref = np.add(np.add(self.prev_dq_ref.copy(), self.dq_ref.copy()) * self.world.getTimeStep() * 0.5,
                            self.prev_q_ref.copy())  # Integral
        self.robot.setPdTarget(pos_targets=np.hstack((self.q_ref.squeeze(), self.gripper_angles)),
                               vel_targets=np.hstack((self.dq_ref.squeeze(), [0., 0.])))

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

    def open_gripper(self):
        self.gripper_angles = [0.04, 0.04]

    def close_gripper(self):
        self.gripper_angles = [0.014, 0.014]

    def reset_data(self):
        self.trajectory_data = {'reach': {'pos': [], 'joint_angle': [], 'joint_velocity': [], 'torque': []},
                                'pick': {'pos': [], 'joint_angle': [], 'joint_velocity': [], 'torque': []},
                                'carry': {'pos': [], 'joint_angle': [], 'joint_velocity': [], 'torque': []},
                                'place': {'pos': [], 'joint_angle': [], 'joint_velocity': [], 'torque': []}}
        self.context_data = {'reach': {'start_pos': [], 'target_pos': []},
                             'pick': {'start_pos': [], 'target_pos': []},
                             'carry': {'start_pos': [], 'target_pos': []},
                             'place': {'start_pos': [], 'target_pos': []}}

    def rot2eul(self, rot):
        beta = -np.arcsin(rot[2, 0])
        alpha = np.arctan2(rot[2, 1] / np.cos(beta), rot[2, 2] / np.cos(beta))
        gamma = np.arctan2(rot[1, 0] / np.cos(beta), rot[0, 0] / np.cos(beta))
        return np.array((alpha, beta, gamma))


if __name__ == '__main__':
    REAL_TIME = False
    dt = 0.001
    episode_no = 0
    env = Environment(timestep=dt)
    env.grid_1_i = 12
    env.isRealTime = REAL_TIME
    server = raisim.RaisimServer(env.world)
    server.launchServer(8080)
    env.reset_robot()
    env.spawn_objects()
    env.create_grid()

    rot_mat = env.robot.getFrameOrientation(7)
    env.euler = env.rot2eul(rot_mat)
    Trajectories = []
    Context = []

    env.next_ep()

    try:
        while episode_no < 64:
            env.realtime = env.world.getWorldTime()
            server.integrateWorldThreadSafe()
            ep_done = env.step()
            if ep_done:
                # Reset to home position
                env.reset_robot(tstart=env.realtime)
                env.released = False
                env.reset_done = True
                print("Episode ", episode_no + 1, "Finished.")

                # Next episode settings
                env.next_ep()
                episode_no += 1
                Trajectories.append(env.trajectory_data)
                Context.append(env.context_data)
                env.reset_data()
                continue
            if env.isRealTime:
                time.sleep(dt)
        np.save('traj3.npy', Trajectories)
        np.save('context3.npy', Context)
        server.killServer()
    except:
        print("Something went wrong in episode {} !!!".format(episode_no + 1))
        np.save('traj3.npy', Trajectories)
        np.save('context3.npy', Context)
        server.killServer()
