import math
import os
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
        self.current_pos = np.zeros(3)
        self.target_pos = np.zeros(3)
        self.start_pos = np.zeros(3)

        self.iterations = 0
        self.t_start = 0.1
        self.t_end = 0

        self.realtime = 0
        self.trajectory_data = {'pos': [], 'vel': [], 'torque': []}

        # TODO: fix gains!!!
        self.p_gain_ref = np.array([200, 500, 200, 500, 200, 200, 200, 200, 200])
        self.d_gain_ref = np.array([10, 50, 10, 50, 10, 10, 10, 0.01, 0.01])
        self.p_gain = np.array([200, 1000, 200, 2000, 200, 200, 200, 100, 100])
        self.d_gain = np.array([10, 50, 10, 50, 10, 10, 10, 0.01, 0.01])

        # q_ref, prev_q_ref, dq_ref, prev_dq_ref
        self.prev_dq_ref = np.zeros((7, 1))
        self.dq_ref = np.zeros((7, 1))
        self.prev_q_ref = np.zeros((7, 1))
        self.q_ref = np.expand_dims(np.array([0, -0.785, 0, -2.356, 0, 1.5708, 0.7853]), axis=1)

        self.gripper_angles = [0., 0.]
        self.i = 0.1

    def record_data(self):
        # time = self.world.getSimulationTime()self.
        pos = self.robot.getFramePosition(11)
        angle = self.robot.getGeneralizedCoordinate()
        vel = self.robot.getGeneralizedVelocity()
        torque = self.robot.getGeneralizedForce()

        # self.trajectory_data['time'].append(time)
        self.trajectory_data['pos'].append(pos)
        self.trajectory_data['angle'].append(angle)
        self.trajectory_data['vel'].append(vel)
        self.trajectory_data['torque'].append(torque)

    def reset_robot(self):
        # initial position
        q = np.expand_dims([0, -0.785, 0, -2.356, 0, 1.5708, 0.7853], axis=1)
        vel = np.expand_dims(np.zeros(7), axis=1)
        self.robot.setPdGains(self.p_gain_ref, self.d_gain_ref)
        self.robot.setPdTarget([0, -0.785, 0, -2.356, 0, 1.5708, 0.7853, 0., 0.], np.zeros([9]))

        while self.realtime < 3.:
            self.realtime = self.world.getWorldTime()
            server.integrateWorldThreadSafe()
            self.set_force(target_joint_angle=q, target_joint_vel=vel, p_gain=self.p_gain, d_gain=self.d_gain)
            time.sleep(dt)

        self.place_pos = [0.7, -0.2, 0.215]
        self.spawn_objects()

    def spawn_objects(self):
        # Objects
        self.table: raisim.Box = self.world.addBox(x=0.8, y=0.8, z=0.2, mass=10, material='default')
        table_pos = [0.7, 0, 0.1]
        self.table.setPosition(table_pos)
        self.table.setName("table")

        self.cube: raisim.Box = self.world.addBox(x=0.04, y=0.04, z=0.03, mass=0.001, material='default')
        cube_pos = [0.5, -0.2, 0.215]
        self.cube.setPosition(cube_pos)
        self.cube.setName("cube")
        self.cube.setAppearance("red")

    def next_ep(self):
        # TODO: new episode settings
        # self.cube.setPosition([0.5+self.i, -0.2, 0.215])  # pick position
        # self.place_pos = [0.7, -0.2+self.i, 0.215]  # place position
        # self.i += 0.1
        self.cube.setPosition([0.5, -0.2, 0.215])  # pick position
        self.place_pos = [0.7, -0.2, 0.215]  # place position

    def get_target_pos(self, cube_pos):
        pos_end = cube_pos
        pos_end[0] += 0.02  # x-axis offset
        pos_end[1] += np.sign(pos_end[1]) * 0.025  # y-axis offset
        pos_end[2] += 0.12  # z-axis offset
        return pos_end

    def wait(self, target_pos):
        if np.linalg.norm(target_pos - self.robot.getFramePosition(11)) <= 0.01 or self.iterations >= (
                self.t_end - self.t_start) * 1000:
            return True
        else:
            return False

    def reach(self):
        if self.set_targets:
            self.home_pos = self.robot.getFramePosition(11)
            self.start_pos = self.home_pos.copy()
            self.target_pos = self.get_target_pos(self.cube.getPosition())
            self.t_start = np.ceil(self.realtime)
            self.t_end = self.t_start + 7
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
            self.start_pos = self.robot.getFramePosition(11)
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
            self.start_pos = self.robot.getFramePosition(11)
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
            self.start_pos = self.robot.getFramePosition(11)
            self.target_pos = self.get_target_pos(self.place_pos)
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
            self.start_pos = self.robot.getFramePosition(11)
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
            self.start_pos = self.robot.getFramePosition(11)
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
            self.start_pos = self.robot.getFramePosition(11)
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
        euler_start_ = [0., 0., 0.]
        euler_end_ = [0., 0., 0.]

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
                                                      euler_start=euler_start_, euler_end=euler_end_)

        self.control(jac_res, twist_res)

        return done

    def control(self, jac, twist):
        self.prev_dq_ref = self.dq_ref
        self.dq_ref = np.linalg.lstsq(jac, twist, rcond=None)[0]
        self.prev_q_ref = self.q_ref
        self.q_ref = np.add(np.add(self.prev_dq_ref, self.dq_ref) * dt * 0.5, self.prev_q_ref)  # Integral
        self.set_force(target_joint_angle=self.q_ref, target_joint_vel=self.dq_ref,
                       p_gain=self.p_gain, d_gain=self.d_gain)

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

    def set_force(self, target_joint_angle, target_joint_vel, p_gain, d_gain):
        joint_angle_cur = self.robot.getGeneralizedCoordinate()
        joint_vel_cur = self.robot.getGeneralizedVelocity()
        joint_angle_err = np.hstack((target_joint_angle.squeeze(), self.gripper_angles)) - joint_angle_cur
        joint_vel_err = np.hstack((target_joint_vel.squeeze(), [0, 0])) - joint_vel_cur

        tau = p_gain * joint_angle_err + d_gain * joint_vel_err
        self.robot.setGeneralizedForce(tau * 10)
        # self.record_data()

    def open_gripper(self):
        self.gripper_angles = [0.04, 0.04]

    def close_gripper(self):
        self.gripper_angles = [0.012, 0.012]


if __name__ == '__main__':
    dt = 0.001
    env = Environment(timestep=dt)
    server = raisim.RaisimServer(env.world)
    server.launchServer(8080)

    env.reset_robot()

    while True:
        env.realtime = env.world.getWorldTime()
        server.integrateWorldThreadSafe()

        ep_done = env.step()
        if ep_done:
            env.next_ep()
        time.sleep(dt)

    # server.killServer()
