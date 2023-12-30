import math
import os
import numpy as np
import raisimpy as raisim
import time
from environment.Franka_Jac_Calculation import Franka_Jacobian
from FuncPoly5th import FuncPoly5th


class Environment:
    def __init__(self, robot_path='/raisimLib/rsc/Panda/panda.urdf',
                 raisim_act_path="/../raisimLib/rsc/activation.raisim", timestep=0.001):
        raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + raisim_act_path)
        self.world = raisim.World()
        self.world.setTimeStep(timestep)
        self.world.addGround()
        robot_urdf_file = os.environ['WORKSPACE'] + robot_path
        self.robot: raisim.ArticulatedSystem = self.world.addArticulatedSystem(robot_urdf_file)
        self.trajectory_data = {'pos': [], 'vel': [], 'torque': []}

        # Objects
        self.table: raisim.Box = self.world.addBox(x=1, y=1, z=0.3, mass=10, material='default', collision_group=1)
        table_pos = [0.7, 0, 0.15]
        self.table.setPosition(table_pos)
        self.table.setName("table")

        self.cube: raisim.Box = self.world.addBox(x=0.05, y=0.05, z=0.05, mass=0.5, material='default',
                                                  collision_group=2)
        cube_pos = [0.5, -0.2, 0.325]
        self.cube.setPosition(cube_pos)
        self.cube.setName("cube")
        self.cube.setAppearance("red")

        # TODO: fix gains!!!
        self.p_gain_ref = np.array([200, 500, 200, 500, 200, 200, 200, 0.1, 0.1])
        self.d_gain_ref = np.array([10, 50, 10, 50, 10, 10, 10, 0.01, 0.01])
        self.p_gain = np.array([200, 1000, 200, 2000, 200, 200, 200, 0.1, 0.1])
        self.d_gain = np.array([10, 50, 10, 50, 10, 10, 10, 0.01, 0.01])

        # q_ref, prev_q_ref, dq_ref, prev_dq_ref
        self.prev_dq_ref = np.zeros((7, 1))
        self.dq_ref = np.zeros((7, 1))
        self.prev_q_ref = np.zeros((7, 1))
        self.q_ref = np.expand_dims(np.array([0, -0.785, 0, -2.356, 0, 1.5708, 0.7853]), axis=1)

    def record_data(self):
        # time = self.world.getSimulationTime()self.
        pos = self.robot.getGeneralizedCoordinate()
        vel = self.robot.getGeneralizedVelocity()
        torque = self.robot.getGeneralizedForce()

        # self.trajectory_data['time'].append(time)
        self.trajectory_data['pos'].append(pos)
        self.trajectory_data['vel'].append(vel)
        self.trajectory_data['torque'].append(torque)

    def reset_robot(self):
        # initial position
        angle = np.array([0, -0.785, 0, -2.356, 0, 1.5708, 0.7853, 0, 0, 0.1,
                          0.1])  # Last 2 joints are prismatic. Hand joint and joint 8 are fixed.
        self.robot.setGeneralizedCoordinate(angle[:-2])
        self.robot.setPdGains(self.p_gain_ref, self.d_gain_ref)
        self.robot.setPdTarget(angle[:-2], np.zeros([9]))
        return self.robot.getFramePosition(11)  # return hand position

    def get_pick_pos(self):
        cube_pos = self.cube.getPosition()
        pos_end = cube_pos
        # pos_end[0] += 0.01     # x-axis offset
        pos_end[1] += np.sign(pos_end[1]) * 0.02  # y-axis offset
        pos_end[2] += 0.14  # z-axis offset
        return pos_end

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
        q = env.robot.getGeneralizedCoordinate()
        jac = Franka_Jacobian(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        return jac, twist

    def set_force(self, target_joint_angle, target_joint_vel, p_gain, d_gain):
        joint_angle_cur = self.robot.getGeneralizedCoordinate()
        joint_vel_cur = self.robot.getGeneralizedVelocity()
        joint_angle_err = np.hstack((target_joint_angle.squeeze(), [0, 0])) - joint_angle_cur
        joint_vel_err = np.hstack((target_joint_vel.squeeze(), [0, 0])) - joint_vel_cur

        tau = p_gain * joint_angle_err + d_gain * joint_vel_err
        self.robot.setGeneralizedForce(tau * 10)
        self.record_data()

    def open_gripper(self):
        angles = self.robot.getGeneralizedCoordinate()
        angles[-2:] = [0.035, 0.035]
        self.robot.setGeneralizedCoordinate(angles)

    def close_gripper(self):
        angles = self.robot.getGeneralizedCoordinate()
        angles[-2:] = [0., 0.]
        self.robot.setGeneralizedCoordinate(angles)


if __name__ == '__main__':
    dt = 0.001
    env = Environment(timestep=dt)

    server = raisim.RaisimServer(env.world)
    server.launchServer(8080)
    pos_start_ = env.reset_robot()
    pos_end_ = env.get_pick_pos()
    euler_start_ = [0., 0., 0.]
    # euler_end_ = [2, 2, 0.2]
    ts = 1
    reach_time = 5
    te = ts + reach_time

    while True:
        rt = env.world.getWorldTime()
        server.integrateWorldThreadSafe()
        jac_res, twist_res = env.trajectory_planning(real_time=rt, t_start=ts, t_end=te, timestep=dt,
                                                     pos_start=pos_start_, pos_end=pos_end_,
                                                     euler_start=euler_start_, euler_end=euler_start_)
        env.control(jac_res, twist_res)

        time.sleep(dt)
    # np.save('traj.npy', env.trajectory_data)
    # server.killServer()
