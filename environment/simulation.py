import math
import os
import numpy as np
import raisimpy as raisim
import time
from Franka_Jac_Calculation import Franka_Jacobian
from FuncPoly5th import FuncPoly5th


class Environment:
    def __init__(self, robot_path='/raisimLib/rsc/Panda/panda_nogrip.urdf',
                 raisim_act_path="/../raisimLib/rsc/activation.raisim", timestep=0.001):
        raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + raisim_act_path)
        self.world = raisim.World()
        self.world.setTimeStep(timestep)
        self.world.addGround()
        robot_urdf_file = os.environ['WORKSPACE'] + robot_path
        self.robot: raisim.ArticulatedSystem = self.world.addArticulatedSystem(robot_urdf_file)
        self.trajectory_data = {'pos': [], 'vel': [], 'torque': []}

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
        # angle = np.array([0, 0.307, 0, -2.7, 0, 3, -2.356, 0, 0, 0.1, 0.1])  # Last 2 joints are prismatic.
        angle = np.array([0, -0.785, 0, -2.356, 0 , 1.5708,  1.5708, 0, 0, 0.1, 0.1])
        print(env.robot.getFramePosition(11))
        self.robot.setGeneralizedCoordinate(angle[:-2])
        self.robot.setPdGains([200, 200, 200, 200, 200, 200, 200, 0.1, 0.1], [10, 10, 10, 10, 10, 10, 10, 0.01, 0.01])
        self.robot.setPdTarget(angle[:-2], np.zeros([9]))

    def spawn_objects(self):
        pass

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
        # TODO: angular velocity
        q = env.robot.getGeneralizedCoordinate()
        jac = Franka_Jacobian(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        return jac, twist

    def set_force(self, target_joint_angle, target_joint_vel, p_gain, d_gain):
        joint_angle_cur = self.robot.getGeneralizedCoordinate()
        joint_vel_cur = self.robot.getGeneralizedVelocity()
        joint_angle_err = np.hstack((target_joint_angle.squeeze(), [0, 0])) - joint_angle_cur
        joint_vel_err = np.hstack((target_joint_vel.squeeze(), [0, 0])) - joint_vel_cur

        tau = p_gain * joint_angle_err + d_gain * joint_vel_err
        self.robot.setGeneralizedForce(tau)
        self.record_data()


if __name__ == '__main__':
    dt = 0.001
    env = Environment(timestep=dt)
    server = raisim.RaisimServer(env.world)
    server.launchServer(8080)
    visBox = server.addVisualBox("t_box", 1, 1, 1, 1, 1, 1, 1)
    visBox.setBoxSize(1, 1 , 0.5)
    BoxPosition = [0.7 , 0 , 0.]
    visBox.setPosition(BoxPosition)

    ObjBox = server.addVisualBox("o_box", 1, 1, 1, 1, 0., 0., 1)
    ObjBox.setBoxSize(0.05, 0.05, 0.05)
    # color = [0.1, 0.1, 0.1]
    # ObjBox.setColor()
    ObjBoxPosition = [0.7 , 0.1 , 0.275 ]
    ObjBox.setPosition(ObjBoxPosition)
    env.reset_robot()
    p_gain_ref = np.array([200, 200, 200, 200, 200, 200, 200, 0.1, 0.1])
    d_gain_ref = np.array([10, 10, 10, 10, 10, 10, 10, 0.01, 0.01])

    prev_dq_ref = np.zeros((7, 1))
    dq_ref = np.zeros((7, 1))
    prev_q_ref = np.zeros((7, 1))
    q_ref = np.expand_dims(np.array([0, 0.307, 0, -2.7, 0, 3, -2.356]), axis=1)
    # pos_start_ = [3.94668356e-01, 2.61303697e-16, 10.32223621e-01]
    pos_start_ = [8.8000000e-02, 5.5897103e-17 ,9.2600000e-01]
    # pos_end_ = [3.94668356e-01, 2.61303697e-16, 12.32223621e-01]
    pos_end_ = ObjBoxPosition
    euler_start_ = [0., 0., 0.]
    euler_end_ = [2, 2, 0.2]

    while True:
        rt = env.world.getWorldTime()
        server.integrateWorldThreadSafe()
        # print("Real time", rt)
        jac_res, twist_res = env.trajectory_planning(real_time=rt, t_start=3, t_end=7, timestep=dt,
                                                     pos_start=pos_start_,
                                                     pos_end=pos_end_ , euler_start=euler_start_ , euler_end= euler_end_ )
        prev_dq_ref = dq_ref
        dq_ref = np.linalg.lstsq(jac_res, twist_res, rcond=None)[0]
        prev_q_ref = q_ref
        q_ref = np.add(np.add(prev_dq_ref, dq_ref) * dt * 0.5, prev_q_ref)      # Integral
        env.set_force(target_joint_angle=q_ref, target_joint_vel=dq_ref,
                      p_gain=p_gain_ref, d_gain=d_gain_ref)
        time.sleep(dt)
    np.save('traj.npy', env.trajectory_data)
    server.killServer()
