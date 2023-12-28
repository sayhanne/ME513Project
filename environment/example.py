# import os
# import numpy as np
# import raisimpy as raisim
# import time
# import math
#
# raisim.World.setLicenseFile(os.path.dirname(os.path.abspath(__file__)) + "/../raisimLib/rsc/activation.raisim")
# world = raisim.World()
# world.setTimeStep(0.001)
#
# # create objects
# # terrainProperties = raisim.TerrainProperties()
# # terrainProperties.frequency = 0.2
# # terrainProperties.zScale = 3.0
# # terrainProperties.xSize = 20.0
# # terrainProperties.ySize = 20.0
# # terrainProperties.xSamples = 50
# # terrainProperties.ySamples = 50
# # terrainProperties.fractalOctaves = 3
# # terrainProperties.fractalLacunarity = 2.0
# # terrainProperties.fractalGain = 0.25
# # hm = world.addHeightMap(0.0, 0.0, terrainProperties)
# world.addGround()
# robot
# robot_urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/../../rsc/Panda/panda.urdf"


# num_joints = robot.getGeneralizedCoordinateDim()
# joint_positions = robot.getGeneralizedCoordinate()
# joint_velocities = robot.getGeneralizedVelocity()
# robot.setPdGains(8*np.ones([16]),np.ones([16]))

# robot.setGeneralizedCoordinate(np.array([0, 0.307, 0, -2.7, 0, 3, -2.356, 0, 0, 0.01, 0.01, 0, 0, 0, 0, 0]))
# num_joints = robot.getGeneralizedCoordinateDim()
# joint_positions = robot.getGeneralizedCoordinate()
# joint_velocities = robot.getGeneralizedVelocity()
# print("Number of Joints:", num_joints)
# print("Joint Positions:", joint_positions)
# print("Joint Velocities:", joint_velocities)

# # ANYmal joint PD controller
# anymal_nominal_joint_config = np.array([0,0.307,0,-2.7,0,3,-2.356, 0.01, 0.01])
# anymal.setGeneralizedCoordinate(anymal_nominal_joint_config)
# anymal.setPdGains(200*np.ones([7]), np.ones([7]))
# anymal.setPdTarget(anymal_nominal_joint_config, np.zeros([7]))

# launch raisim server
# server = raisim.RaisimServer(world)
# server.launchServer(8080)
# time.sleep(0.001)

# q1_ref = 0
# q2_ref = -90 * math.pi / 180
# q3_ref = 0
# q4_ref = -90 * math.pi / 180
# q5_ref = 0 * math.pi / 180
# q6_ref = 179 * math.pi / 180
# q7_ref = 90 * math.pi / 180
#
# pos_ref = np.array([q1_ref, q2_ref, q3_ref,q4_ref, q5_ref, q6_ref,q7_ref,0,0])

# p_gain_1 = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0])
# d_gain_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
# p_gain_2 = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200, 0, 0])
# d_gain_2 = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0])
# pos_ref = np.array([0, 0.307, 0, -2.7, 0, 3, -2.356, 0, 0, 0.01, 0.01])
# vel_ref = np.zeros([11])
#
# robot.setGeneralizedCoordinate(pos_ref[:-2])
# robot.setPdGains(p_gain_1[:-2],  d_gain_1[:-2])
# robot.setPdTarget(pos_ref[:-2], np.zeros([9]))
#
# for i in range(500000):
#
#     server.integrateWorldThreadSafe()
#     time.sleep(0.0005)
#     pos_ref_2 = np.array([i/1000, 0, 0, -2.7, 0, 3, -2.356, 0, 0, 0.01, 0.01])
#
#     pos_mea = robot.getGeneralizedCoordinate()
#     vel_mea = robot.getGeneralizedVelocity()
#
#     position_err = pos_ref_2[:-2] - pos_mea
#     velocity_err = vel_ref[:-2] - vel_mea
#
#     tau = p_gain_2[:-2] * position_err + d_gain_2[:-2] * velocity_err
#     # print(tau)
#     robot.setGeneralizedForce(tau*100)
#
# # server.killServer()




