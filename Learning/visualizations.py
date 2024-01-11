import os.path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def visalize_trajectory(target_positions, truth_trajectory, path):

    for eps, trajectory in enumerate(truth_trajectory):

        target = target_positions[eps, :, :]
        end_effector_pos = trajectory[:, 0:3]

        end_effector_x, end_effector_y, end_effector_z = zip(*end_effector_pos)
        target_x, target_y, target_z = target[:, 0], target[:, 1], target[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(end_effector_x, end_effector_y, end_effector_z, c='blue')
        ax.scatter(target_x, target_y, target_z, c='red')

        file_path = os.path.join(path, f'trajectory{eps}.png')

        plt.title(f'trajectory{eps}.png')
        plt.savefig(file_path)
        plt.close()
    print('trajectories plots are completed.')

def visualize_truth_vs_predicted_trajectory(target_positions, truth_trajectory, estimated_trajectory, path):


        for eps, trajectory in enumerate(truth_trajectory):

            target = target_positions[eps, :, :]
            end_effector_pos_truth = trajectory[:, 0:3]
            end_effector_pos_pred = estimated_trajectory[eps, :, 0:3]

            end_effector_x, end_effector_y, end_effector_z = zip(*end_effector_pos_truth)
            end_effector_x_pred, end_effector_y_pred, end_effector_z_pred = zip(*end_effector_pos_pred)
            target_x, target_y, target_z = target[:, 0], target[:, 1], target[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(end_effector_x, end_effector_y, end_effector_z, c='blue')
            ax.plot(end_effector_x_pred, end_effector_y_pred, end_effector_z_pred, c='green')
            ax.scatter(target_x, target_y, target_z, c='red')

            file_path = os.path.join(path, f'trajectory{eps}.png')

            plt.title(f'trajectory{eps}.png')
            plt.savefig(file_path)
            plt.close()
        print('Training plots are completed.')

