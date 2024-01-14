import numpy as np

n_timestep = 40
n_joints = 7


def merge_data():
    traj1 = np.load("../environment/traj.npy", allow_pickle=True)
    context1 = np.load("../environment/context.npy", allow_pickle=True)
    traj2 = np.load("../environment/traj1.npy", allow_pickle=True)
    context2 = np.load("../environment/context1.npy", allow_pickle=True)
    traj3 = np.load("../environment/traj2.npy", allow_pickle=True)
    context3 = np.load("../environment/context2.npy", allow_pickle=True)
    traj4 = np.load("../environment/traj3.npy", allow_pickle=True)
    context4 = np.load("../environment/context3.npy", allow_pickle=True)
    trajectories = np.hstack((traj1, traj2, traj3, traj4))
    contexts = np.hstack((context1, context2, context3, context4))
    np.save("trajectories.npy", trajectories)
    np.save("contexts.npy", contexts)


def create_task_data(traj, context, key):
    # Extract key-value pairs into a new array of dictionaries
    task_traj_data = [d[key] for d in traj]

    # Extract key-value pairs into a new array of dictionaries
    task_context_data = [d[key] for d in context]

    return task_traj_data, task_context_data


def fix_timestep(data):
    keys = data[0].keys()
    for traj in data:
        for key in keys:
            array = traj.get(key, [])
            if len(array) > n_timestep:
                traj[key] = array[:n_timestep]
    return data


def load_data():
    trajectory_data = np.load('trajectories.npy', allow_pickle=True)
    context_data = np.load('contexts.npy', allow_pickle=True)

    t1t, t1c = create_task_data(trajectory_data, context_data, key="reach")
    t2t, t2c = create_task_data(trajectory_data, context_data, key="pick")
    t3t, t3c = create_task_data(trajectory_data, context_data, key="carry")
    t4t, t4c = create_task_data(trajectory_data, context_data, key="place")

    t1t = fix_timestep(t1t)
    t1c = fix_timestep(t1c)
    t2t = fix_timestep(t2t)
    t2c = fix_timestep(t2c)
    t3t = fix_timestep(t3t)
    t3c = fix_timestep(t3c)
    t4t = fix_timestep(t4t)
    t4c = fix_timestep(t4c)

    reach_traj = np.zeros((256, 40, 24))
    reach_context = np.zeros((256, 40, 6))
    pick_traj = np.zeros((256, 40, 24))
    pick_context = np.zeros((256, 40, 6))
    carry_traj = np.zeros((256, 40, 24))
    carry_context = np.zeros((256, 40, 6))
    place_traj = np.zeros((256, 40, 24))
    place_context = np.zeros((256, 40, 6))

    for i in range(256):
        reach_traj[i, :, 0:3] = t1t[i]["pos"]
        reach_traj[i, :, 3:10] = t1t[i]["joint_angle"]
        reach_traj[i, :, 10:17] = t1t[i]["joint_velocity"]
        reach_traj[i, :, 17:] = t1t[i]["torque"]

        reach_context[i, :, 0:3] = t1c[i]["start_pos"]
        reach_context[i, :, 3:] = t1c[i]["target_pos"]

        pick_traj[i, :, 0:3] = t2t[i]["pos"]
        pick_traj[i, :, 3:10] = t2t[i]["joint_angle"]
        pick_traj[i, :, 10:17] = t2t[i]["joint_velocity"]
        pick_traj[i, :, 17:] = t2t[i]["torque"]

        pick_context[i, :, 0:3] = t2c[i]["start_pos"]
        pick_context[i, :, 3:] = t2c[i]["target_pos"]

        carry_traj[i, :, 0:3] = t3t[i]["pos"]
        carry_traj[i, :, 3:10] = t3t[i]["joint_angle"]
        carry_traj[i, :, 10:17] = t3t[i]["joint_velocity"]
        carry_traj[i, :, 17:] = t3t[i]["torque"]

        carry_context[i, :, 0:3] = t3c[i]["start_pos"]
        carry_context[i, :, 3:] = t3c[i]["target_pos"]

        place_traj[i, :, 0:3] = t4t[i]["pos"]
        place_traj[i, :, 3:10] = t4t[i]["joint_angle"]
        place_traj[i, :, 10:17] = t4t[i]["joint_velocity"]
        place_traj[i, :, 17:] = t4t[i]["torque"]

        place_context[i, :, 0:3] = t4c[i]["start_pos"]
        place_context[i, :, 3:] = t4c[i]["target_pos"]

    np.save("reach_traj.npy", reach_traj)
    np.save("reach_context.npy", reach_context)
    np.save("pick_traj.npy", pick_traj)
    np.save("pick_context.npy", pick_context)
    np.save("carry_traj.npy", carry_traj)
    np.save("carry_context.npy", carry_context)
    np.save("place_traj.npy", place_traj)
    np.save("place_context.npy", place_context)


if __name__ == '__main__':
    load_data()
