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


def dict_to_array(dict_data, n, first_key):
    dim = 3 + 3 * n
    dataset = np.zeros((256, n_timestep, dim))

    keys = dict_data[0].keys()
    for traj_idx, traj in enumerate(dict_data):
        episode = np.zeros((n_timestep, dim))
        i = 0
        for key in keys:
            if key == first_key:
                item = np.array(traj[key])
                episode[:, i:i + item.shape[1]] = item
                i += item.shape[1]
            else:
                item = np.array(traj[key])
                item = item[:, :n]
                episode[:, i:i + item.shape[1]] = item
                i += item.shape[1]
        dataset[traj_idx] = episode
    print(dataset.shape)
    return dataset


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

    reach_traj = dict_to_array(t1t, 7, 'pos')
    reach_context = dict_to_array(t1c, 1, 'start_pos')
    pick_traj = dict_to_array(t2t, 7, 'pos')
    pick_context = dict_to_array(t2c, 1, 'start_pos')
    carry_traj = dict_to_array(t3t, 7, 'pos')
    carry_context = dict_to_array(t3c, 1, 'start_pos')
    place_traj = dict_to_array(t4t, 7, 'pos')
    place_context = dict_to_array(t4c, 1, 'start_pos')
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
