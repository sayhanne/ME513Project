import math

import numpy as np
from scipy import linalg


class CESN():
    def __init__(self):
        self.n_timestep = 160
        self.input_size = 3
        self.output_size = 30
        self.reservoir_size = 3000

        self.reservoir_scale = 0.7 * (100.0 / self.reservoir_size)
        self.input_scale = 1
        # self.input_scale = 0.2 / self.input_size
        # self.bias_scale = 1
        self.alpha = 0.7

        self.W_in = np.random.rand(self.reservoir_size, self.input_size + 1) - 0.5
        self.W_x = np.random.uniform(low=-1.0, high=1.0, size=(self.reservoir_size, self.reservoir_size)) - 0.5
        self.W_out = np.zeros((self.output_size, self.reservoir_size + self.input_size + 1), dtype=np.float64)
        self.initial_x = np.ones((self.reservoir_size, 1)) * 0.5
        self.reservoir_states = np.zeros((self.reservoir_size + self.input_size + 1, self.n_timestep))

        self.W_in *= self.input_scale
        self.W_x *= self.reservoir_scale
        pass
    
    def dict_to_array(self, dict_data, dim):
        dataset = np.zeros((256, 160, dim))
        
        keys = dict_data[0].keys()
        for traj_idx, traj in enumerate(dict_data):
            episode = np.zeros((160, dim))
            i = 0
            for key in keys:
                item = np.array(traj[key])
                episode[:, i:i + item.shape[1]] = item
                i += item.shape[1]
            dataset[traj_idx] = episode

        return dataset
    
    def fix_timestep(self, data):
        target_length = 160
        keys = data[0].keys()
        for traj in data:
            for key in keys :
                array = traj.get(key, [])  
                
                if len(array) > target_length:

                    traj[key] = array[:target_length]

    def load_data(self):
        trajectory_data = np.load('traj.npy' , allow_pickle=True)
        Target_data = np.load('target_pos.npy' , allow_pickle=True)

        self.fix_timestep(trajectory_data)
        self.fix_timestep(Target_data)

        output = self.dict_to_array(trajectory_data, 30)
        context = self.dict_to_array(Target_data, 3)

        return context, output
        

    def normalize_spectral_radius(self):
        spectral_radius = max(abs(linalg.eig(self.W_x)[0]))
        if spectral_radius >= 1:
            self.W_x = self.W_x / spectral_radius
        spectral_radius = max(abs(linalg.eig(self.W_x)[0]))
        return spectral_radius

    def get_reservoir_states(self, con):
        X = np.zeros((self.reservoir_size + self.input_size + 1, self.n_timestep))
        x = self.initial_x
        for t in range(self.n_timestep):
            u = con[t]

            x_new = np.tanh(
                np.matmul(self.W_in, np.hstack((1, u)).reshape(self.input_size + 1, 1)) + np.matmul(self.W_x, x))
            x = (1 - self.alpha) * x + self.alpha * x_new

            u = np.zeros((self.input_size))
            X[:, t] = np.hstack((1, u, x.flatten()))
        return X

    def train_readout(self, Xt, Yt, mode):
        Yt = Yt.T
        beta = 1e-8
        if mode == "ridge":
            print("Obtaining Wout in Ridge mode:")
            self.Wout = linalg.solve(np.matmul(Xt, Xt.T) + beta * np.eye(1 + self.input_size + self.reservoir_size),
                                     np.matmul(Xt, Yt.T)).T
            # self.Wout = np.linalg.lstsq(np.matmul(Xt, Xt.T) + beta * np.eye(1 + self.input_size + self.reservoir_size),
            #                             np.matmul(Xt, Yt.T).T, rcond=None)[0]

        return self.Wout

    def train(self, context, output):
        n_train_episodes = output.shape[0]
        n_timestep_per_eps = output.shape[1]
        sample_index = np.linspace(0, n_timestep_per_eps, self.n_timestep, endpoint=False, dtype=np.int64)

        X_all = np.empty((self.reservoir_size + self.input_size + 1, n_train_episodes * self.n_timestep))
        Y_all = np.empty((n_train_episodes * self.n_timestep, self.output_size))
        Y_train_pred = np.empty((context.shape[0], self.n_timestep, self.output_size))
        Y_train_truth = np.empty((context.shape[0], self.n_timestep, self.output_size))

        for eps in range(n_train_episodes):
            Y = output[eps, sample_index]
            con = context[eps, sample_index]
            X_con = self.get_reservoir_states(con)

            X_all[:, eps * (self.n_timestep):(eps + 1) * (self.n_timestep)] = X_con
            Y_all[eps * self.n_timestep: (eps + 1) * (self.n_timestep), :] = Y
            Y_train_truth = Y

        self.W_out = self.train_readout(X_all, Y_all, mode='ridge')
        predicted_output = np.matmul(self.W_out, X_all).T

        for eps in range(context.shape[0]):
            Y_train_pred[eps] = predicted_output[eps*self.n_timestep:(eps+1)*self.n_timestep,:]

        return math.sqrt(self.MSE(Y_train_truth, Y_train_pred)), self.NRMSE(Y_train_truth, Y_train_pred)

    def predict(self, context):

        n_timestep_per_eps = context.shape[1]
        context_dim = context.shape[-1]
        sample_index = np.linspace(0, n_timestep_per_eps, self.n_timestep, endpoint=False, dtype=np.int64)
        Y_test_pred = np.empty((context.shape[0], self.n_timestep, self.output_size))

        for k in range(context.shape[0]):

            con = context[k, sample_index].reshape(self.n_timestep, context_dim)
            u = con
            Xu = self.get_reservoir_states(u)
            pred = np.matmul(self.Wout, Xu).T
            Y_test_pred[k] = pred

        return Y_test_pred

    def test(self, context, output):


        Y_test_pred = np.empty((context.shape[0], self.n_timestep, self.output_size))
        Y_test_truth = np.empty((context.shape[0], self.n_timestep, self.output_size))

        for eps in range(context.shape[0]):
            cont = np.zeros((1, context.shape[1], 3)) + context[eps,:,:]
            Y = output[eps, :, :]
            predicted_output = self.predict(cont)
            Y_test_truth[eps, :, :] = Y
            Y_test_pred[eps, :, :] = predicted_output

        return math.sqrt(self.MSE(Y_test_truth, Y_test_pred)), self.NRMSE(Y_test_truth, Y_test_pred)

    def MSE(self, truth, estimated):
        MSE_per_dim = []
        for k in range(truth.shape[0]):
            MSE = np.square(np.subtract(truth[k], estimated[k])).mean()
            MSE_per_dim.append(MSE)

        MSE = np.square(np.subtract(truth, estimated)).mean()

        return MSE
    def NRMSE(self, truth, estimated):
        MSE = self.MSE(truth, estimated)
        RMSE = math.sqrt(MSE)
        std = np.std(truth)
        NRMSE = RMSE / std

        return NRMSE

    def save_weights(self,filename):
        weights_to_save = {
            'Recurrent_reservoir_weights': self.W_x,
            'input_weights': self.W_in,
            'output_weights': self.W_out,
            'Reservoir_States': self.reservoir_states
        }

        np.save(filename, **weights_to_save)

    def load_weights(self,filename):
        loaded_weights = np.load(filename)

        self.W_x = loaded_weights['Recurrent_reservoir_weights']
        self.W_in = loaded_weights['input_weights']
        self.W_out = loaded_weights['output_weights']
        self.reservoir_states = loaded_weights['Reservoir_States']
