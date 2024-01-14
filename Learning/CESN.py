import math

import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import visualizations

class CESN():
    def __init__(self):
        self.range = 7
        self.n_timestep = 160
        self.input_size = 3
        self.output_size = 3 + 3 * self.range
        self.reservoir_size = 6000

        self.reservoir_scale = 0.95 * (100.0 / self.reservoir_size)
        self.input_scale = 1
        # self.input_scale = 0.2 / self.input_size
        # self.bias_scale = 1
        self.alpha = 0.8

        self.W_in = np.random.rand(self.reservoir_size, self.input_size + 1) - 0.5
        self.W_x = np.random.uniform(low=-1.0, high=1.0, size=(self.reservoir_size, self.reservoir_size)) - 0.5
        # self.W_out = np.zeros((self.output_size, self.reservoir_size + self.input_size + 1), dtype=np.float64)
        self.initial_x = np.ones((self.reservoir_size, 1)) * 0.5
        self.reservoir_states = np.zeros((self.reservoir_size + self.input_size + 1, self.n_timestep))

        self.W_in *= self.input_scale
        self.W_x *= self.reservoir_scale
        self.W_out = Parameter(torch.randn(1 + self.input_size + self.reservoir_size, 1))

    def dict_to_array(self, dict_data, range, first_key):
        dim = 3 + 3 * range
        dataset = np.zeros((256, 160, dim))

        keys = dict_data[0].keys()
        for traj_idx, traj in enumerate(dict_data):
            episode = np.zeros((160, dim))

            i = 0
            for key in keys:
                if key == first_key:

                    item = np.array(traj[key])
                    episode[:, i:i + item.shape[1]] = item
                    i += item.shape[1]
                else:
                    item = np.array(traj[key])
                    item = item[:, 0:range]
                    episode[:, i:i + item.shape[1]] = item
                    i += item.shape[1]
            dataset[traj_idx] = episode
        print(dataset.shape)
        return dataset

    def fix_timestep(self, data):
        target_length = 160
        keys = data[0].keys()
        for traj in data:
            for key in keys:
                array = traj.get(key, [])  
                
                if len(array) > target_length:

                    traj[key] = array[:target_length]

    def load_data(self):
        trajectory_data = np.load('traj.npy', allow_pickle=True)
        Target_data = np.load('target_pos.npy', allow_pickle=True)

        self.fix_timestep(trajectory_data)
        self.fix_timestep(Target_data)

        output = self.dict_to_array(trajectory_data, self.range, 'pos')
        context = self.dict_to_array(Target_data, 0, 'target_pos')

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
        X = Xt
        Xt = torch.from_numpy(Xt)
        Yt = torch.from_numpy(Yt)


        if mode == "ridge":
            print("Obtaining Wout in Ridge mode:")
            XtT = Xt.t()
            XtXtT = torch.matmul(Xt, XtT)
            identity_term = beta * torch.eye(1 + self.input_size + self.reservoir_size)
            self.W_out = torch.linalg.solve(XtXtT + identity_term, torch.matmul(Xt, Yt.t())).t()


        return self.W_out

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
            Y_train_truth[eps] = Y

        print("X-All shape", X_all.shape)
        self.W_out = self.train_readout(X_all, Y_all, mode='ridge')
        predicted_output = np.matmul(self.W_out, X_all).T


        for eps in range(context.shape[0]):
            Y_train_pred[eps] = predicted_output[eps*self.n_timestep:(eps+1)*self.n_timestep, :]

        visualizations.visualize_truth_vs_predicted_trajectory(context,Y_train_truth, Y_train_pred)
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
            pred = np.matmul(self.W_out, Xu).T
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
