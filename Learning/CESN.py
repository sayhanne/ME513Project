import numpy as np


class CESN(object):

    def __init__(self):
        self.n_iteration = 50
        self.context_dim =  3
        self.feedback_dim = 8
        self.in_dim = self.context_dim + self.feedback_dim
        self.out_dim = self.feedback_dim
        self.reservoir_size = 1500
        self.reservoir = np.zeros(())

    def NormalizeSpectralRadius(self):
        rhow = max(abs(linalg.eig(self.W)[0]))
        print("initial rho is :", rhow)
        if rhow >= 1:
            self.W = self.W / rhow
            # self.W *= 1.25 / rhow
        rhow = max(abs(linalg.eig(self.W)[0]))
        print("rho now is :", rhow)

    def GetX_no_feed(self, u_t):

        X = np.zeros((self.Nx + self.Nu + 1, self.N))
        x = self.x0
        for t in range(self.N):
            u = u_t[t]
            x_new = np.tanh(np.matmul(self.Win, np.hstack((1, u)).reshape(self.Nu + 1, 1)) + np.matmul(self.W, x))
            x = (1 - self.alpha) * x + self.alpha * x_new
            # x = x_new

            u = np.zeros((self.Nu))
            X[:, t] = np.hstack((1, u, x.flatten()))
        return X

    def Train_GetWout(self, Xt, Yt, mode):
        Yt = Yt.T
        if mode == "Ridge":
            beta = 1e-10
            print("Obtaining Wout in Ridge mode:")
            # self.Wout = linalg.solve( np.matmul(Xt,Xt.T) + beta * np.eye(1 + self.Nu + self.Nx) , np.matmul(Xt,Yt.T) ).T
            self.Wout = np.linalg.lstsq(np.matmul(Xt, Xt.T) + beta * np.eye(1 + self.Nu + self.Nx), np.matmul(Xt, Yt.T),
                                        rcond=None)[0].T

        if mode == "Regularization":
            from sklearn.linear_model import Ridge
            clf = Ridge(alpha=0, fit_intercept=False)

            Xt = Xt.T
            Yt = Yt.T

            clf.fit(Xt, Yt)
            self.Wout = clf.coef_

        if mode == "Pseudo":
            print("Obtaining Wout woth Pseudo inverse:")
            PIX = np.linalg.pinv(Xt)
            print("PIX shape", PIX.shape, "Yt shape", Yt.shape)
            self.Wout = np.matmul(PIX.T, Yt.T).T

        print("Train get wout shape: ", self.Wout.shape)
        return self.Wout

    def train_no_feed(self, context, output):

        plot_storage_root_adr = "C:\\Users\\zk036529\\OneDrive - ozyegin.edu.tr\\Belgeler\\Research\\CESN_On_Reacher_Gym\\Plots"
        training_plot_path = plot_storage_root_adr + f"\Train_no_feed_{self.mode}\{self.train_eps}_eps"
        if not os.path.exists(training_plot_path):
            os.makedirs(training_plot_path)
        # this function trains the network when there is no feedback
        n_trials = output.shape[0]
        context_dim = context.shape[-1]
        length = output.shape[1]
        sample_index = np.linspace(0, length, self.N, endpoint=False, dtype=np.int64)

        X_all = np.empty((self.Nx + self.Nu + 1, n_trials * self.N))  # 3d
        Y_all = np.empty((n_trials * self.N, self.Ny))
        Y_train_pred = np.empty((context.shape[0], self.N, self.Ny))
        Y_train_truth = np.empty((context.shape[0], self.N, self.Ny))

        for k in range(n_trials):
            Y = output[k, sample_index].reshape(self.N, self.Ny)
            # print(Y)
            con = context[k, sample_index].reshape(self.N, context_dim)
            # print(con)
            # print(err)
            u = con
            Xu = self.GetX_no_feed(u)
            # print(Xu.shape)
            # err
            X_all[:, k * (self.N):(k + 1) * (self.N)] = Xu
            # print(X_all)
            # print("===")
            Y_all[k * self.N: (k + 1) * (self.N), :] = Y
            Y_train_truth[k] = Y

        self.Wout = self.Train_GetWout(X_all, Y_all, self.mode)
        pred = np.matmul(self.Wout, X_all).T
        # print("Train Y pred shape:", pred.shape)
        for k in range(n_trials):
            Y_train_pred[k] = pred[k * self.N:(k + 1) * self.N, :]

        MSE, MSE_per_dim = self.RMSE(Y_train_truth, Y_train_pred)
        Visualizations.visualize_target_truth_predicted(context, Y_train_truth, Y_train_pred, training_plot_path)
        Visualizations.visualize_truth_predicted_per_dimension(Y_train_truth, Y_train_pred, training_plot_path,
                                                               MSE_per_dim)
        Visualizations.visualize_all_trajectories(context, Y_train_truth, Y_train_pred, training_plot_path)

        return MSE, X_all, self.Wout

    def Predict_no_feedback(self, context):
        length = context.shape[1]
        context_dim = context.shape[-1]
        sample_index = np.linspace(0, length, self.N, endpoint=False, dtype=np.int64)

        Y_test_pred = np.empty((context.shape[0], self.N, self.Ny))
        for k in range(context.shape[0]):
            con = context[k, sample_index].reshape(self.N, context_dim)
            u = con
            Xu = self.GetX_no_feed(u)
            pred = np.matmul(self.Wout, Xu).T
            Y_test_pred[k] = pred
        return Y_test_pred

    def save_weights(self, filename):
        weights_to_save = {
            'Recurrent_reservoir_weights': self.W,
            'input_weights': self.Win,
            'output_weights': self.Wout,
            'Reservoir_States': self.X
        }

        np.savez(filename, **weights_to_save)

    def load_weights(self, filename):
        loaded_weights = np.load(filename)

        self.W = loaded_weights['Recurrent_reservoir_weights']
        self.Win = loaded_weights['input_weights']
        self.Wout = loaded_weights['output_weights']
        self.X = loaded_weights['Reservoir_States']

    def RMSE(self, truth, estimated):

        MSE_per_dim = []
        MSE_per_traj = []
        # dimensions = ["Fingertip_x" , "Fingertip_y"]
        dimensions = ["Fingertip_x", "Fingertip_y", "action_1", "action_2"]
        # dimensions = ["Fingertip_x" , "Fingertip_y" , "action_1" , "action_2" , "q1" , "q2"]

        # for j in range(truth.shape[0]):
        for k in range(truth.shape[2]):
            MSE = np.square(np.subtract(truth[:, :, k], estimated[:, :, k]))
            MSE_per_dim.append(MSE.mean())
            print(f"The Error for {dimensions[k]} is  :", MSE.mean())

        print("For all:")
        print(truth.shape, estimated.shape)
        MSE = np.square(np.subtract(truth, estimated)).mean()
        # RMSE = math.sqrt(MSE)
        print("MSE: ", MSE)
        # print("RMSE: " , RMSE)
        return MSE, MSE_per_dim