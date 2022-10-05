import torch
from torch.utils.data import Dataset, DataLoader
# from data.simdata import *
import numpy as np
from utils.data_helper import *
from scipy.stats import norm
from scipy import interpolate

class Dataset_from_simdata(Dataset):
    def __init__(self, data):
        self.data = data
        self.num_data = data.num_data

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, t, y = self.data.x[idx], self.data.t[idx], self.data.y[idx],
        return (x, t, y)


def get_iter(data, batch_size, shuffle=True, rw=False):
    dataset = Dataset_from_simdata(data)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return iterator

class basedata():
    def __init__(self, n, n_feature=6) -> None:
        self.x = torch.rand([n, n_feature])
        self.num_data = n
        self.n_feature = n_feature
        self.t = self.set_treatment(self.x)
        self.y = self.get_outcome(self.x, self.t)
        self.true_pdf = self.get_correct_pdf()

    def set_pre_treatment(self, x):
        pass

    def get_outcome(self, x, t):
        pass

    def set_treatment(self, x):
        t = self.set_pre_treatment(self.x)
        t = t + torch.randn(self.num_data) * 0.5
        t = sigmod(t)
        return t

    def build_data(self):
        self.t = self.set_treatment(self.x)
        self.y = self.get_outcome(self.x, self.t)
        return self.x, self.t, self.y

    def get_dose(self, t):
        n = t.shape[0]
        x_tmp = torch.rand([10000, self.n_feature])
        dose = torch.zeros(n)
        for i in range(n):
            t_i = t[i]
            psi = self.get_outcome(x_tmp, t_i).mean()
            # psi /= n_test
            dose[i] = psi
        return dose

    def get_correct_conditional_desity(self, x, t):
        derivation_t = derivation_sigmoid(t).numpy()
        t = inverse_sigmoid(t)
        loc = self.set_pre_treatment(x)
        scale = 0.5
        pdf = norm.pdf(t, loc, scale) * derivation_t
        return pdf

    def get_correct_desity(self, t):
        x = torch.rand([10000, 6])
        cde = self.get_correct_conditional_desity(x, t)
        return torch.from_numpy(cde.mean(axis=1))

    def get_correct_pdf(self):
        t_test = torch.linspace(0, 1, 10000).reshape(-1, 1)
        des = self.get_correct_desity(t_test)
        true_pdf = interpolate.interp1d(t_test.squeeze(1), des)
        return true_pdf

    def get_ideal_weights(self, x, t, power=0.5):
        t_ = t.reshape(-1, 1)
        conditional_de = self.get_correct_conditional_desity(x, t)
        des = torch.from_numpy(self.true_pdf(t_).squeeze())
        ideal_weights = des / conditional_de
        ideal_weights = torch.pow(ideal_weights, power)
        return ideal_weights


class data1(basedata):
    def __init__(self, n, n_feature=6) -> None:
        super().__init__(n, n_feature)

    def set_pre_treatment(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        t = (10. * torch.sin(torch.max(x1, torch.max(x2, x3))) + torch.max(x3, torch.max(x4, x5))**3)/(1. + (x1 + x5)**2) + torch.sin(0.5 * x3) * (1. + torch.exp(x4 - 0.5 * x3)) + x3**2 + 2. * torch.sin(x4) + 2.*x5 - 6.5

        return t

    def get_outcome(self, x, t):
        x1 = x[:, 0]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x6 = x[:, 5]
        y = torch.cos((t-0.5) * 3.14159 * 2.) * (t**2 + (4.*torch.max(x1, x6)**3)/(1. + 2.*x3**2)*torch.sin(x4))
        y = y + torch.randn(x.shape[0]) * 0.5

        return y
