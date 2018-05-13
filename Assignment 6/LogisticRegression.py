import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


class LogisticRegression(object):
    def __init__(self, data_path, seed, test_prop=.2):
        self.seed = seed
        self.seed_rng = np.random.RandomState(seed)
        data = pd.read_csv(
            filepath_or_buffer=data_path,
            header=None,
            sep=',')
        self.feature_num = len(data.columns) - 1
        ss = ShuffleSplit(n_splits=1, train_size=None, test_size=test_prop, random_state=seed)
        train_idx, test_idx = ss.split(data.iloc[:, 1:], data.iloc[:, 0]).__next__()
        self.data = {
            'train_dat': data.iloc[train_idx, 1:],
            'train_cls': data.iloc[train_idx, 0],
            'test_dat': data.iloc[test_idx, 1:],
            'test_cls': data.iloc[test_idx, 0]
        }
        for key, val in self.data.items():
            if key.endswith('dat'):
                self.data[key] = np.concatenate([(val - np.mean(val, axis=0)) / np.std(val, axis=0), np.ones((len(val), 1))], axis=-1)
            else:
                self.data[key] = (val - 1).as_matrix()
        # Xavier initialization
        # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # initialize theta with it
        self.theta = None
        self._init_theta()

    def _init_theta(self):
        self.theta = self.seed_rng.uniform(
            -np.sqrt(6 / (self.feature_num + 1)),
            np.sqrt(6 / (self.feature_num + 1)),
            self.feature_num + 1)

    @staticmethod
    def sigmoid(x):
        return np.exp(x) / (1 + np.exp(x))

    def compute_loss(self, X, y):
        return X.T.dot(y - self.sigmoid(X.dot(self.theta)))

    def predict(self, X):
        # Original equation is
        # self.sigmoid(X.dot(self.theta)) >= 0.5,
        # but it equals to
        return X.dot(self.theta) >= .0

    def test(self):
        return np.mean(self.data['test_cls'] == self.predict(self.data['test_dat']))

    def stop_cond(self, new_theta, ite, delta):
        # see whether it converges or not
        return (np.abs(new_theta - self.theta) < delta).all() or ite > 2e5

    def fit(self, learning_rate=1e-6, batch_size=64, stop_delta=3e-6):
        self._init_theta()
        batch_idx = self.seed_rng.choice(len(self.data['train_dat']), batch_size, replace=False)
        print(batch_idx)
        new_theta = self.compute_loss(self.data['train_dat'][batch_idx], self.data['train_cls'][batch_idx]) * learning_rate
        ite = 0
        # loop until convergence
        a = []
        while not self.stop_cond(new_theta, ite, stop_delta):
            print('ite%d: %.2f%%' % (ite, 100 * self.test()))
            a.append(self.test())
            self.theta = new_theta
            batch_idx = self.seed_rng.choice(len(self.data['train_dat']), batch_size)
            new_theta = self.compute_loss(self.data['train_dat'][batch_idx], self.data['train_cls'][batch_idx]) * learning_rate
            ite += 1

        # plt.style.use('ggplot')
        # plt.plot(np.arange(ite), a, lw=1, color='r')
        # plt.savefig('plot.png', dpi=128)


if __name__ == '__main__':
    lr = LogisticRegression('./data/wine_binary.csv', 0)
    lr.fit()

