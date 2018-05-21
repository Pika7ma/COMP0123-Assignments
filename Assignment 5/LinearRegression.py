import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


class LinearRegression(object):
    def __init__(self, seed, learning_rate=1e-6, test_prop=.2):
        self.seed = seed
        self.seed_rng = np.random.RandomState(seed)
        self.data = load_boston()
        # normalization
        self.data['data'] = (self.data['data'] - np.mean(self.data['data'], axis=0)) / np.std(self.data['data'], axis=0)
        ss = ShuffleSplit(n_splits=1, train_size=None, test_size=test_prop, random_state=seed)
        # get random index
        self.train_cv_idx, self.test_idx = ss.split(self.data['data'], self.data['target']).__next__()
        self.learning_rate = learning_rate
        self.feature_num = len(self.data['feature_names'])
        # Xavier initialization
        # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        # initialize theta with it
        self.theta = None
        self._init_theta()

    def cv(self, lamb_list, norm_ord):
        self._cv(
            self.data['data'][self.train_cv_idx],
            self.data['target'][self.train_cv_idx],
            lamb_list,
            norm_ord
        )

    def fit(self, lamb, norm_ord):
        self._fit(
            self.data['data'][self.train_cv_idx],
            self.data['target'][self.train_cv_idx],
            self.data['data'][self.test_idx],
            self.data['target'][self.test_idx],
            lamb,
            norm_ord
        )

    def _init_theta(self):
        self.theta = self.seed_rng.uniform(
            -np.sqrt(6 / (self.feature_num + 1)),
            np.sqrt(6 / (self.feature_num + 1)),
            self.feature_num + 1)

    def stop_cond(self, new_theta, ite, delta=1e-6):
        # see whether it converges or not
        return (np.abs(new_theta - self.theta) < delta).all() or ite > 2e5

    def SGD(self, X, y, lamb, norm_ord):
        # concatenate X with ones to get rid of annoying redundant 1
        X = np.concatenate([np.ones((len(X), 1)), X], axis=-1)
        # vectorization
        delta_theta = (y - X.dot(self.theta)).dot(X)
        # regularization
        if norm_ord == 1:
            delta_theta -= lamb
        elif norm_ord == 2:
            delta_theta -= lamb * self.theta
        delta_theta *= self.learning_rate / len(X)
        return self.theta + delta_theta

    def _cv(self, train_X, train_y, lamb_list, norm_ord=2, batch_size=32, cv_folds=5):
        for lamb in lamb_list:
            accs = []
            cv_split = KFold(n_splits=cv_folds).split(self.data['data'][self.train_cv_idx])
            for cv_train_idx, cv_test_idx in cv_split:
                # get data
                cv_train_X = train_X[cv_train_idx]
                cv_train_y = train_y[cv_train_idx]
                cv_test_X = train_X[cv_test_idx]
                cv_test_y = train_y[cv_test_idx]
                # init theta
                self._init_theta()
                batch_idx = self.seed_rng.choice(len(cv_train_X), batch_size)
                new_theta = self.SGD(cv_train_X[batch_idx], cv_train_y[batch_idx], lamb, norm_ord)
                ite = 0
                # loop until convergence
                while not self.stop_cond(new_theta, ite):
                    self.theta = new_theta
                    batch_idx = self.seed_rng.choice(len(cv_train_X), batch_size)
                    new_theta = self.SGD(cv_train_X[batch_idx], cv_train_y[batch_idx], lamb, norm_ord)
                    ite += 1
                accs.append(self.get_acc(cv_test_X, cv_test_y))
            print('lambda=%f:' % lamb, np.mean(accs))

    def _fit(self, train_X, train_y, test_X, test_y, lamb, batch_size=32, norm_ord=2):
        self._init_theta()
        batch_idx = self.seed_rng.choice(len(train_X), batch_size)
        new_theta = self.SGD(train_X[batch_idx], train_y[batch_idx], lamb, norm_ord)
        ite = 0
        # loop until convergence
        a = []
        while not self.stop_cond(new_theta, ite):
            self.theta = new_theta
            batch_idx = self.seed_rng.choice(len(train_X), batch_size)
            new_theta = self.SGD(train_X[batch_idx], train_y[batch_idx], lamb, norm_ord)
            print('ite%d:' % ite, self.get_acc(test_X, test_y))
            a.append(self.get_acc(test_X, test_y))
            ite += 1

        plt.style.use('ggplot')
        plt.plot(np.arange(ite), a, lw=1, color='r')
        plt.savefig('plot.png', dpi=128)

    def get_acc(self, test_X, test_y):
        return np.linalg.norm(np.concatenate([np.ones((len(test_X), 1)), test_X], axis=-1).dot(self.theta) - test_y, ord=2) / np.sqrt(len(test_X))


if __name__ == '__main__':
    lr = LinearRegression(12345, 1e-2)
    lr.cv([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2], 2)
    lr.fit(1e-6, 2)

