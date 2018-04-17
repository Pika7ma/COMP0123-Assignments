import numpy as np
import matplotlib.pyplot as plt
import os


class Generator(object):
    def __init__(self, seed):
        self.rng = self._seed(seed)
        self.dat_x = None
        self.dat_y = None
        self.dat_x_split = []
        self.info = []

    def _seed(self, seed):
        return np.random.RandomState(seed)

    def _gen_mvnorm(self, mean, cov, n):
        """
        Generate data under multi-variate normal.
        :param mean: given mean
        :type mean: numpy.ndarray
        :param cov: given covariance
        :type cov: numpy.ndarray
        :param n: given data length
        :type n: int
        :return: generated data
        """
        assert len(mean.shape) == 1
        assert len(cov.shape) == 2
        assert (np.array(mean.shape + cov.shape) == mean.shape[0]).all()
        return self.rng.multivariate_normal(mean, cov, n).T

    def gen_mvnorm(self, pri, n=1000, mean=((1, 1), (4, 4), (8, 1))):
        assert len(mean) == len(pri)
        mean = np.array(mean)
        _, dim = mean.shape
        pri = np.array(pri, dtype=np.float32)
        pri /= pri.sum()
        cov = np.eye(dim) * 2
        self.dat_x = np.zeros((dim, n))
        self.dat_y = np.zeros((n, ))
        start = 0
        cls = 0
        for p, m in zip(pri[:-1], mean[:-1]):
            end = int(start + n * p)
            self.dat_x[:, start:end] = self._gen_mvnorm(m, cov, end - start)
            self.dat_y[start:end] = cls
            cls += 1
            start = end
        self.dat_x[:, start:] = self._gen_mvnorm(mean[-1], cov, n - start)
        self.dat_y[start:] = cls
        self._preprocess(pri)

    @staticmethod
    def euclid_disc(s, m, c):
        """
        Compute euclidean discriminator
        :type s: numpy.ndarray
        :type m: numpy.ndarray
        :type c: numpy.ndarray
        :return: euclidean discriminator
        """
        # return -(s - m).T.dot(np.linalg.inv(c)).dot(s - m).diagonal() / 2
        return -np.sum((s - m).T.dot(np.linalg.inv(c)).T * (s - m), axis=0) / 2

    @staticmethod
    def bayes_disc(s, m, c, p):
        """
        Compute bayes discriminator
        :type s: numpy.ndarray
        :type m: numpy.ndarray
        :type c: numpy.ndarray
        :type p: float
        :return: bayes discriminator
        """
        return Generator.euclid_disc(s, m, c) - np.log(np.linalg.det(c)) / 2 + np.log(p)

    def _preprocess(self, pri):
        self.info = []
        self.dat_x_split = []
        for c, p in enumerate(pri):
            s = np.stack([x_layer[self.dat_y == c] for x_layer in self.dat_x], axis=0)
            self.dat_x_split.append((s, p))
            self.info.append((np.mean(s, axis=1), np.cov(s)))

    def print_mean_cov(self):
        for i, (m, c) in enumerate(self.info):
            print('---------Class %d--------' % i)
            print('Mean:\n%s' % m)
            print('Covariance:\n%s' % c)

    def bayes_predict(self):
        prob = np.concatenate([np.stack([Generator.bayes_disc(s, m.reshape((-1, 1)), c, p) for m, c in self.info], axis=0) for s, p in self.dat_x_split], axis=1)
        result = np.argmax(prob, axis=0)
        print('Bayes Classifier Accuracy: %.2f%%' % (np.count_nonzero(result == self.dat_y) / len(result) * 100))

    def euclid_predict(self):
        prob = np.concatenate([np.stack([Generator.euclid_disc(s, m.reshape((-1, 1)), c) for m, c in self.info], axis=0) for s, _ in self.dat_x_split], axis=1)
        result = np.argmax(prob, axis=0)
        print('Euclidean Classifier Accuracy: %.2f%%' % (np.count_nonzero(result == self.dat_y) / len(result) * 100))

    def bayes_plot(self, path):
        plt.figure()
        x_min, x_max = self.dat_x[0].min() - 1, self.dat_x[0].max() + 1
        y_min, y_max = self.dat_x[1].min() - 1, self.dat_x[1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        cc = np.stack((xx.ravel(), yy.ravel()), axis=0)
        prob = np.stack([Generator.bayes_disc(cc, m.reshape((-1, 1)), c, p) for (m, c), (_, p) in zip(self.info, self.dat_x_split)], axis=0)
        result = np.argmax(prob, axis=0).reshape(xx.shape)
        plt.contourf(xx, yy, result, cmap=plt.cm.RdBu)
        for (s, _), c, m in zip(self.dat_x_split, ['orange', 'dimgrey', 'snow'], ['*', '+', 'o']):
            plt.scatter(s[0], s[1], s=50, c=c, marker=m, alpha=0.5)
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, 'bayes_plot.png'), dpi=256)

    def euclid_plot(self, path):
        plt.figure()
        x_min, x_max = self.dat_x[0].min() - 1, self.dat_x[0].max() + 1
        y_min, y_max = self.dat_x[1].min() - 1, self.dat_x[1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        cc = np.stack((xx.ravel(), yy.ravel()), axis=0)
        prob = np.stack([Generator.euclid_disc(cc, m.reshape((-1, 1)), c) for m, c in self.info], axis=0)
        result = np.argmax(prob, axis=0).reshape(xx.shape)
        plt.contourf(xx, yy, result, cmap=plt.cm.RdBu)
        for (s, _), c, m in zip(self.dat_x_split, ['orange', 'dimgrey', 'snow'], ['*', '+', 'o']):
            plt.scatter(s[0], s[1], s=50, c=c, marker=m, alpha=0.5)
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(os.path.join(path, 'euclid_plot.png'), dpi=256)


if __name__ == '__main__':
    g = Generator(12345)
    g.gen_mvnorm([1, 1, 1])
    g.print_mean_cov()
    print('---------Result---------')
    g.bayes_predict()
    g.euclid_predict()
    g.bayes_plot('./plot/811')
    g.euclid_plot('./plot/811')
