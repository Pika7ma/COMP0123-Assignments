import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import argparse


class Dist(object):
    '''
    Base class for various distribution
    '''
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class NormDist(Dist):
    """
    Utilities for normal distribution
    """
    def __init__(self, df):
        """
        Pre-compute $\mu$ and $\sigma$ for normal distribution
        :param df: contains sampling data fitting normal distribution
        :type df: pandas.DataFrame
        """
        data = df.as_matrix()
        self.mu = np.mean(data)
        # Set means delta degrees of freedom to 1 to obtain sample standard error
        self.sigma = np.std(data, ddof=1)
        super(NormDist, self).__init__(df)

    def __call__(self, val, laplacian=False):
        """
        Following the formula of probability density function of normal distribution
        :param val: x-value to be computed
        :param laplacian: laplacian smoothing has no effect on continuous distribution
        :type val: float
        :type laplacian: bool
        :return: possibility density at x-value
        """
        return 1.0 / (np.sqrt(2 * np.pi) * self.sigma) * np.exp(-(val - self.mu) ** 2 / (self.sigma ** 2 * 2))


class DiscreteDist(Dist):
    """
    Unitlites for discrete distribution
    """
    def __init__(self, df):
        """
        Pre-compute the frequency of each value and total size of data
        :param df: contains sampling data which can be classified into finite buckets
        :type df: pandas.DataFrame
        """
        # the hi-performance container Counter will counter all unique value
        # and results in their frequency
        self.cnt = Counter(df)
        self.total = len(df)
        super(DiscreteDist, self).__init__(df)

    def __call__(self, val, laplacian=False):
        """
        Compute the probability under discrete distribution
        :param val: x-value to be computed
        :param laplacian: prevent the probability from being zero, which has massive effect on final result
        :type val: int
        :type laplacian: bool
        :return: possibility density at x-value
        """
        if self.total == 0:
            return 0
        if laplacian:
            return (self.cnt[val] + 1) / (self.total + len(self.cnt))
        return self.cnt[val] / self.total


class NaiveBayes(object):
    def __init__(self, train_path, test_path, laplacian=False):
        """
        Initialize utilities for naive bayes
        :param train_path: path of training data
        :param test_path: path of testing data
        :param laplacian: indicate if we use laplacian smoothing
        :type train_path: str
        :type test_path: str
        :type laplacian: bool
        """
        self.train_path = train_path
        self.test_path = test_path
        # condition_probability[label][column] = Distribution Utility
        self.cond_prob = defaultdict(dict)
        # label_probability[label] = probability
        self.label_prob = dict()
        self.laplacian = laplacian
        self._build()

    def _build(self):
        # dtype should be discriminated by columns
        dtype = {
            'No.': np.int32,
            'Color': np.int32,
            'Root': np.int32,
            'Knocks': np.int32,
            'Texture': np.int32,
            'Umbilicus': np.int32,
            'Touch': np.int32,
            'Density': np.float32,
            'SugerRatio': np.float32,
            'Label': np.int32,
        }
        # load data
        self.train_dat = pd.read_csv(
            filepath_or_buffer=self.train_path,
            sep=',',
            index_col=0,
            dtype=dtype,
        )
        self.test_dat = pd.read_csv(
            filepath_or_buffer=self.test_path,
            sep=',',
            index_col=0,
            dtype=dtype,
        )
        # distribute different class utility for each column
        dist_cols = {
            'Color': DiscreteDist,
            'Root': DiscreteDist,
            'Knocks': DiscreteDist,
            'Texture': DiscreteDist,
            'Umbilicus': DiscreteDist,
            'Touch': DiscreteDist,
            'Density': NormDist,
            'SugerRatio': NormDist,
        }
        # for each unique label in training data
        for label in self.train_dat['Label'].unique():
            # get all columns and their names (except for the label column) from data with exact label
            for name, col in self.train_dat[self.train_dat['Label'] == label].iloc[:, :-1].iteritems():
                # initialize the distribution utility
                self.cond_prob[label][name] = dist_cols[name](col)
            # compute the labeled probability
            if self.laplacian:
                self.label_prob[label] = (len(self.train_dat[self.train_dat['Label'] == label]) + 1) / (len(self.train_dat) + len(self.train_dat['Label'].unique()))
            else:
                self.label_prob[label] = len(self.train_dat[self.train_dat['Label'] == label]) / len(self.train_dat)

    def test(self, verbose=True):
        # account for the accuracy
        acc = 0
        # get an testing example
        for i, (_, row) in enumerate(self.test_dat.iterrows()):
            # record the prediction and its probability
            pred, max_prob = 0, 0
            # for all column in exact label
            for label, dists in self.cond_prob.items():
                # start from its pre-test probability
                prob = self.label_prob[label]
                # times by each conditional probability
                for name, val in row[:-1].iteritems():
                    prob *= dists[name](val, self.laplacian)
                if verbose:
                    print('P(%s)=%f' % (label, prob), end=' ')
                # update the maximum probability and prediction
                if prob > max_prob:
                    max_prob = prob
                    pred = label
            if verbose:
                print()
            # compute the accuracy
            if pred == self.test_dat.iloc[i]['Label']:
                acc += 1
                if verbose:
                    print('Test %d: Predicted Correct' % (i + 1))
            else:
                if verbose:
                    print('Test %d: Predicted Wrong' % (i + 1))
        acc /= len(self.test_dat)
        if verbose:
            print('Total Accuracy: %.2f%%' % (acc * 100))
        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='nb')
    parser.add_argument('--laplacian', dest='laplacian', action='store_true')
    parser.add_argument('--no-laplacian', dest='laplacian', action='store_false')
    parser.set_defaults(laplacian=False)
    args = parser.parse_args()
    nb = NaiveBayes('./data/train.csv', './data/test.csv', laplacian=args.laplacian)
    nb.test(verbose=True)
