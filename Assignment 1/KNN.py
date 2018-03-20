import numpy as np
import pandas as pd
from scipy import spatial
from collections import Counter


class KNN(object):
    def __init__(self, train_path, test_path, *, dat_len=256, tag_len=10):
        """
        Initialize ``k'' nearest neighbour object
        :type train_path: str
        :type test_path: str
        :type dat_len: int
        :type tag_len: int
        :return nothing
        """
        self.train_path = train_path
        self.test_path  = test_path
        self.dat_len    = dat_len
        self.tag_len    = tag_len
        self.train_all  = None
        self.test_dat   = None
        self.test_tag   = None

    def update_seed(self, seed):
        '''
        Set up random seed and build all
        :param seed: random seed
        :return: nothing
        '''
        np.random.seed(seed)
        self.seed = seed
        self._build()

    def _build(self):
        self.train_all = pd.read_csv(
            filepath_or_buffer=self.train_path,
            header=None,
            sep=' ',
            usecols=np.arange(self.dat_len + self.tag_len),
            dtype=np.int32,
        ).sample(frac=1, random_state=self.seed) # randomly shuffle the training data
        self.train_dat = self.train_all.iloc[:, :self.dat_len]
        self.train_tag = self.train_all.iloc[:, self.dat_len:]
        test_all = pd.read_csv(
            filepath_or_buffer=self.test_path,
            header=None,
            sep=' ',
            usecols=np.arange(self.dat_len + self.tag_len),
            dtype=np.int32,
        )
        dat = test_all.iloc[:, :self.dat_len]
        tag = test_all.iloc[:, self.dat_len:]
        self.test_dat = dat.as_matrix()
        self.test_tag = KNN.get_valid_tag(tag)

    @staticmethod
    def get_valid_tag(df):
        """
        Convert an one-hot DataFrame to Numpy array
        :type df: pandas.DataFrame
        :return: numpy.ndarray
        """
        return np.dot(df.as_matrix(), np.arange(10, dtype=np.int32))

    def test(self, k):
        train_tree = spatial.KDTree(self.train_dat.as_matrix())
        # convert one-hot test data into normal one by matrixly dot to an a-range matrix
        train_tag = np.dot(self.train_tag.as_matrix(), np.arange(10, dtype=np.int32))
        return KNN.run(train_tree, train_tag, self.test_dat, self.test_tag, k)

    def test_cv(self, k, frac=0.2, verbose=False):
        # Set verbose to True if we want detailed outputs
        # And we set frac out of all training data as cv set
        assert k >= 1 and 0.0 < frac <= 1.0
        partition = int(1 / 0.2)
        cv_len = int(frac * len(self.train_all))
        acc = list()
        for i in range(partition):
            acc_ = KNN.run(*self.get_cv_dat(i, cv_len), k)
            if verbose:
                print("%d/%d -> accuracy: %.2f%%" % (i + 1, partition, acc_ * 100), end='\r')
            acc.append(acc_)
        if verbose:
            print("CV accuracy when k=%d is %.2f%%" % (k, np.mean(acc) * 100))
        return np.mean(acc)

    @staticmethod
    def run(train_tree, train_tag, test_dat, test_tag, k):
        acc = 0.0
        for dat, tag in zip(test_dat, test_tag):
            # for each test pair, we query it in the k-d tree
            if KNN.query(train_tree, train_tag, dat, k) == tag:
                acc += 1.0
        acc /= len(test_tag)
        return acc

    def get_cv_dat(self, num, cv_len):
        assert num >= 0
        # Segement Train data and tags according to `cv_len`
        # And build a k-d tree based on the data
        start = num * cv_len
        end = start + cv_len
        cv_train_tree = pd.concat([self.train_dat.iloc[:start, :], self.train_dat.iloc[end:, :]])
        cv_train_tree = spatial.KDTree(cv_train_tree.as_matrix())
        cv_train_tag  = pd.concat([self.train_tag.iloc[:start, :], self.train_tag.iloc[end:, :]])
        cv_train_tag  = KNN.get_valid_tag(cv_train_tag)
        cv_test_dat   = self.train_dat.iloc[start:end, :]
        cv_test_dat   = cv_test_dat.as_matrix()
        cv_test_tag   = self.train_tag.iloc[start:end, :]
        cv_test_tag   = KNN.get_valid_tag(cv_test_tag)
        return cv_train_tree, cv_train_tag, cv_test_dat, cv_test_tag

    @staticmethod
    def query(tree, tag, data, k):
        """
        Query a data using specified k-d tree and tag
        :type tree: scipy.spatial.KDTree
        :type tag: numpy.ndarray
        :type data: numpy.ndarray
        :type k: int
        :return: numpy.ndarray
        """
        assert k >= 1
        _, i = tree.query(data, k=k)
        if k == 1:
            return tag[i]
        # Use collection object Counter to get the most common one element (prediction)
        ret, _ = Counter(tag[i]).most_common(1)[0]
        return ret;



if __name__ == '__main__':
    # test
    knn = KNN(
        './data/semeion_train.csv',
        './data/semeion_test.csv',
    )
    knn.update_seed(123456)
    for i in range(1, 11):
        print('%d: %f%%' % (i, knn.test(i) * 100))
