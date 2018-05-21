import numpy as np
import pandas as pd
from collections import defaultdict


class DecisionTree(object):
    def __init__(self, train_path, test_path):
        self.label_name = '好瓜'
        self.train_dat = pd.read_csv(
            filepath_or_buffer=train_path,
            sep=',',
            index_col=0,
            encoding='gb2312',
        )
        self.test_dat = pd.read_csv(
            filepath_or_buffer=test_path,
            sep=',',
            index_col=0,
            encoding='gb2312',
        )
        self.root = dict()

    # unified calling interface
    def build_tree(self):
        self._build_tree(self.root, self.train_dat)

    # all trees should implement this tree-building algorithm
    def _build_tree(self, root, dat):
        raise NotImplementedError

    # predict a single row in data
    def predict(self, row):
        raise NotImplementedError

    # test all data in the testing set
    def test(self):
        accuracy = list()
        for _, row in self.test_dat.iterrows():
            accuracy.append(self.predict(row) == row[self.label_name])
        print(accuracy)
        print(np.mean(accuracy))

    # decide which feature to use for classification
    def feature_decision(self, dat):
        raise NotImplementedError


class ID3DecisionTree(DecisionTree):
    def _build_tree(self, root, dat):
        if len(dat[self.label_name].unique()) == 1:
            root['feature'] = None
            root['isLeaf'] = True
            root['prediction'] = dat[self.label_name].unique()[0]
        elif len(dat.columns) == 2:
            root['feature'] = dat.columns[0]
            root['isLeaf'] = True
            root['prediction'] = dict()
            dat_groups = dat.groupby([root['feature'], self.label_name]).size().groupby(root['feature'])
            for name, dat_group in dat_groups:
                root['prediction'][name] = dat_group.idxmax()[-1]
        else:
            root['feature'] = self.feature_decision(dat)
            root['isLeaf'] = False
            root['children'] = dict()
            for val in dat[root['feature']].unique():
                root['children'][val] = dict()
                self._build_tree(root['children'][val], dat[dat[root['feature']] == val].drop(root['feature'], axis=1))

    def predict(self, row):
        root = self.root
        while not root['isLeaf']:
            root = root['children'][row[root['feature']]]
        if root['feature']:
            return root['prediction'][row[root['feature']]]
        else:
            return root['prediction']

    def feature_decision(self, dat):
        """
        Decide which feature to use for classification
        :param dat: the total data
        :type dat: pandas.core.frame.DataFrame
        :return: the decision feature
        """
        info_loss_dict = defaultdict(float)
        for col_name in dat.iloc[:, :-1]:
            col_sizes = dat.groupby(col_name).size()
            dat_groups = dat.groupby([col_name, self.label_name]).size().groupby(col_name)
            for col_size, (_, dat_group) in zip(col_sizes, dat_groups):
                for label_size in dat_group:
                    info_loss_dict[col_name] -= label_size * np.log(label_size / col_size)
        return min(info_loss_dict, key=info_loss_dict.get)


class CARTDecisionTree(DecisionTree):
    def __init__(self, train_path, test_path):
        super().__init__(train_path, test_path)
        # self._preprocess()

    def _preprocess(self):
        replacements = defaultdict(dict, {
            '色泽': {'浅白': 0, '青绿': 1, '乌黑': 2},
            '根蒂': {'蜷缩': 0, '稍蜷': 1, '硬挺': 2},
            '敲声': {'沉闷': 0, '浊响': 1, '清脆': 2},
            '纹理': {'清晰': 0, '稍糊': 1, '模糊': 2}
        })
        for col_name in self.train_dat:
            self.train_dat[col_name].replace(replacements[col_name], inplace=True)
            self.test_dat[col_name].replace(replacements[col_name], inplace=True)

    def _build_tree(self, root, dat):
        if len(dat[self.label_name].unique()) == 1:
            root['feature'] = None
            root['isLeaf'] = True
            root['prediction'] = dat[self.label_name].unique()[0]
        elif len(dat.columns) == 1:
            root['feature'] = None
            root['isLeaf'] = True
            dat_group = dat.groupby(self.label_name).size()
            root['prediction'] = dat_group.idxmax()
        else:
            root['feature'], root['split_pos'] = self.feature_decision(dat)
            root['isLeaf'] = False
            root['children'] = dict()
            if CARTDecisionTree.is_discrete(root['feature']):
                root['children'][True], root['children'][False] = dict(), dict()
                self._build_tree(root['children'][True], dat[dat[root['feature']] == root['split_pos']].drop(root['feature'], axis=1))
                self._build_tree(root['children'][False], dat[dat[root['feature']] != root['split_pos']].drop(root['feature'], axis=1))
            else:
                root['children'][True], root['children'][False] = dict(), dict()
                self._build_tree(root['children'][True], dat[dat[root['feature']] < root['split_pos']].drop(root['feature'], axis=1))
                self._build_tree(root['children'][False], dat[dat[root['feature']] >= root['split_pos']].drop(root['feature'], axis=1))

    def predict(self, row):
        root = self.root
        while not root['isLeaf']:
            if CARTDecisionTree.is_discrete(root['feature']):
                root = root['children'][row[root['feature']] == root['split_pos']]
            else:
                root = root['children'][row[root['feature']] < root['split_pos']]
        return root['prediction']

    @staticmethod
    def is_discrete(col_name):
        if col_name == '密度':
            return False
        return True

    def compute_gini(self, dat, rest_dat):
        gini = 0.0
        for group in [dat, rest_dat]:
            if len(group) > 0:
                gini += len(group[group[self.label_name] == '是']) * len(group[group[self.label_name] != '是']) / len(group)
        return gini

    def feature_decision(self, dat):
        """
        Decide which feature to use for classification
        :param dat: the total data
        :type dat: pandas.core.frame.DataFrame
        :return: the decision feature
        """
        gini_dict = dict()
        for col_name in dat.iloc[:, :-1]:
            inner_gini_dict = defaultdict(float)
            if CARTDecisionTree.is_discrete(col_name):
                for unique_val in dat[col_name].unique():
                    inner_gini_dict[unique_val] = self.compute_gini(dat[dat[col_name] == unique_val], dat[dat[col_name] != unique_val])
            else:
                vals = np.sort(self.train_dat[col_name].unique())
                for val1, val2 in zip(vals[:-1], vals[1:]):
                    mean = (val1 + val2) / 2
                    inner_gini_dict[mean] = self.compute_gini(dat[dat[col_name] < mean], dat[dat[col_name] >= mean])
            gini_dict[col_name] = inner_gini_dict
        min_gini = np.inf
        ret_col_name, ret_split_pos = str(), float()
        for col_name, inner_gini_dict in gini_dict.items():
            for split_pos, gini in inner_gini_dict.items():
                if gini < min_gini:
                    ret_col_name = col_name
                    ret_split_pos = split_pos
                    min_gini = gini
        return ret_col_name, ret_split_pos

if __name__ == '__main__':
    id3 = ID3DecisionTree('./data/Watermelon-train1.csv', './data/Watermelon-test1.csv')
    id3.build_tree()
    id3.test()
    cart = CARTDecisionTree('./data/Watermelon-train2.csv', './data/Watermelon-test2.csv')
    cart.build_tree()
    cart.test()
