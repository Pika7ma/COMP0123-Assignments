from DecisionTree import *


if __name__ == '__main__':
    id3 = ID3DecisionTree('./data/Watermelon-train1.csv', './data/Watermelon-test1.csv')
    id3.build_tree()
    id3.test()
    cart = CARTDecisionTree('./data/Watermelon-train2.csv', './data/Watermelon-test2.csv')
    cart.build_tree()
    cart.test()
