import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

from sklearn.datasets import load_iris
from ANN.preceptron import MyPerceptron
from sklearn.linear_model import Perceptron


def test_customized_preceptron(X, y):
    model = MyPerceptron()
    model.fit(X, y)
    x_points = np.linspace(4, 7, 10)
    y_ = -(model.w[0] * x_points + model.b) / model.w[1]
    plt.plot(x_points, y_)
    plt.plot(data[:50, 0], data[:50, 1], 'bo', color='blue', label='0')
    plt.plot(data[50:100, 0], data[50:100, 1], 'bo', color='orange', label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    # plt.savefig('iris-demo.png')
    plt.show()


def test_sklearn_preceptron(X, y):
    clf = Perceptron(fit_intercept=True,
                     # until classifying all data correctly
                     tol=None,
                     max_iter=1000,
                     shuffle=True)
    clf.fit(X, y)

    plt.figure(figsize=(10, 10))
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.title('鸢尾花线性数据示例')

    plt.scatter(
        data[:50, 0],
        data[:50, 1],
        c='b',
        label='Iris-setosa',
    )
    plt.scatter(data[50:100, 0],
                data[50:100, 1],
                c='orange',
                label='Iris-versicolor')

    x_ponits = np.arange(4, 8)
    y_ = -(clf.coef_[0][0] * x_ponits + clf.intercept_) / clf.coef_[0][1]

    plt.plot(x_ponits, y_)
    plt.legend()
    plt.grid(False)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    # 取前 100 个数据测试, 选择前两列作为输入特征, 最后一列 label 作为标签
    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    # 考虑一个简化过的二分类问题
    y = np.array([1 if i == 1 else -1 for i in y])

    parser = argparse.ArgumentParser()
    parser.add_argument('--sklearn', action='store_true',
                        help='use sklearn precepton')
    args = parser.parse_args()
    # use preceptron from sklearn
    # `python test_preceptron.py --sklearn`
    if args.sklearn:
        test_sklearn_preceptron(X, y)
    else:
        test_customized_preceptron(X, y)
