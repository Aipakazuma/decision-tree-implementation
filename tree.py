from sklearn.datasets import load_iris
import numpy as np


def calc_gini(y):
    """gini関数.
    
    不純度の評価.
    """
    classes = np.unique(y)
    data_length = y.shape[0]
    gini = 1.0
    for c in classes:
        gini -= (len(y[y == c]) / data_length) ** 2

    return gini


def calc_gini_index(gini, gini_l, gini_r, pl, pr):
    """gini index -> 分割の評価方法."""
    return gini - (pl * gini_l + pr * gini_r)


if __name__ == '__main__':
    # test
    iris_data = load_iris()
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    x = np.array(iris_data.data)
    y = np.array(iris_data.target)

    X = x[:, :2]
    n_all_X = X.shape[0]
    n_X = X.shape[1]

    # 分ける前のgini係数
    gini = calc_gini(y)
    print('分ける前のgini係数 {}'.format(gini,))

    # 一番良いgini係数を補完？していく
    best_gini_index = 0.0
    best_x = None
    best_threshold = None

    # 分割候補の計算
    for n in range(n_X):
        # 分割候補の作成
        # 重複削除
        data_unique = np.unique(x[:, n])
        # (a ~ y + b ~ z) / 2.0
        # -> (a + b) / 2.0
        # -> aとbの中間の値を作成してくれる
        # -> これを全値でやる
        points = (data_unique[:-1] + data_unique[1:]) / 2.0

        for threshold in points:
            # 閾値で2グループに分割
            y_l = y[X[:, n] < threshold]
            y_r = y[X[:, n] >= threshold]

            # 分割後のgini係数を計算
            gini_l = calc_gini(y_l)
            gini_r = calc_gini(y_r)
            p_l = float(y_l.shape[0]) / n_all_X
            p_r = float(y_r.shape[0]) / n_all_X
            gini_index = calc_gini_index(gini, gini_l, gini_r, p_l, p_r)

            if gini_index > best_gini_index:
                best_gini_index = gini_index
                best_x = n
                best_threshold = threshold

    print('best gini {}, best x {}, best threshold {}'.format(best_gini_index,
                                                              best_x,
                                                              best_threshold))
