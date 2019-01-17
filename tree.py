import numpy as np
from graphviz import Digraph


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


def calc_entropy(y):
    """情報エントロピー.
    """
    classes = np.unique(y)
    data_length = y.shape[0]

    val = 0
    for c in classes:
        p = len(y[y == c]) / data_length
        if p != 0.0:
            val -= p * np.log2(p)

    return val


def calc_gini_index(gini, gini_l, gini_r, pl, pr):
    """gini index -> 分割の評価方法."""
    return gini - (pl * gini_l + pr * gini_r)


class _Node():
    def __init__(self, criterion='gini', max_depth=None):
        self.left = None
        self.right = None
        self.data_count = 0
        self.count_by_label = []
        self.label = None
        self.gini_index = 0.0
        self.x = None
        self.n_x = None
        self.threshold = None
        self.criterion = criterion
        self.max_depth = max_depth

    def build(self, X, y, depth=1):
        self.data_count = X.shape[0]

        if len(X.shape) <= 1:
            raise ValueError('Xは2次元で渡してください.')

        self.n_x = X.shape[1]

        # 全部同じクラスなら分割する必要がないので処理終了
        unique, counts = np.unique(y, return_counts=True)
        self.count_by_label = counts
        self.labels = unique
        if len(unique) == 1:
            self.label = unique[0]
            return

        # ラベルを多数決で決める
        self.label = unique[np.argmax(counts)]

        # 木の数が上限と一致したら終了
        if depth == self.max_depth:
            return

        # 分ける前のgini係数
        if self.criterion == 'gini':
            calc_func = calc_gini
        elif self.criterion == 'entropy':
            calc_func = calc_entropy
        else:
            raise ValueError('{}は存在しません.'.format(self.criterion))

        gini = calc_func(y)
        print(gini)

        # 一番良いgini係数を補完？していく
        best_gini_index = 0.0
        best_x = None
        best_threshold = None

        # 分割候補の計算
        for n in range(self.n_x):
            # 分割候補の作成
            # 重複削除
            data_unique = np.unique(X[:, n])
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
                gini_l = calc_func(y_l)
                gini_r = calc_func(y_r)
                p_l = float(y_l.shape[0]) / self.data_count
                p_r = float(y_r.shape[0]) / self.data_count
                gini_index = calc_gini_index(gini, gini_l, gini_r, p_l, p_r)

                if gini_index > best_gini_index:
                    best_gini_index = gini_index
                    best_x = n
                    best_threshold = threshold

        # 不純度が減らなければ終了
        if best_gini_index == 0.0:
            return

        self.gini_index = best_gini_index
        self.x = best_x
        self.threshold = best_threshold

        # 左側の分割
        conditions_l = X[:, self.x] < self.threshold
        x_l = X[conditions_l]
        y_l = y[conditions_l]
        self.left = _Node(criterion=self.criterion, max_depth=self.max_depth)
        self.left.build(x_l, y_l, depth + 1)

        # 右側の分割
        conditions_r = X[:, self.x] >= self.threshold
        x_r = X[conditions_r]
        y_r = y[conditions_r]
        self.right = _Node(criterion=self.criterion, max_depth=self.max_depth)
        self.right.build(x_r, y_r, depth + 1)

    def get_tree_info(self):
        _name = 'name_{}'.format(np.random.uniform())
        if self.x is None:
            # 葉の場合
            return {
                'label': self.label,
                'data_count': self.data_count,
                'name': _name
            }

        # 節の場合
        left = self.left.get_tree_info()
        right = self.right.get_tree_info()
        return {
            'left': left,
            'right': right,
            'data_count': self.data_count,
            'name': _name
        }

    def predict(self, d):
        if self.x is None:
            return self.label

        if d[self.x] < self.threshold:
            return self.left.predict(d)
        else:
            return self.right.predict(d)


class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.root = _Node(criterion=criterion, max_depth=max_depth)
        self.importances = None

    def fit(self, X, y):
        self.root.build(X, y)
        self.compute_feature_importances()

    def predict(self, data):
        results = []
        for d in data:
            results.append(self.root.predict(d))

        return results

    def make_graph(self):
        tree_dict = self.root.get_tree_info()

        # formatはpngを指定(他にはPDF, PNG, SVGなどが指定可)
        g = Digraph(format='png')
        g.attr('node', shape='circle')
        make_graph(tree_dict, g)
        g.render()

    def _compute_feature_importances(self, node):
        if node.x is None:
            return

        self.importances[node.x] += node.gini_index * node.data_count

        self._compute_feature_importances(node.left)
        self._compute_feature_importances(node.right)

    def compute_feature_importances(self, normalize=True):
        self.importances = np.zeros(self.root.n_x)
        self._compute_feature_importances(self.root)
        self.importances /= self.root.data_count

        if normalize:
            normalizer = np.sum(self.importances)

            if normalizer > 0.0:
                self.importances /= normalizer


def make_graph(d, g):
    if 'label' in d:
        # 葉
        g.node(d['name'])
    else:
        # 節
        g.edge(d['name'], d['left']['name'], label=str(d['left']['data_count']))
        g.edge(d['name'], d['right']['name'], label=str(d['right']['data_count']))

        make_graph(d['left'],  g)
        make_graph(d['right'], g)


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.tree import DecisionTreeClassifier
    # test
    iris_data = load_iris()
    # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    x = np.array(iris_data.data)
    y = np.array(iris_data.target)

    # X = x[:, :2]
    X = x
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=71)
    tree = DecisionTree()
    tree.fit(X_train, y_train)

    print(classification_report(y_train, tree.predict(X_train)))
    print(classification_report(y_test, tree.predict(X_test)))
    print(tree.importances)

    s_tree = DecisionTreeClassifier()
    s_tree.fit(X_train, y_train)
    print(classification_report(y_train, s_tree.predict(X_train)))
    print(classification_report(y_test, s_tree.predict(X_test)))
    print(s_tree.feature_importances_)

    tree.make_graph()
