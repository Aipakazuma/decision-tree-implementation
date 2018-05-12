from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from tree import DecisionTree
import pandas as pd
import numpy as np


if __name__ == '__main__':
    train_df = pd.read_csv('~/.kaggle/competitions/titanic/train.csv')
    le_sex = LabelEncoder()
    le_sex.fit(train_df['Sex'])
    train_df.loc[:, 'SexInt'] = le_sex.transform(train_df['Sex'])

    X = np.array(train_df[['SexInt']])
    y = train_df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=71)
    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)

    print(classification_report(y_train, tree.predict(X_train)))
    print(classification_report(y_test, tree.predict(X_test)))

    tree.make_graph()
