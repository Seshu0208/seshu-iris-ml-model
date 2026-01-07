import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    print(df["target"].unique())
    # print(len(df[df["target"]==0]))
    # print(len(df[df["target"]==1]))
    # print(len(df[df["target"]==2]))
    return df

def train_model(n_neighbors=5):

    df = load_data()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train,y_train)

    return knn, X_test, y_test


def evaluate_model(knn, X_test, y_test):
    
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)

    return accuracy, cls_report



# if __name__ == "__main__":

#     knn, X_test, y_test = train_model()

#     acc, cls_report = evaluate_model(knn, X_test, y_test)

#     print(acc)
#     print(cls_report)
