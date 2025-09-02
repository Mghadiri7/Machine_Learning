import numpy as np
import sklearn
from sklearn import svm

def fitness(solution, x_train, x_test, y_train, y_test):
    _x_train = []
    for i in range(len(x_train)):
        lst = []
        for j in range(len(solution)):
            if solution[j]==1:
                lst.append(x_train[i][j])
        _x_train.append(lst)

    _x_test = []
    for i in range(len(x_test)):
        lst = []
        for j in range(len(solution)):
            if solution[j]==1:
                lst.append(x_test[i][j])
        _x_test.append(lst)

    model = svm.SVC(kernel="linear")
    model.fit(_x_train, y_train)
    score = model.score(_x_test, y_test)
    return score