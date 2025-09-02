import constants as c
from innitializer import init_solution
from new_solution import new_solution
import sklearn
# from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn.model_selection import train_test_split
from fitness import fitness

if __name__=="__main__":
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    parent = init_solution(c.N)
    parent_fitness = fitness(parent, x_train, x_test, y_train, y_test)
    for i in range(c.EPOCH):
        child = new_solution(parent, c.N)
        child_fitness = fitness(child, x_train, x_test, y_train, y_test)
        print(f"{i+1}) Parent: {parent_fitness},  Child: {child_fitness}")
        if parent_fitness<child_fitness:
            parent = child.copy()
            parent_fitness = child_fitness

    else:
        print(f"Best Found Features: {parent}   {parent_fitness}")
