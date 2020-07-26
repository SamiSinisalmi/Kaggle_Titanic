import pandas as pd
from titanic import Titanic

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

titanic = Titanic(train, test)
X_train, X_test, y_train, y_test = titanic.machine_learning()
