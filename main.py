import pandas as pd
from titanic import Titanic

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

titanic = Titanic(train, test)
titanic.machine_learning()
train_after_processing = titanic.get_train_data()
test_after_processing = titanic.get_test_data()
titanic.results()
