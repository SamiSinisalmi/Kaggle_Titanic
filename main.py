import pandas as pd
from titanic import Titanic

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

titanic = Titanic(train, test)
titanic.machine_learning('evaluation')

train_processed, test_processed = titanic.return_data()
