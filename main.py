import pandas as pd
from titanic import Titanic

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

titanic = Titanic(train, test)
titanic.preprocess()
titanic.machine_learning('evaluation')
titanic.machine_learning('submission')

train_processed, test_processed = titanic.return_data()
