import pandas as pd
from titanic import Titanic

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

titanic = Titanic(train, test)
titanic.preprocessing._estimate_age(train)