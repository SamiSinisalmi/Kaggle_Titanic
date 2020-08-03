from sklearn import preprocessing
import pandas as pd

class Preprocessing():
    
    def __init__(self):
        print('Preprocessing object created')
        
        self.train = 0
        self.test = 0
        
        self.encoder = preprocessing.LabelEncoder()
        
    def _extract_titles(self):
        # Extract titles from names for training data
        titles = []
        for name in self.train['Name']:
            titles.append(name.split()[1])
        titles = pd.DataFrame({'Name': titles})
        self.train.update(titles)
        # Extract titles from names for test data
        titles = []
        for name in self.test['Name']:
            titles.append(name.split()[1])
        titles = pd.DataFrame({'Name': titles})
        self.test.update(titles)
        
    def _extract_features(self):
        # Extracts features from data, replaces data with features from data
        self._extract_titles()
        
    def _fillnan_str(self):
        # Replaces NaN values with 'Unknown'
        column = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        for name in column:
            self.train[name].fillna('Unknown', inplace=True)
            self.test[name].fillna('Unknown', inplace=True)
            
    def _fillnan_numeric(self):
        # Replaces NaN values with 0
        column = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        for name in column:
            self.train[name].fillna(0, inplace=True)
            self.test[name].fillna(0, inplace=True)
        
    def _numerice_columns(self):
        # Replaces all non numeric data values with a numeric value
        self._fillnan_str()
        self._fillnan_numeric()
        
        column = ['Sex', 'Name', 'Ticket', 'Cabin', 'Embarked']
        for name in column:
            self.train[name] = self.encoder.fit_transform(self.train[name])
            self.test[name] = self.encoder.fit_transform(self.test[name])

    def preprocess(self, train, test):
        
        print('Starting preprocessing...')
        
        self.train = train
        self.test = test
             
        self._extract_features()
        self._numerice_columns()
        
        print('Preprocessing done')
        print()
        
        return self.train, self.test
    
    def __str__(self):
        return 'Preprocessing'
    