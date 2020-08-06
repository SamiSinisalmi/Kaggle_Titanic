from sklearn import preprocessing
import pandas as pd

class Preprocessing():
    
    def __init__(self):
        print('Preprocessing object created')
        
        self.data = 0
        
        self.encoder = preprocessing.LabelEncoder()
        
    def _extract_titles(self):
        # Extract titles from names for training data
        titles = []
        for name in self.data['Name']:
            titles.append(name.split(',')[1].split('.')[0])
        titles = pd.DataFrame({'Name': titles})
        self.data.update(titles)
        
    def _estimate_missing_age(self):
        # Fills nan age values with an estimated age value
        self.data.Age = self.data[['Name', 'Age']].apply(
                self._get_age_estimation, axis=1)
        
    def _get_age_estimation(self, row):
        # Estimates age based on title from name
        title = row[0]
        age = row[1]
        if pd.isnull(age):
            if title == ' Miss':
                return 21.8
            elif title == ' Mrs':
                return 35.72
            elif title == ' Master':
                return 4.57
            elif title == ' Mr':
                return 32.37
            elif title == ' Capt' or title == ' Col' or title == ' Major':
                return 48
            elif title == ' Jonkheer' or title == ' Don' or title == ' Sir':
                return 40.5
            else:
                return 42.33
        else:
            return age
        
    def _extract_features(self):
        # Extracts features from data, replaces data with features from data
        self._extract_titles()
        self._estimate_missing_age()
        
    def _fillnan_str(self):
        # Replaces NaN values with 'Unknown'
        column = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        for name in column:
            self.data[name].fillna('Unknown', inplace=True)
            
    def _fillnan_numeric(self):
        # Replaces NaN values with 0
        column = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
        for name in column:
            self.data[name].fillna(-1, inplace=True)
        
    def _numerice_columns(self):
        # Replaces all non numeric data values with a numeric value
        self._fillnan_str()
        self._fillnan_numeric()

        column = ['Sex', 'Name', 'Ticket', 'Cabin', 'Embarked']
        for name in column:
            self.data[name] = self.encoder.fit_transform(self.data[name])

    def preprocess(self, data):
        
        print('Starting preprocessing...')
        
        self.data = data
             
        self._extract_features()
        self._numerice_columns()
        
        print('Preprocessing done')
        print()
        
        return self.data
    
    def __str__(self):
        return 'Preprocessing'
    