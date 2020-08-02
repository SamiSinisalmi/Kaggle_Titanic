import pandas as pd

from preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.rfc_model import RFC_model

class Titanic():
    
    def __init__(self, train, test):
        '''
        :param train:   train data, used for modelling
        :param test:    test data, used for model evaluation
        '''
        
        print('Titanic object created')
        
        self.train = train
        self.test = test
        
        self.accuracy_score = 0
        
        self.preprocessing = Preprocessing()
        
        print()
    
    def _preprocess(self):
        # Preprocesses data using the preprocessing object
        self.train, self.test = self.preprocessing.preprocess(
                self.train, self.test)
    
    def machine_learning(self, mode):
        
        print('Selected mode:', mode)
        print()
        
        self._preprocess()
        
        y = self.train['Survived']
        X = self.train.drop('Survived', axis=1)
        
        clf = RFC_model()
        
        if (mode == 'submission'):
            
            filename = 'submission.csv'
            print('Training the classifier...')
            model = clf.fit(X, y)
            y_pred = model.predict(self.test)
            print('Training done')
            
            print('Writing results to csv file', filename + '...')
            output = pd.DataFrame({'PassengerId': self.test['PassengerId'],
                                   'Survived': y_pred})
            output.to_csv(filename, index=False)
            print('Writing done')
        
        elif (mode == 'evaluation'):
            
            X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.4, random_state=0)
            
            print('Training the classifier...')
            model = clf.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print('Training done')
            self.accuracy_score = accuracy_score(y_test, y_pred)
            self.results()
            
        else:
            
            print('Select mode (submission/evaluation)')
        
        print()
        
    def _results(self):
        print('Results:')
        print('Accuracy score: {:.4f}'.format(self.accuracy_score))
        
    def __str__(self):
        return 'Titanic'
