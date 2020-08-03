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
        
    def __str__(self):
        return 'Titanic'
    
    def _preprocess(self):
        # Preprocesses data using the preprocessing object
        self.train, self.test = self.preprocessing.preprocess(
                self.train, self.test)
    
    def machine_learning(self, mode):
        
        print('Selected mode:', mode)
        print()
        
        # Passenger id columns are removed from data, passenger id column is
        # saved from test data for submission labeling
        passenger_id = self.test['PassengerId']
        self.test = self.test.drop('PassengerId', axis=1)
        self.train = self.train.drop('PassengerId', axis=1)
        
        # Preprocess training and test data
        self._preprocess()
        
        # Training data is split into X (data) and y (labels)
        y = self.train['Survived']
        X = self.train.drop('Survived', axis=1)
        
        # Used classifier
        clf = RFC_model()
        
        if (mode == 'submission'):
            # In submission mode classifier is trained using all training data
            # available, trained model is used to classify test data, finally
            # classified data is written into a csv file in submission format
            # for kaggle
            filename = 'submission.csv'
            print('Training the classifier...')
            model = clf.fit(X, y)
            y_pred = model.predict(self.test)
            print('Training done')
            
            print('Writing results to csv file', filename + '...')
            output = pd.DataFrame({'PassengerId': passenger_id,
                                   'Survived': y_pred})
            output.to_csv(filename, index=False)
            print('Writing done')
        
        elif (mode == 'evaluation'):
            # In evaluation mode classifier is trained using partial training
            # data and the classifier performance is evaluated using
            # rest of the training data
            X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.4, random_state=0)
            
            print('Training the classifier...')
            model = clf.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            print('Training done')
            self.accuracy_score = accuracy_score(y_test, y_pred)
            self._results()
            
        else:
            # Prints error message if the entered mode command is incorrect
            print('Error:')
            print('Select mode (submission/evaluation)')
        
        print()
        
    def return_data(self):
        return self.train, self.test
        
    def _results(self):
        # Prints usefull information gained in the evaluation process
        print('Results:')
        print('Accuracy score: {:.4f}'.format(self.accuracy_score))
