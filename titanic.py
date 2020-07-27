from preprocessing import Preprocessing

from sklearn.model_selection import train_test_split

class Titanic():
    
    def __init__(self, train, test):
        '''
        :param train:   train data, used for modelling
        :param test:    test data, used for model evaluation
        '''
        
        print('Titanic object created')
        
        self.train = train
        self.test = test
        
        self.preprocessing = Preprocessing()
        
        print()
        
    def get_train_data(self):
        
        return self.train
        
    def get_test_data(self):
        
        return self.test
    
    def _preprocess(self):
        
        self.train, self.test = self.preprocessing.preprocess(
                self.train, self.test)
    
    def machine_learning(self):
        
        self._preprocess()
        
        y = self.train['Survived']
        X = self.train.drop('Survived', axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4, random_state=0)
        
    def results(self):
        print('TODO')
