from preprocessing import Preprocessing

from sklearn.model_selection import train_test_split

class Titanic():
    
    def __init__(self, train, test):
        '''
        :param train:   train data, used for modelling
        :param test:    test data, used for model evaluation
        '''
        
        print('Titanic object created')
        
        self.y = train['Survived']
        self.X = train.drop('Survived', axis=1)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.4, random_state=0)
        
        self.train = train
        self.test = test
        
        self.preprocessing = Preprocessing()
        
    def _get_train_data(self):
        return self.train
        
    def _get_test_data(self):
        return self.test
    
    def machine_learning(self):
        print('TODO')
        
    def results(self):
        print('TODO')
