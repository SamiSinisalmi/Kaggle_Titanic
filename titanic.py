from preprocessing import Preprocessing

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
        
    def _get_train_data(self):
        return self.train
        
    def _get_test_data(self):
        return self.test
