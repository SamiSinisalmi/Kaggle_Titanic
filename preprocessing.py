class Preprocessing():
    
    def __init__(self):
        print('Preprocessing object created')
        
        self.train = 0
        self.test = 0
        
    def _name(self):
        print('TODO')
        
    def _sex(self):
        print('TODO')
        
    def _ticket(self):
        print('TODO')
        
    def _cabin(self):
        print('TODO')
        
    def _embark(self):
        print('TODO')
        
    def _fillna_with_zeros(self):
        self.train.fillna(0, inplace=True)
        self.test.fillna(0, inplace=True)

    def preprocess(self, train, test):
        
        print('Starting preprocessing...')
        
        self.train = train
        self.test = test
             
        self._fillna_with_zeros()
        
        print('Preprocessing done')
        
        return self.train, self.test
    