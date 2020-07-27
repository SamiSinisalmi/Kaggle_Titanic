class Preprocessing():
    
    def __init__(self):
        print('Preprocessing object created')
        
        self.train = 0
        self.test = 0
        
    def _process_age(self):
        print('TODO')
        
    def _process_cabin(self):
        print('TODO')
        
    def _process_fare(self):
        print('TODO')
        
    def _process_embark(self):
        print('TODO')
        
    def _fillna_with_zeros(self):
        self.train.fillna(0, inplace=True)
        self.test.fillna(0, inplace=True)

    def preprocess(self, train, test):
        
        print('Starting preprocessing...')
        
        self.train = train
        self.test = test
        
        self._process_age()
        self._process_cabin()
        self._process_embark()
        self._process_fare()      
        self._fillna_with_zeros()
        
        print('Preprocessing done')
        
        return self.train, self.test
    