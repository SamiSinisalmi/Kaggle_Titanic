from sklearn.svm import SVC

class SVM_model(SVC):
     
    def __init__(self):
        
        SVC.__init__(self,
                     kernel='rbf',
                     gamma='scale')
     
        self.name = 'Support vector classifier'
     
    def __str__(self):
        return self.name