from sklearn.ensemble import RandomForestClassifier

class RFC_model(RandomForestClassifier):
     
    def __init__(self):
        
        RandomForestClassifier.__init__(self,
                                        n_estimators=100
                                        )
     
        self.name = 'Random forest classifier'
     
    def __str__(self):
        return self.name