from sklearn.ensemble import RandomForestClassifier

class RFC_model(RandomForestClassifier):
     
    def __init__(self):
        
        RandomForestClassifier.__init__(self,
                                        n_estimators=100,
                                        criterion='gini',
                                        max_depth=None,
                                        min_samples_split=10,
                                        min_samples_leaf=1,
                                        max_features='auto',
                                        oob_score=True,
                                        n_jobs=-1,
                                        random_state=1,
                                        )
     
        self.name = 'Random forest classifier'
     
    def __str__(self):
        return self.name