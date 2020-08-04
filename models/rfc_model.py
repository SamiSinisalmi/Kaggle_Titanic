from sklearn.ensemble import RandomForestClassifier

class RFC_model(RandomForestClassifier):
     
    def __init__(self):
        
        RandomForestClassifier.__init__(self,
                                        n_estimators=500,
                                        criterion='gini',
                                        max_depth=1,
                                        min_samples_split=2,
                                        min_samples_leaf=1,
                                        max_features='auto',
                                        oob_score=False,
                                        n_jobs=-1,
                                        random_state=None,
                                        )
     
        self.name = 'Random forest classifier'
     
    def __str__(self):
        return self.name