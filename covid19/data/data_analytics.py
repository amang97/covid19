import numpy as np
from sklearn.feature_selection import SelectKBest
# from ..utilities.config import 

class DataAnalytics:
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
    
    def p_y(self):
        """ Input - 
                (None)
                assumes labels as binary 0 and 1
            Output -
                empirical Probability estimate p_hat(y = 1)
        """
        return self.y.sum()/self.y.count()
    
    