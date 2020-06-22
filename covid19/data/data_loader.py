import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, dfp):
        """ Input - None
        """
        super().__init__()
        self.dfp = dfp      # Data file path
        self.data = None    # Initial load of the dataset
        # self.X = None       # data frame containing the selected features
        # self.y = None       # label column
        self.sr = None      # split ratio
        self.seed = None    # seed value to reproduce results

    # Setter Functions
    def read_data(self, fl=None, ll=None):
        """ Input -
                dfp: (Path object) data file path to the dataset csv file
                fl: (list of strings) feature column header names
                ll: (list of strings) label column header name
                default: features and label lists are empty
            Output -
                X: data frame containing the selected features
                y: label column
        """        
        # Handle none cases
        if fl is None:
            fl = []
        if ll is None:
            ll = []
        
        # Load data from CSV file
        self.data = pd.read_csv(self.dfp, usecols=fl+ll).dropna()
        X = self.data[fl]
        y = self.data[ll]
        return X, y

    def stratified_split(self, X, y, sr, seed):
        """ Input -
                sr: (float) split ratio % of test data
                seed: (int)
                default: stratify sampling is done using label column values
            Output -
                X_tr: Train set using split ratio & stratified sampling 
                y_tr: Corresponding Train labels
                X_t: Test set using split ratio & stratified sampling
                y_t: Corresponding Test labels
        """
        self.sr = sr
        self.seed = seed
        X_tr, X_t, y_tr, y_t = train_test_split(X, y,\
                                                test_size = sr,\
                                                random_state = seed,\
                                                stratify=y)
        return X_tr, X_t, y_tr, y_t

