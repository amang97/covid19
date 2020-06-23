import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif

from ..utilities.config import DP, FP
class DataAnalytics:
    def __init__(self, data, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.data = data
    
    def p_y(self):
        """ Input - 
                (None)
                assumes labels as binary 0 and 1
            Output -
                empirical Probability estimate p_hat(y = 1)
        """
        return self.y.sum()/self.y.count()

    def s2nr(self, confl):
        X0 = self.X.loc[(self.y.to_numpy() == 0), confl]
        X1 = self.X.loc[(self.y.to_numpy() == 1), confl]
        return (abs(X0.mean() - X1.mean())).div((X0.std() + X1.std()))

    def visualize(self, ll, fn):
        plt.figure()
        sns.pairplot(self.data[ll])
        plt.savefig(fn)
        return None

    def correlation_matrix(self):
        cor = self.data.corr()
        cor.to_csv(FP['CORR'])
        return cor
    
    def select_features(self, catfl, confl, k_cat=None, k_con=None, cat_mode=None, con_mode=None): 
        """ Input -
                catfl: (list) categorical features list of header names
                confl: (list) continuous feature list of header names
                k_cat: (int) number of categorical features desired. default: All features
                k_con: (int) number of continuous features desired. default: All features
                cat_mode: (string)
                    - chi2: chi squared statistic to select features
                    - mi: mutual information to select features
                    default: None, No feature selection is done
                con_mode: (string)
                    - anova_f: ANOVA F-value between label/feature for classification tasks
                    default: None, No feature selection is done
            Output - 
                X transformed with the selected continuous and categorical features
                index array of selected features
        """
        k_cat = 'all' if k_cat is None else k_cat
        k_con = 'all' if k_con is None else k_con

        def chi_square_selection(self, catfl=catfl, k=k_cat):
            """ Input -
                    catfl: (list) categorical features list of header names
                    k: (int) number of features desired. default: All features
                Output -
                    - X transformed with the best k categorical features based
                    on the chi squared statistic
                    - The array of index of features selected
            """
            fs = SelectKBest(score_func=chi2, k=k)
            return fs.fit_transform(self.X[catfl], np.ravel(self.y)),\
                    fs.get_support(indices=True)
        
        def mutual_info_selection(self, catfl=catfl, k=k_cat):
            """ Input -
                    catfl: (list) categorical label features headers list
                    k: (int) number of features desired. default: All features
                Output -
                    - X transformed with the best k categorical features based on
                    mutual information 
                    - The array of index of features selected
            """
            fs = SelectKBest(score_func=mutual_info_classif, k=k)
            return fs.fit_transform(self.X[catfl], np.ravel(self.y)),\
                    fs.get_support(indices=True)
        
        def anova_f_selection(self, confl=confl, k=k_con):
            """ Input -
                    confl: (list) continuous label features headers list
                    k: (int) number of features desired. default: All features
                Output -
                    - displays the graph of negative log p_values and prints the p_values too
                    - X transformed with the best k categorical features based on 
                    - The array of index of features selected based on one way ANOVA using
                        the F statistic (consequently the p_values)
            """
            fs = SelectKBest(score_func=f_classif, k=k)
            fs.fit(self.X[confl], np.ravel(self.y))
            scores = -np.log10(fs.pvalues_)
            scores /= scores.max()
            print(f'p_values: {fs.pvalues_}')
            print('\n---------------**********---------------\n')
            plt.figure()
            plt.bar(np.arange(self.X[confl].shape[1]) - .45, scores, width=.2,\
            label=r'Univariate score ($-Log(p_{value})$)')
            plt.title("Comparing feature selection")
            plt.xlabel('Feature number')
            plt.yticks(())
            plt.axis('tight')
            plt.legend(loc='upper right')
            plt.savefig(FP['RFIMG'])
            return fs.transform(self.X[confl]), fs.get_support(indices=True)
        
        # Select Categorical Features
        if cat_mode is 'chi2':
            X_cat, cat_fs = chi_square_selection(self, catfl, k_cat)
        elif cat_mode is 'mi':
            X_cat, cat_fs = mutual_info_selection(self, catfl, k_cat)

        # Select Continuous features
        if con_mode is 'anova_f':
            X_con, con_fs = anova_f_selection(self, confl, k_con)

        # print(X_cat.shape, X_con.shape)
        return np.concatenate((X_cat, X_con), axis=1),\
                [cat_fs, con_fs]
