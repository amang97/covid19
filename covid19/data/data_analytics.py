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
        return round((self.y.sum()/self.y.count()).squeeze(), DP['ROUND'])

    def s2nr(self, confl):
        X0 = self.X.loc[(self.y.to_numpy() == 0), confl]
        X1 = self.X.loc[(self.y.to_numpy() == 1), confl]
        py = self.p_y()
        num = (X0.mean() - X1.mean())**2
        denom = (1-py)*X0.var() + (py)*X1.var()
        return (num).div(denom)

    def univariate_bayes_error(self, catfl):
        ube_scores = []
        p1 = self.p_y()
        p0 = 1 - p1

        for i, catf in enumerate(catfl):
            categories = self.X[catf].unique()
            if (len(categories) == 1):
                ube_scores.append(-1)
                continue

            X0 = self.X.loc[(self.y.to_numpy() == 0), catf]
            X1 = self.X.loc[(self.y.to_numpy() == 1), catf]
            n0 = X0.shape[0]
            n1 = X1.shape[0]

            ube = 0
            for cat in categories:
                pX_p0givenX = p0 * ((X0.loc[(X0 == cat)].shape[0])/n0)
                pX_p1givenX = p1 * ((X1.loc[(X1 == cat)].shape[0])/n1)
                ube += max(pX_p0givenX, pX_p1givenX)
            ube_scores.append(round(ube, DP['ROUND']))
        return ube_scores

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
            fs.fit(self.X[catfl], np.ravel(self.y))
            relevant_pvals = fs.pvalues_[fs.get_support(indices=True)]
            print(f'categorical features pvalues: {-np.log10(relevant_pvals)}')
            return fs.transform(self.X[catfl]),\
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

        def snr_selection(self, confl=confl, k=k_con):
            snr_scores = self.s2nr(confl)
            features_index = range(len(confl))
            sorted_snr = sorted(snr_scores, reverse=True)
            best_features = [x for _,x in sorted(zip(snr_scores,confl), reverse=True)]
            best_features_index = [x for _,x in sorted(zip(snr_scores,features_index), reverse=True)]
            
            # Plot results
            plt.figure()
            plt.bar(best_features, sorted_snr, width=.2, label=r'SNR Values')
            plt.xticks(rotation=45)
            plt.axis('tight')
            plt.legend(loc='upper right')
            plt.savefig(FP['SNR'])

            # transform and return the data
            best_feature_cols = [confl[i] for i in sorted(best_features_index[:k])]
            return self.X[best_feature_cols], sorted(best_features_index[:k])

        def ube_selection(self, catfl=catfl, k=k_cat):
            ube_scores = self.univariate_bayes_error(catfl)
            features_index = range(len(catfl))
            sorted_ube = sorted(ube_scores, reverse=True)
            print(sorted_ube)
            best_features = [x for _,x in sorted(zip(ube_scores,catfl), reverse=True)]
            best_features_index = [x for _,x in sorted(zip(ube_scores,features_index), reverse=True)]

            # Plot results
            plt.figure()
            plt.bar(best_features, sorted_ube, label=r'UBE Scores')
            plt.xticks(rotation=45)
            plt.axis('tight')
            plt.legend(loc='upper right')
            plt.savefig(FP['UBE'])

            # transform and return the data
            best_feature_cols = [catfl[i] for i in sorted(best_features_index[:k])]
            return self.X[best_feature_cols], sorted(best_features_index[:k])

        # Select Categorical Features
        if cat_mode is 'chi2':
            X_cat, cat_fs = chi_square_selection(self, catfl, k_cat)
        elif cat_mode is 'mi':
            X_cat, cat_fs = mutual_info_selection(self, catfl, k_cat)
        elif cat_mode is 'ube':
            X_cat, cat_fs = ube_selection(self, catfl, k_cat)

        # Select Continuous features
        if con_mode is 'anova_f':
            X_con, con_fs = anova_f_selection(self, confl, k_con)
        elif con_mode is 'snr':
            X_con, con_fs = snr_selection(self, confl, k_con)

        # print(X_cat.shape, X_con.shape)
        return np.concatenate((X_cat, X_con), axis=1), [cat_fs, con_fs]

    def visualize(self, ll, fn):
        plt.figure()
        sns.pairplot(self.data[ll])
        plt.savefig(fn)
        return None

    def correlation_matrix(self):
        cor = round(self.data.corr(), DP['ROUND'])
        cor.to_csv(FP['CORR'])
        return cor
    
    def heatmap(self, fl):
        plt.figure()
        sns.heatmap(self.data[fl].corr(), annot=True, fmt='.1g', square=True)
        plt.xticks(rotation=45)
        plt.savefig(FP['COR_HMP'])

