import numpy as np
from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from .ml.svm import SVM
from .ml.random_forest import random_forest
from .ml.ffnn import train_ffnn, test_ffnn
from .utilities.config import FP, DP
from .data.data_loader import DataLoader
from .data.data_analytics import DataAnalytics
from .utilities.argparser import parse_arguments

def tn(y_pred, y_true): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_pred, y_true): return confusion_matrix(y_true, y_pred)[0, 1]
def specificity(y_pred, y_true): return round(tn(y_true, y_pred)/ (tn(y_true, y_pred) + fp(y_true, y_pred)), DP['ROUND'])

def main():
    # Command line argument parsing
    args = parse_arguments()

    # Instantiate Data Loader and Analytics
    dl = DataLoader(FP['DATA'])

    # Load the data set
    X, y = dl.read_data(fl=(DP['CATFL']+DP['CONFL']), ll=DP['LL'])

    X_fs = X
    if (args.dataAnalytics):
        # instantiate data analytics
        da = DataAnalytics(dl.data, X, y)
        da.heatmap(DP['CONFL'])

        # print(f'Number of data points in class 1: {da.p_y()}')
        # print('\n---------------**********---------------\n')

        # # print(f'Auto Correlation Matrix: {da.correlation_matrix()}')
        # # print('\n---------------**********---------------\n')


        # print(f'SNR for numerical variables:\n')
        # print(da.s2nr(confl=DP['CONFL']))
        # print('\n---------------**********---------------\n')

        print(f'UBE for categorical variables:\n')
        print(da.univariate_bayes_error(DP['CATFL']))
        print('\n---------------**********---------------\n')

        X_fs, fs = da.select_features(DP['CATFL'], DP['CONFL'],\
                                        k_cat=DP['NUM_CAT'],\
                                        k_con=DP['NUM_CON'],\
                                        cat_mode=DP['CAT_FS_MODE'],\
                                        con_mode=DP['CON_FS_MODE'])
        print(f'Selected Categorical Features:\n')
        print(np.array(DP['CATFL'])[fs[0]])

        print(f'\nSelected Numerical Features:')
        print(np.array(DP['CONFL'])[fs[1]])
        print('\n---------------**********---------------\n')

    #     da.visualize(np.array(DP['CATFL'])[fs[0]], FP['CAT_GRAPH'])
    #     da.visualize(np.array(DP['CONFL'])[fs[1]], FP['CON_GRAPH'])

    # # Stratified Splitting of the dataset
    # X_tr, X_t, y_tr, y_t = dl.stratified_split(X_fs, y, DP['SR'], DP['SEED'])
    # print(X_tr.shape, X_t.shape)

    # if (args.trainSVM):
    #     SVM(X_tr, y_tr, 'rbf', FP['SVM_RBF'])
    #     SVM(X_tr, y_tr, 'linear', FP['SVM_LINEAR'])

    # # Load Saved SVM models and evaluate model
    # rbf_svm_mdl = load(FP['SVM_RBF'])
    # y_p = rbf_svm_mdl.predict(X_t)
    # print(f'RBF Kernel SVM Accuracy: {accuracy_score(y_p, y_t)}')
    # print(f'RBF Kernel SVM Precision: {round(precision_score(y_p, y_t),3)}')
    # print(f'RBF Kernel SVM Sensitivity/Recall: {round(recall_score(y_p, y_t),3)}')
    # print(f'RBF Kernel SVM Specificity/Selectivity: {specificity(y_p, y_t)}')
    # print(f'RBF Kernel SVM F1: {round(f1_score(y_p, y_t),3)}')
    # print('\n---------------**********---------------\n')

    # linear_svm_mdl = load(FP['SVM_LINEAR'])
    # y_p = linear_svm_mdl.predict(X_t)
    # print(f'Linear Kernel SVM Accuracy: {accuracy_score(y_p, y_t)}')
    # print(f'Linear Kernel SVM Precision: {round(precision_score(y_p, y_t),3)}')
    # print(f'Linear Kernel SVM Sensitivity/Recall: {round(recall_score(y_p, y_t),3)}')
    # print(f'Linear Kernel SVM Specificity/Selectivity: {specificity(y_p, y_t)}')
    # print(f'Linear Kernel SVM F1 Score: {round(f1_score(y_p, y_t),3)}')
    # print('\n---------------**********---------------\n')
    
    # if (args.trainRF):
    #     feature_scores = random_forest(X_tr, y_tr, FP['RFMDL'])

    #     # Load Random Forest Classifier
    #     rfc = load(FP['RFMDL'])
    #     y_p = rfc.predict(X_t)
    #     print(f'RFC Accuracy: {accuracy_score(y_p, y_t)}')
    #     print(f'RFC Precision: {round(precision_score(y_p, y_t),3)}')
    #     print(f'RFC Sensitivity/Recall: {round(recall_score(y_p, y_t),3)}')
    #     print(f'RFC Specificity/Selectivity: {specificity(y_p, y_t)}')
    #     print(f'RFC F1: {round(f1_score(y_p, y_t),3)}')
    #     print('\n---------------**********---------------\n')

    # if (args.trainNN):
    #     train_ffnn(X_tr, y_tr, './saved_models/NN/FFNN1.joblib')
    #     test_ffnn(X_t, y_t, './saved_models/NN/FFNN1.joblib')

    return 0

if __name__ == "__main__":
    main()