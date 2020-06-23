from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .ml.svm import SVM
from .ml.random_forest import random_forest
from .utilities.config import FP, DP
from .data.data_loader import DataLoader
from .data.data_analytics import DataAnalytics
from .utilities.argparser import parse_arguments

def main():
    # Command line argument parsing
    args = parse_arguments()

    # Instantiate Data Loader and Analytics
    dl = DataLoader(FP['DATA'])

    # Load the data set
    X, y = dl.read_data(fl=(DP['CATFL']+ DP['CONFL']), ll=DP['LL'])

    # # Stratified Splitting of the dataset
    # X_tr, X_t, y_tr, y_t = dl.stratified_split(X, y, DP['SR'], DP['SEED'])

    # instantiate data analytics
    da = DataAnalytics(X, y)

    X_fs, fs = da.select_features(DP['CATFL'], DP['CONFL'],\
                                    k_cat=5, k_con=3,\
                                    cat_mode='chi2',\
                                    con_mode='anova_f')
    print(fs)
    print('\n---------------**********---------------\n')
    X_tr, X_t, y_tr, y_t = dl.stratified_split(X_fs, y, DP['SR'], DP['SEED'])

    # # Obtain analytics
    # print(da.p_y())

    if (args.trainSVM):
        SVM(X_tr, y_tr, 'linear', './saved_models/SVM/linear.joblib')
        SVM(X_tr, y_tr, 'rbf', './saved_models/SVM/rbf.joblib')
    
    if (args.trainRF):
        random_forest(X_tr, y_tr, './saved_models/RF/rf.joblib')

    # Load Saved SVM models and measure accuracies
    rbf_svm_mdl = load('./saved_models/SVM/rbf.joblib')
    y_p = rbf_svm_mdl.predict(X_t)
    print(f'RBF Kernel SVM Accuracy: {accuracy_score(y_p, y_t)}')
    print(f'RBF Kernel SVM Precision: {precision_score(y_p, y_t)}')
    print(f'RBF Kernel SVM Recall: {recall_score(y_p, y_t)}')
    print(f'RBF Kernel SVM F1: {f1_score(y_p, y_t)}')
    print('\n---------------**********---------------\n')

    linear_svm_mdl = load('./saved_models/SVM/linear.joblib')
    y_p = linear_svm_mdl.predict(X_t)
    print(f'Linear Kernel SVM Accuracy: {accuracy_score(y_p, y_t)}')
    print(f'Linear Kernel SVM Precision: {precision_score(y_p, y_t)}')
    print(f'Linear Kernel SVM Recall: {recall_score(y_p, y_t)}')
    print(f'Linear Kernel SVM F1 Score: {f1_score(y_p, y_t)}')
    print('\n---------------**********---------------\n')

    # Load Random Forest Classifier
    rfc = load('./saved_models/RF/rf.joblib')
    y_p = rfc.predict(X_t)
    print(f'RFC Accuracy: {accuracy_score(y_p, y_t)}')
    print(f'RFC Precision: {precision_score(y_p, y_t)}')
    print(f'RFC Recall: {recall_score(y_p, y_t)}')
    print(f'RFC F1: {f1_score(y_p, y_t)}')
    print('\n---------------**********---------------\n')

    return 0

if __name__ == "__main__":
    main()