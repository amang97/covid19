import itertools
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
from .utilities.evaluation_metrics import EvaluationMetrics

def main():
    # Command line argument parsing
    args = parse_arguments()

    # Instantiate Data Loader and Analytics
    dl = DataLoader(FP['DATA'])

    # Load the data set
    X, y = dl.read_data(fl=(DP['CATFL']+DP['CONFL']), ll=DP['LL'])

    X_fs = X
    train_sets = []
    if (args.dataAnalytics):
        # instantiate data analytics
        da = DataAnalytics(dl.data, X, y)
        da.heatmap(DP['CONFL'])

        print(f'Number of data points in class 1: {da.p_y()}')
        print('\n---------------**********---------------\n')

        # # print(f'Auto Correlation Matrix: {da.correlation_matrix()}')
        # # print('\n---------------**********---------------\n')


        print(f'SNR for numerical variables:\n')
        print(da.s2nr(confl=DP['CONFL']))
        print('\n---------------**********---------------\n')

        print(f'UBE for categorical variables:\n')
        # print(da.univariate_bayes_error(DP['CATFL']))
        print('\n---------------**********---------------\n')

        X_fs, fs = da.select_features(DP['CATFL'], DP['CONFL'],\
                                        k_cat=DP['NUM_CAT'],\
                                        k_con=DP['NUM_CON'],\
                                        cat_mode=DP['CAT_FS_MODE'],\
                                        con_mode=DP['CON_FS_MODE'])
        fs_combs = list(itertools.combinations(fs[0][DP['NUM_FS_FIXED']:], DP['NUM_FS_COMB']))

        for i,p in enumerate(list(fs_combs)):
            # print(f'Selected Categorical Features @ permutation {i}:\n')
            cat_fs_ = np.array(DP['CATFL'])[sorted(list(fs[0][0:DP['NUM_FS_FIXED']]) + list(p))]
            con_fs_ = np.array(DP['CONFL'])[fs[1]]
            print(cat_fs_, con_fs_)
            train_sets.append(X[np.concatenate((cat_fs_, con_fs_))])
            
        print(f'\nSelected Categorical Features:')
        print(np.array(DP['CATFL'])[fs[0]])
        print('\n---------------**********---------------\n')

        print(f'\nSelected Numerical Features:')
        print(fs[1])
        print(np.array(DP['CONFL'])[fs[1]])
        print('\n---------------**********---------------\n')

    rbfSvm_metrics = EvaluationMetrics()
    linearSvm_metrics = EvaluationMetrics()
    rfc_metrics = EvaluationMetrics()
    for j, X_fs in enumerate(train_sets):
        # Stratified Splitting of the dataset
        X_tr, X_t, y_tr, y_t = dl.stratified_split(X_fs, y, DP['SR'], DP['SEED'])
        # print(X_tr.shape, X_t.shape)

        if (args.trainSVM):
            SVM(X_tr, y_tr, 'rbf', FP['SVM_RBF']+str(j)+FP['MODEL_EXT'])
            SVM(X_tr, y_tr, 'linear', FP['SVM_LINEAR']+str(j)+FP['MODEL_EXT'])
            print("training SVM completed")

        if (args.testSVM):
            # Load Saved SVM models and evaluate model
            rbf_svm_mdl = load(FP['SVM_RBF']+str(j)+FP['MODEL_EXT'])
            y_p = rbf_svm_mdl.predict(X_t)
            rbfSvm_metrics.evaluate_all_metrics(y_p, y_t, DP['ROUND'])

            linear_svm_mdl = load(FP['SVM_LINEAR']+str(j)+FP['MODEL_EXT'])
            y_p = linear_svm_mdl.predict(X_t)
            linearSvm_metrics.evaluate_all_metrics(y_p, y_t, DP['ROUND'])
        
        if (args.trainRF):
            feature_scores = random_forest(X_tr, y_tr, FP['RFMDL']+str(j)+FP['MODEL_EXT'])
            print("RF training completed")

        if (args.testRF):
            # Load Random Forest Classifier
            rfc = load(FP['RFMDL']+str(j)+FP['MODEL_EXT'])
            y_p = rfc.predict(X_t)
            rfc_metrics.evaluate_all_metrics(y_p, y_t, DP['ROUND'])

        if (args.trainNN):
            train_ffnn(X_tr, y_tr, './saved_models/NN/FFNN1.joblib')
            test_ffnn(X_t, y_t, './saved_models/NN/FFNN1.joblib')

    if (args.testSVM): 
        print('RBF Kernel:\n')
        print(rbfSvm_metrics)
        print(rbfSvm_metrics.max_metrics())
        rbfSvm_metrics.visualize('rbf_svm')
        print('\n')

        print('Linear Kernel:\n')
        print(linearSvm_metrics)
        print(linearSvm_metrics.max_metrics())
        linearSvm_metrics.visualize('linear_svm')
        print('\n')

    if (args.testRF):
        print('RFC:\n')
        print(rfc_metrics)
        print(rfc_metrics.max_metrics())
        rfc_metrics.visualize('rfc_svm')
        print('\n')

    return 0

if __name__ == "__main__":
    main()