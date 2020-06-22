from joblib import load
from sklearn.metrics import accuracy_score

from .ml.svm import SVM
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
    X, y = dl.read_data(fl=DP['FL'], ll=DP['LL'])

    # Stratified Splitting of the dataset
    X_tr, X_t, y_tr, y_t = dl.stratified_split(X, y, DP['SR'], DP['SEED'])

    # instantiate data analytics
    da = DataAnalytics(X, y)

    # Obtain analytics
    print(da.p_y())


    # # Train SVM classifiers
    # print(args.trainSVM)
    # if (args.trainSVM):
    #     SVM(X_tr, y_tr)
    
    # # Load Saved SVM models and measure accuracies
    # rbf_svm_mdl = load(FP['SVM_RBF'])
    # y_p = rbf_svm_mdl.predict(X_t)
    # print(f'RBF Kernel SVM Accuracy: {accuracy_score(y_p, y_t)}')
    
    # linear_svm_mdl = load(FP['SVM_LINEAR'])
    # y_p = linear_svm_mdl.predict(X_t)
    # print(f'RBF Linear SVM Accuracy: {accuracy_score(y_p, y_t)}')
    return 0

if __name__ == "__main__":
    main()